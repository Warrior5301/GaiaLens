from __future__ import annotations

import base64
import os
import re
import threading
import time
import textwrap
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import ee
import requests
from ddgs import DDGS
import trafilatura
from geoclip_client import GeoCLIPGeolocator


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

EE_PROJECT = os.getenv("EE_PROJECT")
OUTPUT_DIR = Path(os.getenv("LOCATION_EXPLORER_OUTPUT_DIR", "./outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NOMINATIM_ENDPOINT = os.getenv(
    "NOMINATIM_ENDPOINT",
    "https://nominatim.openstreetmap.org/reverse",
)
NOMINATIM_USER_AGENT = os.getenv(
    "NOMINATIM_USER_AGENT",
    "remote-sensing-vlm-location-explorer/0.1",
)
NOMINATIM_TIMEOUT = float(os.getenv("NOMINATIM_TIMEOUT", "15"))
NOMINATIM_MIN_INTERVAL_SECONDS = float(
    os.getenv("NOMINATIM_MIN_INTERVAL_SECONDS", "1.1")
)

WEB_MAX_RESULTS = int(os.getenv("WEB_MAX_RESULTS", "6"))
WEB_FETCH_TOP_PAGES = int(os.getenv("WEB_FETCH_TOP_PAGES", "3"))
WEB_PAGE_TIMEOUT = float(os.getenv("WEB_PAGE_TIMEOUT", "15"))
SATELLITE_THUMBNAIL_DIMENSIONS = int(
    os.getenv("SATELLITE_THUMBNAIL_DIMENSIONS", "1536")
)

# One best-available imagery tier is returned. NAIP is used where available;
# Sentinel-2, Landsat, and MODIS are fallback sources.
LOOKBACK_DAYS = int(os.getenv("SATELLITE_LOOKBACK_DAYS", "365"))
CLOUD_LIMIT = float(os.getenv("S2_CLOUD_LIMIT", "30"))


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class LandmarkCandidate:
    name: str
    osm_type: str
    osm_id: int
    lat: float
    lon: float
    distance_m: float
    score: float
    prominence_score: float
    type_score: float
    wikipedia: str | None
    wikidata: str | None
    tags: dict[str, str]


@dataclass
class SatelliteImageResult:
    source: str
    kind: str
    gsd_m: float
    visualization: str
    acquisition_date: str | None
    image_url: str
    local_path: str | None
    public_url: str | None
    footprint_radius_m: float


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str
    source_type: str
    extracted_text: str | None = None


# -----------------------------------------------------------------------------
# Earth Engine
# -----------------------------------------------------------------------------

class EarthEngineClient:
    """Retrieves small RGB/spectral thumbnails around a coordinate."""

    def __init__(self, project: str | None = None):
        self.project = project or EE_PROJECT
        self._initialized = False

    def initialize(self, interactive_auth: bool = False) -> None:
        if self._initialized:
            return

        if interactive_auth:
            ee.Authenticate()

        if not self.project:
            raise RuntimeError(
                "EE_PROJECT is not set. Create/select an Earth Engine Cloud project "
                "and set EE_PROJECT before initializing."
            )

        try:
            ee.Initialize(project=self.project)
        except Exception as exc:
            raise RuntimeError(
                "Earth Engine initialization failed. Authenticate first (for example "
                "with ee.Authenticate() in Colab) and make sure EE_PROJECT is valid."
            ) from exc

        self._initialized = True

    @staticmethod
    def _point_and_region(lat: float, lon: float, radius_m: float) -> tuple[ee.Geometry, ee.Geometry]:
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(radius_m).bounds()
        return point, region

    @staticmethod
    def _safe_first(collection: ee.ImageCollection) -> ee.Image | None:
        try:
            if collection.size().getInfo() == 0:
                return None
            return ee.Image(collection.first())
        except Exception:
            return None

    def _latest_s2(self, region: ee.Geometry) -> tuple[ee.Image | None, str | None]:
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=LOOKBACK_DAYS)

        base = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(str(start), str(end))
        )

        filtered = base.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", CLOUD_LIMIT))
        image = self._safe_first(filtered.sort("system:time_start", False))
        if image is None:
            image = self._safe_first(base.sort("system:time_start", False))
        if image is None:
            return None, None

        date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()
        return image, date

    def _latest_landsat(self, region: ee.Geometry) -> tuple[ee.Image | None, str | None]:
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=LOOKBACK_DAYS * 2)

        l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        collection = (
            l8.merge(l9)
            .filterBounds(region)
            .filterDate(str(start), str(end))
            .filter(ee.Filter.lte("CLOUD_COVER", 35))
        )
        image = self._safe_first(collection.sort("system:time_start", False))
        if image is None:
            return None, None

        date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()
        return image, date

    def _latest_modis(self, region: ee.Geometry) -> tuple[ee.Image | None, str | None]:
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=120)
        collection = (
            ee.ImageCollection("MODIS/061/MOD09GA")
            .filterBounds(region)
            .filterDate(str(start), str(end))
            .sort("system:time_start", False)
        )
        image = self._safe_first(collection)
        if image is None:
            return None, None

        date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()
        return image, date

    def _latest_naip(self, region: ee.Geometry) -> tuple[ee.Image | None, str | None]:
        # NAIP is continental-US aerial imagery, not satellite imagery.
        collection = (
            ee.ImageCollection("USDA/NAIP/DOQQ")
            .filterBounds(region)
            # NAIP assets vary by acquisition year: some have RGBN, others RGB.
            # Normalize to the shared RGB bands before mosaicking adjacent tiles.
            .select(["R", "G", "B"])
        )
        latest = self._safe_first(collection.sort("system:time_start", False))
        if latest is None:
            return None, None

        # NAIP is stored as adjacent image tiles. Selecting only the first tile
        # can leave most of the requested region masked (black in the thumbnail).
        # Mosaic all intersecting tiles, with newer acquisitions taking priority.
        image = collection.sort("system:time_start").mosaic()
        date = ee.Date(latest.get("system:time_start")).format("YYYY-MM-dd").getInfo()
        return image, date

    @staticmethod
    def _thumbnail_url(
        image: ee.Image,
        region: ee.Geometry,
        *,
        bands: list[str],
        min_value: float | list[float],
        max_value: float | list[float],
        dimensions: int = SATELLITE_THUMBNAIL_DIMENSIONS,
    ) -> str:
        visual = image.select(bands).visualize(
            min=min_value,
            max=max_value,
            bands=bands,
            forceRgbOutput=True,
        )
        return visual.getThumbURL({
            "region": region,
            "dimensions": dimensions,
            "format": "jpg",
            "crs": "EPSG:4326",
        })

    @staticmethod
    def _download(url: str, path: Path) -> None:
        response = requests.get(
            url,
            timeout=60,
            headers={"User-Agent": "remote-sensing-vlm-location-explorer/0.1"},
        )
        if not response.ok:
            # Earth Engine includes the specific thumbnail computation error
            # in the response body; retain a bounded excerpt for diagnostics.
            detail = " ".join(response.text.split())[:1200]
            raise RuntimeError(
                f"Earth Engine thumbnail download failed (HTTP {response.status_code}): {detail}"
            )
        content_type = response.headers.get("content-type", "")
        if "image" not in content_type:
            raise RuntimeError(f"Earth Engine thumbnail did not return an image: {content_type}")
        path.write_bytes(response.content)

    def retrieve(self, lat: float, lon: float, radius_m: float = 3000.0) -> list[SatelliteImageResult]:
        if not self._initialized:
            self.initialize(interactive_auth=False)

        _, region = self._point_and_region(lat, lon, radius_m)
        run_dir = OUTPUT_DIR / uuid.uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=True)

        # Return one best-available true-color image. NAIP provides the finest
        # resolution where available; otherwise use Sentinel-2, Landsat, then MODIS.
        naip, naip_date = self._latest_naip(region)
        if naip is not None:
            url = self._thumbnail_url(
                naip,
                region,
                bands=["R", "G", "B"],
                min_value=0,
                max_value=255,
            )
            path = run_dir / "01_naip_0p6m.jpg"
            self._download(url, path)
            return [
                SatelliteImageResult(
                    source="USDA NAIP",
                    kind="aerial_orthophoto",
                    gsd_m=0.6,
                    visualization="true_color",
                    acquisition_date=naip_date,
                    image_url=url,
                    local_path=str(path),
                    public_url=f"/outputs/{run_dir.name}/{path.name}",
                    footprint_radius_m=radius_m,
                )
            ]

        s2, s2_date = self._latest_s2(region)
        if s2 is not None:
            url = self._thumbnail_url(
                s2.multiply(0.0001),
                region,
                bands=["B4", "B3", "B2"],
                min_value=0.0,
                max_value=0.30,
            )
            path = run_dir / "02_sentinel2_10m_true_color.jpg"
            self._download(url, path)
            return [
                SatelliteImageResult(
                    source="Sentinel-2 SR Harmonized",
                    kind="satellite",
                    gsd_m=10.0,
                    visualization="true_color",
                    acquisition_date=s2_date,
                    image_url=url,
                    local_path=str(path),
                    public_url=f"/outputs/{run_dir.name}/{path.name}",
                    footprint_radius_m=radius_m,
                )
            ]

        landsat, landsat_date = self._latest_landsat(region)
        if landsat is not None:
            scaled = landsat.select(["SR_B4", "SR_B3", "SR_B2"]).multiply(0.0000275).add(-0.2)
            url = self._thumbnail_url(
                scaled,
                region,
                bands=["SR_B4", "SR_B3", "SR_B2"],
                min_value=0.0,
                max_value=0.30,
            )
            path = run_dir / "03_landsat_30m_true_color.jpg"
            self._download(url, path)
            return [
                SatelliteImageResult(
                    source="Landsat 8/9 Collection 2 Level 2",
                    kind="satellite",
                    gsd_m=30.0,
                    visualization="true_color",
                    acquisition_date=landsat_date,
                    image_url=url,
                    local_path=str(path),
                    public_url=f"/outputs/{run_dir.name}/{path.name}",
                    footprint_radius_m=radius_m,
                )
            ]

        modis, modis_date = self._latest_modis(region)
        if modis is None:
            return []
        scaled = modis.select(
            ["sur_refl_b01", "sur_refl_b04", "sur_refl_b03"]
        ).multiply(0.0001)
        url = self._thumbnail_url(
            scaled,
            region,
            bands=["sur_refl_b01", "sur_refl_b04", "sur_refl_b03"],
            min_value=0.0,
            max_value=0.35,
        )
        path = run_dir / "05_modis_500m_true_color.jpg"
        self._download(url, path)
        return [
            SatelliteImageResult(
                source="MODIS Terra MOD09GA.061",
                kind="satellite",
                gsd_m=500.0,
                visualization="true_color",
                acquisition_date=modis_date,
                image_url=url,
                local_path=str(path),
                public_url=f"/outputs/{run_dir.name}/{path.name}",
                footprint_radius_m=radius_m,
            )
        ]


# -----------------------------------------------------------------------------
# Nominatim reverse geocoding
# -----------------------------------------------------------------------------

class NominatimReverseGeocoder:
    """Resolve one GeoCLIP point to the nearest suitable OSM address/place."""

    _request_lock = threading.Lock()
    _last_request_started = 0.0
    _cache: dict[tuple[float, float], dict[str, Any]] = {}

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": NOMINATIM_USER_AGENT,
                "Accept": "application/json",
            }
        )

    @staticmethod
    def _to_location_metadata(payload: dict[str, Any]) -> dict[str, Any]:
        address = payload.get("address") or {}
        return {
            "source": "Nominatim",
            "display_name": payload.get("display_name"),
            "name": payload.get("name"),
            "category": payload.get("category") or payload.get("class"),
            "type": payload.get("type"),
            "addresstype": payload.get("addresstype"),
            "latitude": float(payload["lat"]) if payload.get("lat") else None,
            "longitude": float(payload["lon"]) if payload.get("lon") else None,
            "country": address.get("country"),
            "state": address.get("state") or address.get("region"),
            "city": (
                address.get("city")
                or address.get("town")
                or address.get("village")
                or address.get("municipality")
            ),
            "district": (
                address.get("city_district")
                or address.get("district")
                or address.get("county")
                or address.get("state_district")
            ),
            "postcode": address.get("postcode"),
            "address": address,
            "osm_type": payload.get("osm_type"),
            "osm_id": payload.get("osm_id"),
            "licence": payload.get("licence"),
        }

    @staticmethod
    def _location_hint(location: dict[str, Any]) -> str | None:
        address = location.get("address") or {}
        keys = (
            "neighbourhood",
            "suburb",
            "city_district",
            "city",
            "town",
            "village",
            "municipality",
            "county",
            "state",
            "country",
        )
        parts = list(dict.fromkeys(address[key] for key in keys if address.get(key)))
        return ", ".join(parts) or location.get("display_name")

    def reverse(
        self, lat: float, lon: float
    ) -> tuple[dict[str, Any] | None, str | None]:
        cache_key = (round(lat, 5), round(lon, 5))
        cls = type(self)

        with cls._request_lock:
            cached = cls._cache.get(cache_key)
            if cached is not None:
                return cached, None

            wait_seconds = NOMINATIM_MIN_INTERVAL_SECONDS - (
                time.monotonic() - cls._last_request_started
            )
            if wait_seconds > 0:
                time.sleep(wait_seconds)

            cls._last_request_started = time.monotonic()
            try:
                response = self.session.get(
                    NOMINATIM_ENDPOINT,
                    params={
                        "lat": lat,
                        "lon": lon,
                        "format": "jsonv2",
                        "zoom": 18,
                        "addressdetails": 1,
                        "extratags": 1,
                        "namedetails": 1,
                    },
                    timeout=NOMINATIM_TIMEOUT,
                )
                response.raise_for_status()
                payload = response.json()
                if payload.get("error"):
                    raise RuntimeError(str(payload["error"]))

                location = self._to_location_metadata(payload)
                cls._cache[cache_key] = location
                if len(cls._cache) > 2048:
                    cls._cache.pop(next(iter(cls._cache)))
                return location, None
            except Exception as exc:
                return None, f"{type(exc).__name__}: {exc}"


# -----------------------------------------------------------------------------
# Web context search
# -----------------------------------------------------------------------------

class WebContextSearcher:
    """Searches the web for context about a landmark or coordinate."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "remote-sensing-vlm-location-explorer/0.1",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        self.search_errors: list[str] = []
        self.last_status = "Search has not run."

    @staticmethod
    def _clean_title(title: str) -> str:
        return re.sub(r"\s+", " ", title or "").strip()

    @staticmethod
    def _compact_search_text(text: str) -> str:
        compact = re.sub(r"\s+", " ", text or "").strip()
        sentences = re.split(r"(?<=[.!?])\s+", compact)
        return "\n".join(
            textwrap.shorten(sentence, width=220, placeholder="…")
            for sentence in sentences[:6]
            if sentence.strip()
        )

    def _search(
        self, query: str, mode: str = "text", max_results: int = WEB_MAX_RESULTS
    ) -> list[dict[str, Any]]:
        try:
            with DDGS() as ddgs:
                if mode == "news":
                    results = list(ddgs.news(query, max_results=max_results))
                else:
                    results = list(ddgs.text(query, max_results=max_results))
            return results
        except Exception as exc:
            self.search_errors.append(f"{mode} search: {type(exc).__name__}: {exc}")
            return []

    def _extract_page(self, url: str) -> str | None:
        try:
            response = self.session.get(url, timeout=WEB_PAGE_TIMEOUT, allow_redirects=True)
            response.raise_for_status()
            extracted = trafilatura.extract(
                response.text,
                include_comments=False,
                include_tables=False,
            )
            if not extracted:
                return None

            # Avoid sending long article histories when the page has a distinct
            # History section. Keep at most six short, readable context lines.
            extracted = re.split(
                r"(?im)^\s*History\s*(?:\[\s*edit\s*\])?\s*$",
                extracted,
                maxsplit=1,
            )[0]
            page_lines = [
                line.strip()
                for line in extracted.splitlines()
                if line.strip() and line.strip().lower() != "[edit]"
            ]
            if len(page_lines) > 1 and page_lines[1].startswith(page_lines[0]):
                page_lines = page_lines[1:]
            compact_text = re.sub(r"\s+", " ", " ".join(page_lines)).strip()
            if not compact_text:
                return None

            sentences = re.split(r"(?<=[.!?])\s+", compact_text)
            context_lines = [
                textwrap.shorten(sentence, width=220, placeholder="…")
                for sentence in sentences[:6]
                if sentence.strip()
            ]
            return "\n".join(context_lines) or None
        except Exception:
            return None

    def search_landmark(self, landmark: LandmarkCandidate) -> list[WebResult]:
        query = f'"{landmark.name}" landmark location history significance'
        news_query = f'"{landmark.name}" latest news'

        raw_results = self._search(query, mode="text")
        raw_news = self._search(news_query, mode="news")

        combined: list[tuple[dict[str, Any], str]] = []
        for item in raw_results:
            combined.append((item, "web"))
        for item in raw_news:
            combined.append((item, "news"))

        results: list[WebResult] = []
        seen_urls: set[str] = set()

        for item, source_type in combined:
            url = item.get("href") or item.get("url")
            title = item.get("title") or ""
            snippet = item.get("body") or item.get("snippet") or ""
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            results.append(
                WebResult(
                    title=self._clean_title(title),
                    url=url,
                    snippet=re.sub(r"\s+", " ", snippet).strip(),
                    source_type=source_type,
                )
            )
            if len(results) >= WEB_MAX_RESULTS:
                break

        # Fetch a few pages for richer context, but retain the original URL/snippet even
        # if extraction fails. This gives the frontend both evidence and provenance.
        for result in results[:WEB_FETCH_TOP_PAGES]:
            result.extracted_text = self._extract_page(result.url)

        return results

    def search_coordinates(
        self, lat: float, lon: float, location_hint: str | None = None
    ) -> list[WebResult]:
        """Search DuckDuckGo with coordinates and an optional admin-area hint."""
        self.search_errors.clear()
        coordinate_plain = f"{lat:.6f} {lon:.6f}"
        lat_hemi = "N" if lat >= 0 else "S"
        lon_hemi = "E" if lon >= 0 else "W"
        coordinate_dms_style = (
            f"{abs(lat):.5f} {lat_hemi} {abs(lon):.5f} {lon_hemi}"
        )
        area = location_hint or ""
        queries = [
            f"{coordinate_plain} {area} location city place",
            f"what is at latitude {lat:.6f} longitude {lon:.6f} {area}",
            f"places and geography near {coordinate_dms_style} {area}",
        ]

        raw_results: list[dict[str, Any]] = []
        for query in queries:
            raw_results = self._search(query, mode="text", max_results=1)
            if raw_results:
                break

        results: list[WebResult] = []
        seen_urls: set[str] = set()
        for item in raw_results:
            url = item.get("href") or item.get("url")
            title = item.get("title") or ""
            snippet = item.get("body") or item.get("snippet") or ""
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            results.append(
                WebResult(
                    title=self._clean_title(title),
                    url=url,
                    snippet=self._compact_search_text(snippet),
                    source_type="coordinate_search",
                )
            )
            if results:
                break

        if results:
            self.last_status = (
                "DuckDuckGo returned the first result for coordinates"
                + (f" in {location_hint}." if location_hint else ".")
            )
        elif self.search_errors:
            self.last_status = "DuckDuckGo search errors: " + " | ".join(self.search_errors)
        else:
            self.last_status = (
                "DuckDuckGo returned no indexed results for the coordinate-based queries."
            )

        return results


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------

class LocationExplorer:
    def __init__(self):
        self.geolocator = GeoCLIPGeolocator()
        self.ee = EarthEngineClient()
        self.reverse_geocoder = NominatimReverseGeocoder()
        self.web = WebContextSearcher()

    def explore_image(
        self,
        image_bytes: bytes,
        imagery_radius_m: float = 3000.0,
        top_k: int = 5,
    ) -> dict[str, Any]:
        predictions = self.geolocator.predict(image_bytes, top_k=top_k)
        primary = max(predictions, key=lambda item: item["probability"])
        result = self.explore(
            primary["latitude"],
            primary["longitude"],
            imagery_radius_m=imagery_radius_m,
        )
        result["geolocation_candidates"] = predictions
        result["coordinates"]["source"] = "GeoCLIP"
        result["coordinates"]["probability"] = primary["probability"]
        result["coordinates"]["note"] = (
            "Coordinates are the highest-probability approximate GeoCLIP estimate."
        )
        return result

    def explore(self, lat: float, lon: float, imagery_radius_m: float = 3000.0) -> dict[str, Any]:
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Invalid latitude/longitude")

        # Retrieve imagery, reverse-geocode the point, then use the place names
        # with the existing coordinate-based web search.
        # Enforce the public response contract even if an imagery client or
        # override returns multiple candidates.
        satellite_results = self.ee.retrieve(
            lat, lon, radius_m=imagery_radius_m
        )[:1]
        location_anchor, geocoding_error = self.reverse_geocoder.reverse(lat, lon)
        location_hint = (
            self.reverse_geocoder._location_hint(location_anchor)
            if location_anchor
            else None
        )
        web_results = self.web.search_coordinates(lat, lon, location_hint)
        web_context_status = self.web.last_status
        if geocoding_error:
            web_context_status += f" Nominatim reverse geocoding failed: {geocoding_error}"

        satellite_images = []
        for result in satellite_results:
            image_data = asdict(result)
            if result.local_path:
                image_data["image_base64"] = base64.b64encode(
                    Path(result.local_path).read_bytes()
                ).decode("ascii")
                image_data["image_mime_type"] = "image/jpeg"
            else:
                image_data["image_base64"] = None
                image_data["image_mime_type"] = None
            satellite_images.append(image_data)

        return {
            # Top-level coordinates make the predicted location easy to consume;
            # retain the nested form for clients already using the original shape.
            "latitude": lat,
            "longitude": lon,
            "coordinates": {
                "latitude": lat,
                "longitude": lon,
                "note": "Coordinates are treated as approximate location estimates.",
            },
            "location_hint": location_hint,
            "location_attribution": (
                location_anchor.get("licence") if location_anchor else None
            ),
            "satellite_images": satellite_images,
            "location_anchor": location_anchor,
            "nearby_landmark_candidates": [],
            "web_context": [asdict(x) for x in web_results],
            "web_context_status": web_context_status,
        }



