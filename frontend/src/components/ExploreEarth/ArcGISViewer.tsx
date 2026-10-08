"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import Map from "@arcgis/core/Map";
import SceneView from "@arcgis/core/views/SceneView";
import SceneLayer from "@arcgis/core/layers/SceneLayer";
import Sketch from "@arcgis/core/widgets/Sketch";
import GraphicsLayer from "@arcgis/core/layers/GraphicsLayer";
import "@arcgis/core/assets/esri/themes/dark/main.css";
import FloatingChatWindow from "@/components/ExploreEarth/FloatingChatWindow";
import { Compass, Mountain, Trees, SunMedium, Landmark, Waves, Satellite } from "lucide-react";

// 3-D layer helper
const add3DLayer = (viewRef: React.RefObject<SceneView | null>, url: string) => {
  if (!viewRef.current || !viewRef.current.map || !url.trim()) return;
  const layer = new SceneLayer({ url: url.trim() });
  viewRef.current.map.add(layer);
};

interface DiscoverPlace {
  label: string;
  region: string;
  lat: number;
  lon: number;
  zoom: number;
  icon: React.ComponentType<{ className?: string }>;
}

const DISCOVER_PLACES: DiscoverPlace[] = [
  { label: "Himalayas", region: "Mt. Everest & Peaks", lat: 27.9881, lon: 86.9250, zoom: 11, icon: Mountain },
  { label: "Amazon Rainforest", region: "Amazonas, Brazil", lat: -3.4653, lon: -62.2159, zoom: 8, icon: Trees },
  { label: "Grand Canyon", region: "Arizona, USA", lat: 36.1069, lon: -112.1129, zoom: 11, icon: SunMedium },
  { label: "Pyramids of Giza", region: "Cairo, Egypt", lat: 29.9792, lon: 31.1342, zoom: 14, icon: Landmark },
  { label: "Great Barrier Reef", region: "Coral Sea, Australia", lat: -18.2871, lon: 147.6992, zoom: 8, icon: Waves },
];

interface BBox {
  west: number;
  south: number;
  east: number;
  north: number;
}

interface EEImage {
  url: string;
  year: number;
}

const BASEMAPS = [
  { id: "satellite", label: "Satellite" },
  { id: "hybrid", label: "Hybrid" },
  { id: "streets", label: "Streets" },
  { id: "terrain", label: "Terrain" },
] as const;

export default function ArcGISViewer() {
  const mapDivRef = useRef<HTMLDivElement>(null);
  const viewRef = useRef<SceneView | null>(null);
  const mapRef = useRef<Map | null>(null);
  const sketchRef = useRef<Sketch | null>(null);
  const glRef = useRef<GraphicsLayer | null>(null);

  const [isLoaded, setIsLoaded] = useState(false);
  const [coords, setCoords] = useState({ lat: 20, lon: 78, alt: 15000 });
  const [selectedBBox, setSelectedBBox] = useState<BBox | null>(null);
  const [eeImages, setEeImages] = useState<EEImage[]>([]);
  const [eeLoading, setEeLoading] = useState(false);
  const [basemap, setBasemap] = useState<string>("satellite");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchLoading, setSearchLoading] = useState(false);
  const [showSearch, setShowSearch] = useState(false);
  const [exploreOpen, setExploreOpen] = useState(false);
  const exploreTimerRef = useRef<NodeJS.Timeout | null>(null);
  const [selectMode, setSelectMode] = useState(false);
  // 3D Tiles state for ArcGIS
  const [tiles3DActive, setTiles3DActive] = useState(false);
  const [tiles3DLoading, setTiles3DLoading] = useState(false);
  const [tiles3DError, setTiles3DError] = useState<string | null>(null);
  const tiles3DRef = useRef<SceneLayer | null>(null);

  const [eeOpen, setEeOpen] = useState(false);
  const [eePinned, setEePinned] = useState(false);
  const eeTimerRef = useRef<NodeJS.Timeout | null>(null);

  const [selectedImage, setSelectedImage] = useState<EEImage | null>(null);
  const [lightboxZoom, setLightboxZoom] = useState(1);
  const [lightboxPan, setLightboxPan] = useState({ x: 0, y: 0 });
  const lightboxDragRef = useRef({ active: false, startX: 0, startY: 0, ox: 0, oy: 0 });

// Lightbox pan handling
useEffect(() => {
  const onMove = (e: MouseEvent) => {
    if (!lightboxDragRef.current.active) return;
    const dx = e.clientX - lightboxDragRef.current.startX;
    const dy = e.clientY - lightboxDragRef.current.startY;
    setLightboxPan({ x: lightboxDragRef.current.ox + dx, y: lightboxDragRef.current.oy + dy });
  };
  const onUp = () => {
    lightboxDragRef.current.active = false;
  };
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
  return () => {
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
  };
}, []);

// Lightbox interaction handlers
const handleLightboxWheel = (e: React.WheelEvent<HTMLImageElement>) => {
  e.preventDefault();
  const delta = e.deltaY;
  if (delta < 0) {
    setLightboxZoom(z => Math.min(6, +(z + 0.3).toFixed(2)));
  } else {
    setLightboxZoom(z => Math.max(1, +(z - 0.3).toFixed(2)));
  }
};

const handleLightboxMouseDown = (e: React.MouseEvent<HTMLImageElement>) => {
  e.stopPropagation();
  lightboxDragRef.current.active = true;
  lightboxDragRef.current.startX = e.clientX;
  lightboxDragRef.current.startY = e.clientY;
  lightboxDragRef.current.ox = lightboxPan.x;
  lightboxDragRef.current.oy = lightboxPan.y;
};

  // ── AI Assistant Chat Handler ──────────────────────────────────────────────
  const handleChatSend = useCallback(async (message: string, history: any[]): Promise<string> => {
    try {
      const formData = new FormData();
      let queryPrompt = message;
      if (selectedBBox) {
        queryPrompt = `[Geographic Context - BBox: South ${selectedBBox.south.toFixed(4)}, North ${selectedBBox.north.toFixed(4)}, West ${selectedBBox.west.toFixed(4)}, East ${selectedBBox.east.toFixed(4)}]\n\n${message}`;
      }
      formData.append("query", queryPrompt);

      if (eeImages.length > 0 && eeImages[0].url) {
        try {
          const res = await fetch(eeImages[0].url);
          const blob = await res.blob();
          formData.append("images", new File([blob], `sentinel2_${eeImages[0].year}.png`, { type: "image/png" }));
        } catch {
          // ignore if image fetch fails
        }
      }

      const recentTurns = history.slice(-6).map((h) => ({
        role: h.role,
        content: h.content,
      }));
      formData.append("history", JSON.stringify(recentTurns));

      const res = await fetch("/api/chat", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      return data.response || data.details || data.error || "Analysis completed.";
    } catch (err: any) {
      console.error("Explore Earth chat error:", err);
      return `⚠️ Unable to communicate with GaiaLens backend: ${err?.message || "Service unavailable"}`;
    }
  }, [selectedBBox, eeImages]);

  useEffect(() => {
    if (!mapDivRef.current) return;
    const graphicsLayer = new GraphicsLayer();
    glRef.current = graphicsLayer;
    const map = new Map({ basemap: "satellite", layers: [graphicsLayer] });
    mapRef.current = map;
    const view = new SceneView({
      container: mapDivRef.current,
      map,
      camera: { position: { latitude: 20, longitude: 78, z: 15_000_000 }, tilt: 0, heading: 0 },
      ui: { components: [] },
    });
    view.on("pointer-move", (evt) => {
      const pt = view.toMap({ x: evt.x, y: evt.y });
      if (pt && pt.latitude != null && pt.longitude != null) {
        setCoords({ lat: +pt.latitude.toFixed(4), lon: +pt.longitude.toFixed(4), alt: Math.round((view.camera?.position?.z ?? 0) / 1000) });
      }
    });
    view.when(() => setIsLoaded(true)).catch(console.error);
    viewRef.current = view;
    const sketch = new Sketch({
      view, layer: graphicsLayer, creationMode: "single", availableCreateTools: ["rectangle"],
      visibleElements: {
        createTools: { rectangle: true, point: false, polyline: false, polygon: false, circle: false },
        selectionTools: { "rectangle-selection": false, "lasso-selection": false },
        settingsMenu: false, undoRedoMenu: false,
      },
    });
    sketchRef.current = sketch;
    sketch.on("create", async (evt: any) => {
      if (evt.state === "complete") {
        // Extent is in Web Mercator (EPSG:3857); project to geographic (EPSG:4326)
        const ext = evt.graphic.geometry.extent;
        const { project, load } = await import("@arcgis/core/geometry/projection");
        const SpatialReference = (await import("@arcgis/core/geometry/SpatialReference")).default;
        await load();
        const geographicExtent = project(ext, new SpatialReference({ wkid: 4326 })) as any;
        const bbox: BBox = {
          west: geographicExtent.xmin,
          south: geographicExtent.ymin,
          east: geographicExtent.xmax,
          north: geographicExtent.ymax,
        };
        setSelectedBBox(bbox);
        setSelectMode(false);
        doFetchEE(bbox);
      }
    });

    return () => { sketch.destroy(); view.destroy(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
  if (mapRef.current) (mapRef.current as any).basemap = basemap;
}, [basemap]);

useEffect(() => {
  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Escape') {
      setSelectedImage(null);
    } else if (selectedImage && eeImages.length > 1) {
      const idx = eeImages.findIndex(i => i.year === selectedImage.year);
      if (e.key === 'ArrowLeft') {
        setSelectedImage(eeImages[(idx - 1 + eeImages.length) % eeImages.length]);
      } else if (e.key === 'ArrowRight') {
        setSelectedImage(eeImages[(idx + 1) % eeImages.length]);
      } else if (e.key === '+' || e.key === '=') {
        setLightboxZoom(z => Math.min(6, +(z + 0.3).toFixed(2)));
      } else if (e.key === '-') {
        setLightboxZoom(z => Math.max(1, +(z - 0.3).toFixed(2)));
      }
    }
  };
  window.addEventListener('keydown', handleKeyDown);
  return () => window.removeEventListener('keydown', handleKeyDown);
}, [selectedImage, eeImages]);
useEffect(() => {
  if (mapRef.current) (mapRef.current as any).basemap = basemap;
}, [basemap]);

  const handleExploreEnter = useCallback(() => {
    if (exploreTimerRef.current) {
      clearTimeout(exploreTimerRef.current);
      exploreTimerRef.current = null;
    }
    setExploreOpen(true);
  }, []);

  const handleExploreLeave = useCallback(() => {
    if (exploreTimerRef.current) clearTimeout(exploreTimerRef.current);
    exploreTimerRef.current = setTimeout(() => {
      setExploreOpen(false);
    }, 350);
  }, []);

  const flyToPlace = useCallback((place: DiscoverPlace) => {
    if (!viewRef.current) return;
    viewRef.current.goTo(
      { target: [place.lon, place.lat], zoom: place.zoom, tilt: 45 },
      { duration: 2200, easing: "in-out-cubic" }
    ).catch(() => {
      viewRef.current?.goTo({ target: { latitude: place.lat, longitude: place.lon }, zoom: place.zoom });
    });
    if (exploreTimerRef.current) {
      clearTimeout(exploreTimerRef.current);
      exploreTimerRef.current = null;
    }
    setExploreOpen(false);
  }, []);

  const handleEeEnter = useCallback(() => {
    if (eeTimerRef.current) {
      clearTimeout(eeTimerRef.current);
      eeTimerRef.current = null;
    }
    setEeOpen(true);
  }, []);

  const handleEeLeave = useCallback(() => {
    if (eePinned) return;
    if (eeTimerRef.current) clearTimeout(eeTimerRef.current);
    eeTimerRef.current = setTimeout(() => {
      setEeOpen(false);
    }, 350);
  }, [eePinned]);

  const doSearch = useCallback(async () => {
    const q = searchQuery.trim();
    if (!q || !viewRef.current) return;
    setSearchLoading(true);
    try {
      const url = `https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates?f=json&SingleLine=${encodeURIComponent(q)}&maxLocations=1`;
      const res = await fetch(url);
      const data = await res.json();
      const c = data.candidates?.[0];
      if (c?.location) {
        viewRef.current.goTo({ target: { latitude: c.location.y, longitude: c.location.x }, zoom: 10 });
        setShowSearch(false);
      }
    } catch (e) { console.error("Search failed:", e); }
    finally { setSearchLoading(false); }
  }, [searchQuery]);

  const toggleSelectMode = () => {
    if (!selectMode) {
      sketchRef.current?.create("rectangle");
      setSelectMode(true);
    } else {
      sketchRef.current?.cancel();
      setSelectMode(false);
    }
  };

  const toggle3DTiles = async () => {
    if (!viewRef.current) return;
    // If already active, remove layer
    if (tiles3DActive && tiles3DRef.current) {
      viewRef.current.map?.remove(tiles3DRef.current);
      tiles3DRef.current = null;
      setTiles3DActive(false);
      setTiles3DError(null);
      return;
    }
    // Use the current view extent as the bounding box (no explicit bbox query needed)
    const extent = (viewRef.current as any).extent;
    if (!extent) {
      setTiles3DError("Unable to determine view extent for 3‑D tiles.");
      return;
    }
    const baseUrl = "https://tiles.arcgis.com/tiles/8f4b3e6a4c5b4f9d/SceneServer/layers/0";
    // ArcGIS SceneLayer does not accept query parameters; request the layer directly
    const url = baseUrl;
    setTiles3DLoading(true);
    try {
      add3DLayer(viewRef, url);
      const added = viewRef.current.map?.layers.find((l: any) => l.type === "scene");
      tiles3DRef.current = added as any;
      setTiles3DActive(true);
    } catch (e: any) {
      console.error("Failed to add 3D Tiles layer:", e);
      setTiles3DError(e?.message ?? String(e));
    } finally {
      setTiles3DLoading(false);
    }
  };

  const doFetchEE = async (bbox: BBox) => {
    setEeLoading(true);
    setEeImages([]);
    setEeOpen(true);
    setEePinned(true);
    try {
      const res = await fetch("/api/earth-engine", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(bbox) });
      const data = await res.json();
      setEeImages((data.images ?? []).map((img: any) => ({ url: img.url, year: img.year })));
    } catch (e) { console.error("Earth Engine fetch failed:", e); }
    finally { setEeLoading(false); }
  };

  return (
    <div className="relative h-screen w-full overflow-hidden bg-black" style={{ fontFamily: "'Inter', sans-serif" }}>
      <div ref={mapDivRef} className="absolute inset-0" />

      {/* Bottom left: Earth Engine icon button (at exact intersection of Explore vertical line and Basemaps horizontal line) */}
      <div
        className="fixed bottom-6 left-6 z-40 group"
        onMouseEnter={handleEeEnter}
        onMouseLeave={handleEeLeave}
      >
        {/* Expanded Box / Dropdown (opens upward from the icon) */}
        {(eeOpen || eePinned) && (
          <div
            className="absolute bottom-full left-0 pb-2 w-72 pointer-events-auto"
            onMouseEnter={handleEeEnter}
            onMouseLeave={handleEeLeave}
          >
            <div className="rounded-2xl bg-neutral-900/95 border border-white/10 backdrop-blur-xl shadow-2xl overflow-hidden animate-in fade-in slide-in-from-bottom-2 duration-200">
              {/* Header */}
              <div className="flex items-center justify-between px-3.5 py-2.5 border-b border-white/10 bg-white/[0.03]">
                <div className="flex items-center gap-2">
                  <Satellite className="h-4 w-4 text-sky-400" />
                  <div>
                    <span className="text-xs font-semibold text-white">Earth Engine</span>
                    <span className="text-[10px] text-white/40 ml-1.5">Sentinel-2</span>
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setEePinned(false);
                    setEeOpen(false);
                  }}
                  className="h-6 w-6 rounded-md flex items-center justify-center text-white/50 hover:text-white hover:bg-white/10 transition-colors text-xs font-bold"
                  title="Minimize"
                >
                  —
                </button>
              </div>

              {/* Content: Empty state or 4 years images */}
              {!selectedBBox && eeImages.length === 0 ? (
                <div className="p-4 text-center flex flex-col items-center gap-2">
                  <div className="h-9 w-9 rounded-full bg-white/5 border border-white/10 flex items-center justify-center text-white/40">
                    <Satellite className="h-4 w-4" />
                  </div>
                  <p className="text-xs text-white/70 leading-relaxed px-2">
                    Please select an area on the globe to view past 4 years of imagery.
                  </p>
                  <button
                    onClick={() => {
                      toggleSelectMode();
                      setEeOpen(false);
                      setEePinned(false);
                    }}
                    className="mt-1 px-3 py-1.5 rounded-full bg-white/10 hover:bg-white/20 border border-white/15 text-white text-xs font-medium transition-colors flex items-center gap-1.5"
                  >
                    <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                      <rect x={3} y={3} width={7} height={7} /><rect x={14} y={3} width={7} height={7} />
                      <rect x={14} y={14} width={7} height={7} /><rect x={3} y={14} width={7} height={7} />
                    </svg>
                    Select Area
                  </button>
                </div>
              ) : (
                <div>
                  {selectedBBox && (
                    <div className="px-3 pt-2 text-[10px] font-mono text-white/40 leading-relaxed">
                      [{selectedBBox.south.toFixed(2)}°, {selectedBBox.west.toFixed(2)}°] → [{selectedBBox.north.toFixed(2)}°, {selectedBBox.east.toFixed(2)}°]
                    </div>
                  )}
                  <div className="grid grid-cols-2 gap-1.5 p-2.5">
                    {eeLoading ? (
                      Array.from({ length: 4 }).map((_, i) => (
                        <div key={i} className="aspect-square bg-white/5 rounded-xl border border-white/5 flex flex-col items-center justify-center gap-1">
                          <span className="h-4 w-4 rounded-full border border-t-white border-white/20 animate-spin" />
                          <span className="text-[10px] text-white/40 font-mono">Fetching…</span>
                        </div>
                      ))
                    ) : eeImages.length === 0 ? (
                      <div className="col-span-2 text-center text-white/30 text-xs py-4">No images returned</div>
                    ) : (
                      eeImages.slice(0, 4).map((img, i) => (
                        <div
                          key={i}
                          className="relative aspect-square rounded-xl overflow-hidden group/thumb cursor-pointer border border-white/10 hover:border-white/30 transition-all shadow-md hover:scale-[1.02]"
                          onClick={() => setSelectedImage(img)}
                        >
                          <img src={img.url} alt={String(img.year)} className="w-full h-full object-cover" />
                          <div className="absolute top-1 left-1 px-1.5 py-0.5 rounded bg-black/70 backdrop-blur-sm text-white/90 text-[10px] font-mono font-bold">
                            {img.year}
                          </div>
                          <div className="absolute inset-0 bg-black/40 opacity-0 group-hover/thumb:opacity-100 transition-opacity flex items-center justify-center text-[10px] text-white font-medium">
                            Click to expand
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Earth Engine Icon Button (Always visible at bottom-left intersection) */}
        <button
          id="earth-engine-icon-btn"
          onClick={() => {
            if (eeTimerRef.current) clearTimeout(eeTimerRef.current);
            setEePinned((p) => !p);
            setEeOpen((o) => !o);
          }}
          onMouseEnter={handleEeEnter}
          onMouseLeave={handleEeLeave}
          className={`h-10 w-10 rounded-full border backdrop-blur-md shadow-lg flex items-center justify-center transition-all duration-200 ${
            eeOpen || eePinned
              ? "bg-white/20 border-white/30 text-white shadow-sky-500/10 ring-2 ring-white/20"
              : "bg-black/70 border-white/10 text-white/80 hover:bg-white/15 hover:text-white hover:scale-105"
          }`}
          title="Earth Engine Satellite Imagery"
        >
          <Satellite className="h-4 w-4" />
        </button>
      </div>

      {/* Bottom right: Coordinates bar (matched to h-10 baseline) */}
      {isLoaded && (
        <div className="fixed bottom-6 right-6 h-10 flex items-center gap-3 bg-black/70 backdrop-blur-md border border-white/10 rounded-full px-4 text-xs text-white/80 font-mono select-none pointer-events-none z-20">
          <span><span className="text-white/40 mr-1">LAT</span>{coords.lat}°</span>
          <span className="text-white/20">|</span>
          <span><span className="text-white/40 mr-1">LNG</span>{coords.lon}°</span>
          <span className="text-white/20">|</span>
          <span><span className="text-white/40 mr-1">ALT</span>{coords.alt} km</span>
        </div>
      )}

      {/* Top left Explore/Discover button with dropdown (h-10 pill matching EE icon column) */}
      <div
        className="fixed top-7 left-6 z-50 group"
        onMouseEnter={handleExploreEnter}
        onMouseLeave={handleExploreLeave}
      >
        <button
          id="explore-discover-btn"
          onClick={() => {
            if (exploreTimerRef.current) clearTimeout(exploreTimerRef.current);
            setExploreOpen((v) => !v);
          }}
          className={`flex items-center h-10 rounded-full border backdrop-blur-md shadow-lg transition-all duration-300 ${
            exploreOpen
              ? "bg-white/20 border-white/30 text-white px-3.5 gap-2"
              : "bg-black/60 border-white/10 text-white/90 hover:bg-white/15 hover:text-white w-10 justify-center group-hover:w-auto group-hover:px-3.5 group-hover:gap-2"
          }`}
          title="Discover Earth"
        >
          <Compass className={`h-4 w-4 shrink-0 transition-transform duration-500 ${exploreOpen ? "rotate-90 text-sky-400" : "group-hover:rotate-45 text-white"}`} />
          <span
            className={`text-sm font-semibold transition-all duration-300 overflow-hidden whitespace-nowrap ${
              exploreOpen ? "max-w-28 opacity-100" : "max-w-0 opacity-0 group-hover:max-w-28 group-hover:opacity-100"
            }`}
          >
            Discover
          </span>
        </button>

        {/* Dropdown with 5 iconic places on Earth (seamless hover bridge via top-full pt-2) */}
        {exploreOpen && (
          <div
            className="absolute top-full left-0 pt-2 w-64 pointer-events-auto"
            onMouseEnter={handleExploreEnter}
            onMouseLeave={handleExploreLeave}
          >
            <div className="rounded-2xl bg-neutral-900/95 border border-white/10 backdrop-blur-xl shadow-2xl p-2 space-y-1 animate-in fade-in slide-in-from-top-2 duration-200">
              <div className="px-2.5 py-1 text-[10px] font-mono text-white/40 uppercase tracking-wider">
                Explore Earth
              </div>
              {DISCOVER_PLACES.map((place) => {
                const Icon = place.icon;
                return (
                  <button
                    key={place.label}
                    onClick={() => flyToPlace(place)}
                    className="flex items-center gap-3 w-full px-2.5 py-2 rounded-xl text-left hover:bg-white/10 transition-colors group/item"
                  >
                    <div className="h-8 w-8 rounded-lg bg-white/5 border border-white/10 flex items-center justify-center text-white/70 group-hover/item:text-white group-hover/item:bg-white/15 group-hover/item:border-white/20 transition-all shrink-0">
                      <Icon className="h-4 w-4" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-semibold text-white group-hover/item:text-sky-300 transition-colors truncate">
                        {place.label}
                      </div>
                      <div className="text-[10px] text-white/40 truncate">
                        {place.region}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Top right actions: Select Area + Search (aligned with Navbar) */}
      <div className="fixed top-7 right-6 flex items-center gap-2.5 z-50">
        <button
          id="select-area-btn"
          onClick={toggleSelectMode}
          className={`flex items-center gap-2 h-10 px-4 rounded-full text-sm font-semibold border backdrop-blur-md shadow-lg transition-all ${
            selectMode
              ? "bg-green-500/20 border-green-400 text-green-300 ring-2 ring-green-400/30"
              : "bg-black/60 border-white/10 text-white/90 hover:bg-white/15 hover:text-white"
          }`}
        >
          <svg className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <rect x={3} y={3} width={7} height={7} /><rect x={14} y={3} width={7} height={7} />
            <rect x={14} y={14} width={7} height={7} /><rect x={3} y={14} width={7} height={7} />
          </svg>
          {selectMode ? "Cancel" : "Select Area"}
        </button>

        <button
          id="search-btn"
          onClick={() => setShowSearch(v => !v)}
          className={`flex items-center gap-2 h-10 px-4 rounded-full text-sm font-semibold border backdrop-blur-md shadow-lg transition-all ${
            showSearch
              ? "bg-white/20 border-white/30 text-white"
              : "bg-black/60 border-white/10 text-white/90 hover:bg-white/15 hover:text-white"
          }`}
          title="Search location"
        >
          {searchLoading ? (
            <span className="inline-block h-4 w-4 rounded-full border border-t-white border-white/20 animate-spin" />
          ) : (
            <svg className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <circle cx={11} cy={11} r={8} />
              <path d="m21 21-4.35-4.35" />
            </svg>
          )}
          Search
        </button>
      </div>

      {/* Search dropdown popover */}
      {showSearch && (
        <div className="fixed top-[78px] right-6 w-80 pointer-events-auto z-50">
          <form
            onSubmit={(e) => { e.preventDefault(); doSearch(); }}
            className="flex items-center gap-2 rounded-2xl bg-neutral-900/95 border border-white/10 backdrop-blur-xl shadow-2xl px-3.5 py-2.5"
          >
            <svg className="h-4 w-4 text-white/40 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <circle cx={11} cy={11} r={8} />
              <path d="m21 21-4.35-4.35" />
            </svg>
            <input
              id="search-input"
              autoFocus
              type="text"
              placeholder="Search location…"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="flex-1 bg-transparent text-sm text-white placeholder-white/40 outline-none"
            />
            <button
              type="submit"
              disabled={searchLoading}
              className="rounded-full bg-white/20 hover:bg-white/30 px-3 py-1 text-xs font-semibold text-white transition-colors"
            >
              {searchLoading ? "..." : "Go"}
            </button>
            <button
              type="button"
              onClick={() => { setShowSearch(false); setSearchQuery(""); }}
              className="text-white/40 hover:text-white/70 text-xs px-1"
              title="Close search"
            >
              ✕
            </button>
          </form>
        </div>
      )}

      {/* Bottom center: Basemap Switcher Bar (matched to h-10 baseline) */}
      <div className="fixed bottom-6 left-1/2 -translate-x-1/2 h-10 flex items-center gap-1 bg-black/70 backdrop-blur-md border border-white/10 rounded-full px-1.5 z-20">
        {BASEMAPS.map((bm) => (
          <button
            key={bm.id}
            id={`basemap-${bm.id}`}
            onClick={() => setBasemap(bm.id)}
            className={`h-7 px-3.5 rounded-full text-xs font-medium transition-all ${
              basemap === bm.id
                ? "bg-white text-black font-semibold shadow-sm"
                : "text-white/70 hover:text-white hover:bg-white/10"
            }`}
          >
            {bm.label}
          </button>
        ))}
        {/* Separator */}
        <span className="w-px h-4 bg-white/10 mx-1" />
        {/* 3-D Tiles quick-add */}
        <button
          id="add-3d-tiles-btn"
          onClick={toggle3DTiles}
          className="h-7 px-3 rounded-full text-xs font-medium transition-all text-white/70 hover:text-white hover:bg-white/10 flex items-center gap-1.5"
          title="Toggle 3D Tiles layer"
        >
          <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
          </svg>
          3D Tiles
        </button>
      </div>

      <FloatingChatWindow title="GaiaLens Assistant" selectedBBox={selectedBBox}
        onSend={async (msg, history) => {
          try {
            const res = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: msg, history, bbox: selectedBBox }) });
            const data = await res.json();
            return data.reply ?? "Sorry, I could not process that.";
          } catch { return "Sorry, there was a connection error."; }
        }} />

      {selectedImage && (
        <div
          onClick={() => { setSelectedImage(null); setLightboxZoom(1); setLightboxPan({ x: 0, y: 0 }); }}
          className="fixed inset-0 bg-black/85 backdrop-blur-md flex items-center justify-center pointer-events-auto z-[250]"
        >
          <div
            onClick={e => e.stopPropagation()}
            className="relative flex flex-col w-[98vw] max-w-[1100px] max-h-[95vh] rounded-3xl overflow-hidden border border-white/[0.15] shadow-2xl bg-neutral-950"
          >
            {/* Modal Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-white/[0.08] bg-white/[0.02]">
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5">
                  {(eeImages.length > 0 ? [...eeImages].sort((a, b) => a.year - b.year) : [selectedImage]).map((img) => {
                    const isSelected = img.year === selectedImage.year;
                    return (
                      <button
                        key={img.year}
                        onClick={() => {
                          setSelectedImage(img);
                          setLightboxZoom(1);
                          setLightboxPan({ x: 0, y: 0 });
                        }}
                        className={`px-2.5 py-1 rounded-lg font-bold text-xs tracking-wide transition-all ${
                          isSelected
                            ? "bg-white/20 border border-white/40 text-white shadow-sm"
                            : "bg-white/5 border border-white/10 text-white/60 hover:bg-white/10 hover:text-white"
                        }`}
                        title={`View ${img.year}`}
                      >
                        {img.year}
                      </button>
                    );
                  })}
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-white">Sentinel-2 Multi-spectral Imagery</h3>
                  <p className="text-[11px] text-white/40">True Color (B4-Red, B3-Green, B2-Blue) · Harmonized L2A</p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                {/* Zoom Controls */}
                <div className="flex items-center gap-1 bg-black/60 border border-white/10 rounded-full px-2 py-1">
                  <button
                    onClick={() => setLightboxZoom(z => Math.max(1, +(z - 0.3).toFixed(2)))}
                    disabled={lightboxZoom <= 1}
                    className="h-7 w-7 rounded-full text-white/70 hover:text-white disabled:opacity-30 flex items-center justify-center text-sm font-bold transition-colors"
                    title="Zoom out (−)"
                  >−</button>
                  <span className="text-xs font-mono text-white/80 px-2 min-w-[50px] text-center">
                    {Math.round(lightboxZoom * 100)}%
                  </span>
                  <button
                    onClick={() => setLightboxZoom(z => Math.min(6, +(z + 0.3).toFixed(2)))}
                    disabled={lightboxZoom >= 6}
                    className="h-7 w-7 rounded-full text-white/70 hover:text-white disabled:opacity-30 flex items-center justify-center text-sm font-bold transition-colors"
                    title="Zoom in (+)"
                  >+</button>
                  {lightboxZoom > 1 && (
                    <button
                      onClick={() => { setLightboxZoom(1); setLightboxPan({ x: 0, y: 0 }); }}
                      className="text-[10px] text-white/60 hover:text-white px-2 font-medium"
                    >Reset</button>
                  )}
                </div>

                {/* Year navigation */}
                {eeImages.length > 1 && (
                  <div className="flex items-center gap-1 border-l border-white/10 pl-2">
                    <button
                      onClick={() => {
                        const idx = eeImages.findIndex(i => i.year === selectedImage.year);
                        const prev = eeImages[(idx - 1 + eeImages.length) % eeImages.length];
                        setSelectedImage(prev); setLightboxZoom(1); setLightboxPan({ x: 0, y: 0 });
                      }}
                      title="Previous year (←)"
                      className="h-8 w-8 rounded-full bg-white/5 hover:bg-white/15 border border-white/10 text-white flex items-center justify-center text-sm transition-colors"
                    >‹</button>
                    <button
                      onClick={() => {
                        const idx = eeImages.findIndex(i => i.year === selectedImage.year);
                        const next = eeImages[(idx + 1) % eeImages.length];
                        setSelectedImage(next); setLightboxZoom(1); setLightboxPan({ x: 0, y: 0 });
                      }}
                      title="Next year (→)"
                      className="h-8 w-8 rounded-full bg-white/5 hover:bg-white/15 border border-white/10 text-white flex items-center justify-center text-sm transition-colors"
                    >›</button>
                  </div>
                )}

                {/* Close */}
                <button
                  onClick={() => { setSelectedImage(null); setLightboxZoom(1); setLightboxPan({ x: 0, y: 0 }); }}
                  className="h-8 w-8 rounded-full bg-white/10 hover:bg-white/20 text-white flex items-center justify-center text-sm transition-colors ml-2"
                  title="Close (Esc)"
                >✕</button>
              </div>
            </div>

            {/* Image Viewport */}
            <div
              onWheel={handleLightboxWheel}
              onMouseDown={handleLightboxMouseDown}
              className={`relative flex-1 overflow-hidden flex items-center justify-center bg-black min-h-[500px] max-h-[80vh] select-none ${
                lightboxZoom > 1 ? "cursor-grab active:cursor-grabbing" : "cursor-zoom-in"
              }`}
            >
              <div
                style={{
                  transform: `scale(${lightboxZoom}) translate(${lightboxPan.x / lightboxZoom}px, ${lightboxPan.y / lightboxZoom}px)`,
                  transformOrigin: "center center",
                  transition: lightboxDragRef.current.active ? "none" : "transform 0.15s ease-out",
                }}
                className="w-full h-full"
              >
                <img
                  src={selectedImage.url}
                  alt={String(selectedImage.year)}
                  draggable={false}
                  className="w-full h-full object-cover pointer-events-none"
                  onError={e => {
                    (e.target as HTMLImageElement).src = `https://placehold.co/600x400/0a0a0a/4a4a4a?text=${selectedImage?.year}`;
                  }}
                />
              </div>

              {/* Scroll zoom hint */}
              <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2 bg-black/70 border border-white/10 rounded-full px-3 py-1 backdrop-blur-md text-[10px] text-white/50 pointer-events-none">
                <svg viewBox="0 0 20 20" fill="none" className="h-3.5 w-3.5 opacity-70">
                  <circle cx="10" cy="10" r="7" stroke="currentColor" strokeWidth="1.5" />
                  <path d="M10 6v3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
                <span>Scroll to zoom{lightboxZoom > 1 ? " · Drag to pan" : ""} · ← → to navigate</span>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="px-6 py-3 border-t border-white/[0.08] bg-white/[0.02] flex items-center justify-between text-xs text-white/50">
              <div className="flex items-center gap-4 font-mono text-[11px]">
                <span>Year: <b className="text-white">{selectedImage.year}</b></span>
                <span>Platform: <b className="text-white">Sentinel-2 MSI</b></span>
                {selectedBBox && (
                  <span>BBox: <b className="text-white">[{selectedBBox.south.toFixed(3)}, {selectedBBox.west.toFixed(3)}] → [{selectedBBox.north.toFixed(3)}, {selectedBBox.east.toFixed(3)}]</b></span>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {!isLoaded && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black z-50">
          <div className="relative h-16 w-16 mb-4">
            <div className="absolute inset-0 rounded-full border-2 border-white/10" />
            <div className="absolute inset-0 rounded-full border-t-2 border-sky-400 animate-spin" />
          </div>
          <p className="text-white/60 text-sm tracking-widest uppercase">Loading Globe…</p>
        </div>
      )}

      {/* Floating AI Assistant Chat Window */}
      <FloatingChatWindow
        selectedBBox={selectedBBox}
        onSend={handleChatSend}
      />
    </div>
  );
}
