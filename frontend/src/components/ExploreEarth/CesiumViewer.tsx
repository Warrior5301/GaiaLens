"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { motion, AnimatePresence } from "motion/react";
import FloatingChatWindow from "./FloatingChatWindow";

// Altitude (metres) below which the "Select Area" tool becomes available
const SELECT_AREA_ALTITUDE_THRESHOLD = 200_000;

// ─── Types ────────────────────────────────────────────────────────────────────
type ImageryMode = "satellite" | "hybrid" | "osm" | "terrain";

type CameraPreset = {
    label: string;
    icon: string;
    lat: number;
    lng: number;
    height: number;
    pitch: number;
    heading: number;
};

// ─── Presets ──────────────────────────────────────────────────────────────────
const PRESETS: CameraPreset[] = [
    { label: "Earth", icon: "🌍", lat: 20, lng: 78, height: 15_000_000, pitch: -90, heading: 0 },
    { label: "Himalayas", icon: "🏔️", lat: 27.99, lng: 86.93, height: 40_000, pitch: -30, heading: 20 },
    { label: "New Delhi", icon: "🏛️", lat: 28.61, lng: 77.21, height: 15_000, pitch: -45, heading: 0 },
    { label: "Mumbai", icon: "🌊", lat: 19.08, lng: 72.88, height: 12_000, pitch: -40, heading: 340 },
    { label: "Amazon", icon: "🌿", lat: -3.47, lng: -62.22, height: 200_000, pitch: -60, heading: 0 },
    { label: "Sahara", icon: "🏜️", lat: 23.42, lng: 25.66, height: 400_000, pitch: -70, heading: 0 },
];

// ─── Imagery modes ────────────────────────────────────────────────────────────
const MODES: { id: ImageryMode; label: string; icon: string }[] = [
    { id: "satellite", label: "Satellite", icon: "🛰️" },
    { id: "hybrid", label: "Hybrid", icon: "🗺️" },
    { id: "osm", label: "Streets", icon: "🛣️" },
    { id: "terrain", label: "Terrain", icon: "⛰️" },
];

// ─── Main Component ───────────────────────────────────────────────────────────
export default function CesiumViewer() {
    // The Cesium canvas container — NEVER re-mounted after init
    const containerRef = useRef<HTMLDivElement>(null);
    const viewerRef = useRef<any>(null);
    const rafRef = useRef<number | null>(null);
    const initializedRef = useRef(false); // guards against Strict Mode double-fire

    // DOM refs for coordinate HUD — updated directly, never via setState
    const latRef = useRef<HTMLSpanElement>(null);
    const lngRef = useRef<HTMLSpanElement>(null);
    const altRef = useRef<HTMLSpanElement>(null);

    // React state only for structural UI changes (not per-frame updates)
    const [isLoaded, setIsLoaded] = useState(false);
    const [loadError, setLoadError] = useState<string | null>(null);
    const [activeMode, setActiveMode] = useState<ImageryMode>("satellite");
    const [currentPreset, setCurrentPreset] = useState("Earth");
    const [searchQuery, setSearchQuery] = useState("");
    const [showSearch, setShowSearch] = useState(false);
    const [tiles3DActive, setTiles3DActive] = useState(false);
    const [tiles3DLoading, setTiles3DLoading] = useState(false);
    const [tiles3DError, setTiles3DError] = useState<string | null>(null);
    const tileset3DRef = useRef<any>(null);

    // ── Rectangle selector state ───────────────────────────────────────────────
    const [cameraAlt,      setCameraAlt]      = useState(15_000_000); // metres
    const [selectModeOn,   setSelectModeOn]   = useState(false);
    const [isDrawing,      setIsDrawing]      = useState(false);
    const [rectScreen,     setRectScreen]     = useState<{x:number;y:number;w:number;h:number} | null>(null);
    const [selectedBBox,   setSelectedBBox]   = useState<{north:number;south:number;east:number;west:number} | null>(null);
    const drawStartRef  = useRef<{x:number;y:number} | null>(null);
    const rectOverlayRef = useRef<HTMLDivElement>(null);

    // ── Earth Engine imagery state ─────────────────────────────────────────────
    type GeeImage = { year: number; date: string; url: string; mock?: boolean };
    const [geeImages,            setGeeImages]            = useState<GeeImage[]>([]);
    const [geeLoading,           setGeeLoading]           = useState(false);
    const [expandedImage,        setExpandedImage]        = useState<GeeImage | null>(null);
    const [showImagePanel,       setShowImagePanel]       = useState(false);
    const [imagePanelMinimized,  setImagePanelMinimized]  = useState(false);

    // Lightbox zoom & pan state
    const [lightboxZoom, setLightboxZoom] = useState(1);
    const [lightboxPan,  setLightboxPan]  = useState({ x: 0, y: 0 });
    const lightboxDragRef = useRef({ active: false, startX: 0, startY: 0, ox: 0, oy: 0 });

    const openLightbox = useCallback((img: GeeImage) => {
        setExpandedImage(img);
        setLightboxZoom(1);
        setLightboxPan({ x: 0, y: 0 });
    }, []);

    // Lightbox mouse wheel zoom handler
    const handleLightboxWheel = useCallback((e: React.WheelEvent) => {
        e.preventDefault();
        e.stopPropagation();
        const factor = e.deltaY < 0 ? 1.2 : 0.83;
        setLightboxZoom(prev => {
            const next = Math.min(6, Math.max(1, +(prev * factor).toFixed(2)));
            if (next === 1) setLightboxPan({ x: 0, y: 0 });
            return next;
        });
    }, []);

    // Lightbox pan drag handlers
    const handleLightboxPanStart = useCallback((e: React.MouseEvent) => {
        if (lightboxZoom <= 1) return;
        e.preventDefault();
        lightboxDragRef.current = {
            active: true,
            startX: e.clientX,
            startY: e.clientY,
            ox: lightboxPan.x,
            oy: lightboxPan.y,
        };
    }, [lightboxZoom, lightboxPan]);

    // Global mouse listeners for lightbox pan
    useEffect(() => {
        const onMouseMove = (e: MouseEvent) => {
            if (!lightboxDragRef.current.active) return;
            const dx = e.clientX - lightboxDragRef.current.startX;
            const dy = e.clientY - lightboxDragRef.current.startY;
            setLightboxPan({
                x: lightboxDragRef.current.ox + dx,
                y: lightboxDragRef.current.oy + dy,
            });
        };
        const onMouseUp = () => {
            lightboxDragRef.current.active = false;
        };
        window.addEventListener("mousemove", onMouseMove);
        window.addEventListener("mouseup", onMouseUp);
        return () => {
            window.removeEventListener("mousemove", onMouseMove);
            window.removeEventListener("mouseup", onMouseUp);
        };
    }, []);

    // Keyboard navigation (Escape to close, Left/Right arrow to cycle images, +/- to zoom)
    useEffect(() => {
        const handleKeyDown = (e: globalThis.KeyboardEvent) => {
            if (e.key === "Escape") {
                if (expandedImage) {
                    setExpandedImage(null);
                } else if (selectModeOn) {
                    setSelectModeOn(false);
                    setRectScreen(null);
                } else if (showSearch) {
                    setShowSearch(false);
                }
            } else if (expandedImage && geeImages.length > 1) {
                if (e.key === "ArrowLeft") {
                    const idx = geeImages.findIndex(i => i.year === expandedImage.year);
                    const prev = geeImages[(idx - 1 + geeImages.length) % geeImages.length];
                    openLightbox(prev);
                } else if (e.key === "ArrowRight") {
                    const idx = geeImages.findIndex(i => i.year === expandedImage.year);
                    const next = geeImages[(idx + 1) % geeImages.length];
                    openLightbox(next);
                } else if (e.key === "+" || e.key === "=") {
                    setLightboxZoom(z => Math.min(6, +(z + 0.3).toFixed(2)));
                } else if (e.key === "-") {
                    setLightboxZoom(z => Math.max(1, +(z - 0.3).toFixed(2)));
                }
            }
        };
        window.addEventListener("keydown", handleKeyDown);
        return () => window.removeEventListener("keydown", handleKeyDown);
    }, [expandedImage, selectModeOn, showSearch, geeImages, openLightbox]);

    // Image panel drag (DOM-direct, no React state for perf)
    const imagePanelRef     = useRef<HTMLDivElement>(null);
    const imagePanelPosRef  = useRef({ x: 20, y: 130 });
    const imagePanelDragRef = useRef({ active: false, startX: 0, startY: 0, ox: 0, oy: 0 });

    // Image panel drag listeners with boundary clamping
    useEffect(() => {
        const onMove = (e: MouseEvent) => {
            if (!imagePanelDragRef.current.active) return;
            const dx = e.clientX - imagePanelDragRef.current.startX;
            const dy = e.clientY - imagePanelDragRef.current.startY;
            imagePanelPosRef.current.x = Math.max(10, Math.min(window.innerWidth - 270, imagePanelDragRef.current.ox + dx));
            imagePanelPosRef.current.y = Math.max(76, Math.min(window.innerHeight - 100, imagePanelDragRef.current.oy + dy));
            const el = imagePanelRef.current;
            if (el) { el.style.left = `${imagePanelPosRef.current.x}px`; el.style.top = `${imagePanelPosRef.current.y}px`; }
        };
        const onUp = () => { imagePanelDragRef.current.active = false; document.body.style.userSelect = ''; };
        window.addEventListener('mousemove', onMove);
        window.addEventListener('mouseup', onUp);
        return () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); };
    }, []);

    // ── Bootstrap Cesium from CDN ─────────────────────────────────────────────
    useEffect(() => {
        // Guard: only ever initialise once (Strict Mode fires effects twice)
        if (initializedRef.current) return;
        initializedRef.current = true;

        const ionToken = process.env.NEXT_PUBLIC_CESIUM_ION_TOKEN;
        if (!ionToken || ionToken === "YOUR_CESIUM_ION_TOKEN_HERE") {
            setLoadError("Cesium Ion token not set — add NEXT_PUBLIC_CESIUM_ION_TOKEN to .env.local");
            return;
        }

        // Inject widget CSS once
        if (!document.getElementById("cesium-css")) {
            const link = document.createElement("link");
            link.id = "cesium-css";
            link.rel = "stylesheet";
            link.href = "https://cesium.com/downloads/cesiumjs/releases/1.122/Build/Cesium/Widgets/widgets.css";
            document.head.appendChild(link);
        }

        // Load CesiumJS script once
        const loadScript = (): Promise<void> => {
            if ((window as any).Cesium) return Promise.resolve();
            return new Promise((resolve, reject) => {
                const s = document.createElement("script");
                s.src = "https://cesium.com/downloads/cesiumjs/releases/1.122/Build/Cesium/Cesium.js";
                s.onload = () => resolve();
                s.onerror = () => reject(new Error("CesiumJS script failed to load"));
                document.head.appendChild(s);
            });
        };

        loadScript()
            .then(async () => {
                if (!containerRef.current) return;

                const Cesium = (window as any).Cesium;
                Cesium.Ion.defaultAccessToken = ionToken;

                // ── Create Viewer ──────────────────────────────────────────
                const viewer = new Cesium.Viewer(containerRef.current, {
                    terrain: Cesium.Terrain.fromWorldTerrain({
                        requestWaterMask: true,
                        requestVertexNormals: true,
                    }),
                    imageryProvider: false,
                    baseLayerPicker: false,
                    navigationHelpButton: false,
                    homeButton: false,
                    geocoder: false,
                    sceneModePicker: false,
                    animation: false,
                    timeline: false,
                    fullscreenButton: false,
                    infoBox: false,
                    selectionIndicator: false,
                    // Suppress the Cesium credit banner (it gets its own div)
                    creditContainer: Object.assign(document.createElement("div"), {
                        style: "display:none",
                    }),
                    shadows: false,
                    msaaSamples: 4,
                });

                // ── Add satellite imagery ──────────────────────────────────
                const aerial = await Cesium.IonImageryProvider.fromAssetId(2);
                viewer.imageryLayers.addImageryProvider(aerial);

                // ── Globe settings ─────────────────────────────────────────
                viewer.scene.globe.depthTestAgainstTerrain = true;
                viewer.scene.globe.enableLighting = true;
                viewer.scene.screenSpaceCameraController.minimumZoomDistance = 200;

                // ── Camera position HUD — RAF-throttled, DOM-direct ────────
                // Using requestAnimationFrame instead of setState prevents
                // React re-renders that would trigger Cesium canvas resizes.
                const updateHUD = () => {
                    const carto = viewer.camera.positionCartographic;
                    if (carto && latRef.current && lngRef.current && altRef.current) {
                        latRef.current.textContent = `${Cesium.Math.toDegrees(carto.latitude).toFixed(4)}°`;
                        lngRef.current.textContent = `${Cesium.Math.toDegrees(carto.longitude).toFixed(4)}°`;
                        const km = carto.height / 1000;
                        altRef.current.textContent = km > 1
                            ? `${km.toFixed(1)} km`
                            : `${Math.round(carto.height)} m`;
                    }
                    rafRef.current = requestAnimationFrame(updateHUD);
                };
                rafRef.current = requestAnimationFrame(updateHUD);

                viewerRef.current = viewer;
                setIsLoaded(true);
            })
            .catch((err) => {
                console.error("Cesium init error:", err);
                setLoadError("Failed to initialise CesiumJS. Check console.");
            });

        return () => {
            if (rafRef.current) cancelAnimationFrame(rafRef.current);
            if (viewerRef.current && !viewerRef.current.isDestroyed()) {
                viewerRef.current.destroy();
                viewerRef.current = null;
            }
        };
    }, []);

    // ── Altitude monitor — updates setCameraAlt every 500 ms ──────────────────
    useEffect(() => {
        const id = setInterval(() => {
            const viewer = viewerRef.current;
            if (!viewer || viewer.isDestroyed()) return;
            const h = viewer.camera.positionCartographic?.height ?? 15_000_000;
            setCameraAlt(h);
        }, 500);
        return () => clearInterval(id);
    }, []);

    // Deactivate select mode if user zooms out too far
    useEffect(() => {
        if (cameraAlt > SELECT_AREA_ALTITUDE_THRESHOLD && selectModeOn) {
            setSelectModeOn(false);
            setIsDrawing(false);
            setRectScreen(null);
            drawStartRef.current = null;
        }
    }, [cameraAlt, selectModeOn]);

    // ── Fly-to preset ─────────────────────────────────────────────────────────
    const flyTo = useCallback((preset: CameraPreset) => {
        const Cesium = (window as any).Cesium;
        const viewer = viewerRef.current;
        if (!Cesium || !viewer) return;

        viewer.camera.flyTo({
            destination: Cesium.Cartesian3.fromDegrees(preset.lng, preset.lat, preset.height),
            orientation: {
                heading: Cesium.Math.toRadians(preset.heading),
                pitch: Cesium.Math.toRadians(preset.pitch),
                roll: 0,
            },
            duration: 2.5,
        });
        setCurrentPreset(preset.label);
    }, []);

    // ── Imagery mode switch (async providers) ─────────────────────────────────
    const switchMode = useCallback(async (mode: ImageryMode) => {
        const Cesium = (window as any).Cesium;
        const viewer = viewerRef.current;
        if (!Cesium || !viewer) return;

        viewer.imageryLayers.removeAll();
        try {
            let provider;
            switch (mode) {
                case "satellite": provider = await Cesium.IonImageryProvider.fromAssetId(2); break;
                case "hybrid": provider = await Cesium.IonImageryProvider.fromAssetId(3); break;
                case "terrain": provider = await Cesium.IonImageryProvider.fromAssetId(3845); break;
                case "osm":
                    provider = await Cesium.OpenStreetMapImageryProvider.fromUrl(
                        "https://tile.openstreetmap.org/"
                    );
                    break;
            }
            if (provider) viewer.imageryLayers.addImageryProvider(provider);
        } catch (e) {
            console.error("Imagery switch error:", e);
        }
        setActiveMode(mode);
    }, []);

    // ── Geocoder search via Nominatim ─────────────────────────────────────────
    const handleSearch = useCallback(async (e: React.FormEvent) => {
        e.preventDefault();
        if (!searchQuery.trim()) return;
        try {
            const res = await fetch(
                `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(searchQuery)}&limit=1`,
                { headers: { "Accept-Language": "en" } }
            );
            const data = await res.json();
            if (data?.[0]) {
                flyTo({
                    label: data[0].display_name.split(",")[0],
                    icon: "📍",
                    lat: parseFloat(data[0].lat),
                    lng: parseFloat(data[0].lon),
                    height: 10_000,
                    pitch: -45,
                    heading: 0,
                });
                setShowSearch(false);
                setSearchQuery("");
            }
        } catch { /* silent */ }
    }, [searchQuery, flyTo]);

    // ── Earth Engine: fetch 4 years of imagery for a bbox ─────────────────────
    const fetchGeeImages = useCallback(async (bbox: {north:number;south:number;east:number;west:number}) => {
        setGeeLoading(true);
        setShowImagePanel(true);
        setGeeImages([]);
        try {
            const res = await fetch("/api/earth-engine", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(bbox),
            });
            const data = await res.json();
            if (data.images && Array.isArray(data.images)) {
                setGeeImages(data.images);
            } else {
                console.warn("GEE response:", data);
            }
        } catch (err) {
            console.error("GEE fetch error:", err);
        } finally {
            setGeeLoading(false);
        }
    }, []);

    // ── Rectangle drawing on canvas ────────────────────────────────────────────
    const pickLatLng = useCallback((screenX: number, screenY: number) => {
        const Cesium = (window as any).Cesium;
        const viewer = viewerRef.current;
        if (!Cesium || !viewer) return null;
        // Get the canvas bounding rect to offset screen coords
        const rect = viewer.canvas.getBoundingClientRect();
        const pos = viewer.scene.camera.pickEllipsoid(
            new Cesium.Cartesian2(screenX - rect.left, screenY - rect.top),
            viewer.scene.globe.ellipsoid
        );
        if (!pos) return null;
        const carto = Cesium.Cartographic.fromCartesian(pos);
        return {
            lat: Cesium.Math.toDegrees(carto.latitude),
            lng: Cesium.Math.toDegrees(carto.longitude),
        };
    }, []);

    const handleCanvasMouseDown = useCallback((e: React.MouseEvent) => {
        if (!selectModeOn) return;
        e.preventDefault();
        drawStartRef.current = { x: e.clientX, y: e.clientY };
        setIsDrawing(true);
        setRectScreen({ x: e.clientX, y: e.clientY, w: 0, h: 0 });
    }, [selectModeOn]);

    const handleCanvasMouseMove = useCallback((e: React.MouseEvent) => {
        if (!isDrawing || !drawStartRef.current) return;
        const sx = drawStartRef.current.x;
        const sy = drawStartRef.current.y;
        setRectScreen({
            x: Math.min(sx, e.clientX),
            y: Math.min(sy, e.clientY),
            w: Math.abs(e.clientX - sx),
            h: Math.abs(e.clientY - sy),
        });
    }, [isDrawing]);

    const handleCanvasMouseUp = useCallback((e: React.MouseEvent) => {
        if (!isDrawing || !drawStartRef.current) return;
        setIsDrawing(false);
        const start = pickLatLng(drawStartRef.current.x, drawStartRef.current.y);
        const end   = pickLatLng(e.clientX, e.clientY);
        drawStartRef.current = null;
        if (!start || !end) return;
        const bbox = {
            north: Math.max(start.lat, end.lat),
            south: Math.min(start.lat, end.lat),
            east:  Math.max(start.lng, end.lng),
            west:  Math.min(start.lng, end.lng),
        };
        setSelectedBBox(bbox);
        setSelectModeOn(false);
        fetchGeeImages(bbox);
    }, [isDrawing, pickLatLng, fetchGeeImages]);

    // ── 3D Buildings toggle (Cesium Ion OSM Buildings — free, no Google needed) ─
    const toggle3DTiles = useCallback(async () => {
        const Cesium = (window as any).Cesium;
        const viewer = viewerRef.current;
        if (!Cesium || !viewer) return;

        if (tiles3DActive) {
            // ── Disable: remove buildings tileset ─────────────────────────
            if (tileset3DRef.current) {
                viewer.scene.primitives.remove(tileset3DRef.current);
                tileset3DRef.current = null;
            }
            setTiles3DActive(false);
            setTiles3DError(null);
            return;
        }

        // ── Enable: load Cesium Ion OSM Buildings (Asset 2275207) ─────────
        // This is 100% free with any Cesium Ion account — no Google Cloud needed.
        setTiles3DLoading(true);
        try {
            const tileset = await Cesium.Cesium3DTileset.fromIonAssetId(2275207, {
                showCreditsOnScreen: true,
            });

            // Style buildings with a clean white/glass look
            tileset.style = new Cesium.Cesium3DTileStyle({
                color: "color('white', 0.85)",
            });

            viewer.scene.primitives.add(tileset);
            tileset3DRef.current = tileset;

            // Keep the globe + satellite imagery visible underneath
            // (buildings sit on top of the terrain naturally)

            // Tilt camera to show buildings from an angle
            const carto = viewer.camera.positionCartographic;
            const currentHeight = carto?.height ?? 15_000_000;

            // Only fly down if we're too high to see buildings
            if (currentHeight > 3000) {
                const lng = Cesium.Math.toDegrees(carto?.longitude ?? 77.21);
                const lat = Cesium.Math.toDegrees(carto?.latitude  ?? 28.61);
                viewer.camera.flyTo({
                    destination: Cesium.Cartesian3.fromDegrees(lng, lat, 1200),
                    orientation: {
                        heading: Cesium.Math.toRadians(0),
                        pitch:   Cesium.Math.toRadians(-35),
                        roll:    0,
                    },
                    duration: 2.5,
                });
            }

            setTiles3DActive(true);
            setTiles3DError(null);
        } catch (err: any) {
            const detail =
                err?.message ??
                (typeof err === "string" ? err : null) ??
                JSON.stringify(err);
            console.error("OSM Buildings load failed:", { detail, raw: err });
            setTiles3DError(`3D Buildings error: ${String(detail).slice(0, 150)}`);
            setTimeout(() => setTiles3DError(null), 8000);
        } finally {
            setTiles3DLoading(false);
        }
    }, [tiles3DActive]);



    // ── Render ────────────────────────────────────────────────────────────────
    return (
        <>
            {/*
             * The Cesium canvas sits in a FIXED div so it is completely outside
             * React's layout flow. React state changes never reflow this div,
             * so Cesium never sees a "resize" and the flickering is eliminated.
             */}
            <div
                ref={containerRef}
                style={{ position: "fixed", inset: 0, zIndex: 10 }}
            />

            {/* ── Mouse-event overlay for rectangle drawing ─────────────────── */}
            {selectModeOn && (
                <div
                    className="fixed inset-0 z-[55]"
                    style={{ cursor: isDrawing ? "crosshair" : "crosshair" }}
                    onMouseDown={handleCanvasMouseDown}
                    onMouseMove={handleCanvasMouseMove}
                    onMouseUp={handleCanvasMouseUp}
                >
                    {/* Live rectangle outline */}
                    {rectScreen && isDrawing && (
                        <div
                            className="absolute border-2 border-sky-400/80 bg-sky-400/10 pointer-events-none"
                            style={{ left: rectScreen.x, top: rectScreen.y, width: rectScreen.w, height: rectScreen.h }}
                        />
                    )}
                </div>
            )}

            {/* Wrapper for all overlay UI — sits above Cesium */}
            <div className="fixed inset-0 z-[60] pointer-events-none">

                {/* Loading / Error */}
                <AnimatePresence>
                    {(!isLoaded || loadError) && (
                        <motion.div
                            initial={{ opacity: 1 }}
                            exit={{ opacity: 0, transition: { duration: 0.8 } }}
                            className="absolute inset-0 z-50 flex flex-col items-center justify-center gap-5 bg-neutral-950 pointer-events-auto"
                        >
                            {loadError ? (
                                <>
                                    <span className="text-4xl">🌐</span>
                                    <h2 className="text-lg font-semibold text-white">Cesium Error</h2>
                                    <p className="max-w-sm text-center text-sm text-neutral-400">{loadError}</p>
                                    <code className="rounded-lg bg-neutral-900 border border-white/10 px-4 py-2 text-xs text-emerald-400">
                                        NEXT_PUBLIC_CESIUM_ION_TOKEN=your_token
                                    </code>
                                    <a
                                        href="https://ion.cesium.com/tokens"
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="text-xs text-sky-400 underline"
                                    >
                                        Get a free token at ion.cesium.com →
                                    </a>
                                </>
                            ) : (
                                <>
                                    <div className="relative h-16 w-16">
                                        <div className="absolute inset-0 rounded-full border-2 border-white/10" />
                                        <div className="absolute inset-0 rounded-full border-t-2 border-sky-400 animate-spin" />
                                        <div className="absolute inset-2 rounded-full border-t-2 border-blue-500 animate-spin [animation-duration:1.5s] [animation-direction:reverse]" />
                                    </div>
                                    <p className="text-sm text-neutral-400 animate-pulse">Loading Cesium Globe…</p>
                                </>
                            )}
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* ── Secondary bar — sits below the global Navbar (~76px) ── */}
                <div className="absolute inset-x-0 top-[76px] flex items-center justify-between px-6 py-2 pointer-events-auto">

                    {/* Cesium Ion status badge */}
                    {isLoaded && (
                        <span className="flex items-center gap-1.5 text-xs text-white/50 bg-black/40 backdrop-blur-md rounded-full px-3 py-1.5 border border-white/[0.06]">
                            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                            Cesium Ion
                        </span>
                    )}

                    {/* Coordinate HUD — updated via DOM ref, zero React re-renders */}
                    <div className="hidden md:flex items-center gap-4 bg-black/50 backdrop-blur-md rounded-full px-4 py-1.5 border border-white/[0.08]">
                        <span className="text-[10px] font-mono text-white/40">LAT</span>
                        <span ref={latRef} className="text-[11px] font-mono text-white">—</span>
                        <span className="text-[10px] font-mono text-white/40">LNG</span>
                        <span ref={lngRef} className="text-[11px] font-mono text-white">—</span>
                        <span className="text-[10px] font-mono text-white/40">ALT</span>
                        <span ref={altRef} className="text-[11px] font-mono text-white">—</span>
                    </div>

                    {/* Top right actions: Select Area + Search */}
                    <div className="flex items-center gap-2">
                        {isLoaded && cameraAlt < SELECT_AREA_ALTITUDE_THRESHOLD && (
                            <motion.button
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                whileHover={{ scale: 1.04 }}
                                whileTap={{ scale: 0.96 }}
                                onClick={() => { setSelectModeOn(v => !v); setRectScreen(null); }}
                                className={`flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold border backdrop-blur-md shadow-lg transition-all ${
                                    selectModeOn
                                        ? "bg-sky-500 border-sky-400/50 text-white shadow-sky-500/30 ring-2 ring-sky-400/40"
                                        : "bg-white/10 border-white/10 text-white/90 hover:bg-white/15 hover:text-white"
                                }`}
                            >
                                <svg viewBox="0 0 20 20" fill="none" className="h-4 w-4">
                                    <rect x="3" y="3" width="14" height="14" rx="1" stroke="currentColor" strokeWidth="1.5" strokeDasharray="3 2" />
                                    <path d="M10 7v6M7 10h6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                                </svg>
                                <span>{selectModeOn ? "Cancel Selection" : "Select Area"}</span>
                            </motion.button>
                        )}

                        {/* Search toggle */}
                        <button
                            onClick={() => setShowSearch(v => !v)}
                            className={`flex items-center gap-2 rounded-full border px-4 py-2 text-sm transition-colors backdrop-blur-md ${
                                showSearch
                                    ? "bg-white/20 border-white/30 text-white"
                                    : "bg-white/10 border-white/10 text-white/80 hover:bg-white/15 hover:text-white"
                            }`}
                        >
                            <svg viewBox="0 0 20 20" fill="none" className="h-4 w-4">
                                <circle cx="9" cy="9" r="6" stroke="currentColor" strokeWidth="1.5" />
                                <path d="M13.5 13.5L17 17" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                            </svg>
                            <span>Search</span>
                        </button>
                    </div>
                </div>

                {/* ── Selection active floating guide banner ─────────────── */}
                <AnimatePresence>
                    {selectModeOn && (
                        <motion.div
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="absolute top-[125px] left-1/2 -translate-x-1/2 flex items-center gap-2.5 bg-sky-950/85 border border-sky-400/40 text-sky-200 text-xs px-4 py-2 rounded-full backdrop-blur-xl shadow-2xl pointer-events-auto z-50"
                        >
                            <span className="h-2 w-2 rounded-full bg-sky-400 animate-ping" />
                            <span>Click and drag on the globe to draw a rectangle area</span>
                            <button
                                onClick={() => { setSelectModeOn(false); setRectScreen(null); }}
                                className="ml-2 text-sky-300 hover:text-white underline font-semibold text-[11px]"
                            >
                                Cancel (Esc)
                            </button>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* ── Search overlay ─────────────────────────────────────── */}
                <AnimatePresence>
                    {showSearch && (
                        <motion.div
                            key="search"
                            initial={{ opacity: 0, y: -8 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -8 }}
                            transition={{ duration: 0.18 }}
                            className="absolute top-[136px] left-1/2 -translate-x-1/2 w-full max-w-md px-4 pointer-events-auto z-50"
                        >
                            <form
                                onSubmit={handleSearch}
                                className="flex items-center gap-3 rounded-2xl bg-neutral-900/95 border border-white/10 backdrop-blur-xl shadow-2xl px-4 py-3"
                            >
                                <svg viewBox="0 0 20 20" fill="none" className="h-4 w-4 text-white/40 shrink-0">
                                    <circle cx="9" cy="9" r="6" stroke="currentColor" strokeWidth="1.5" />
                                    <path d="M13.5 13.5L17 17" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                                </svg>
                                <input
                                    autoFocus
                                    value={searchQuery}
                                    onChange={e => setSearchQuery(e.target.value)}
                                    placeholder="Fly to any place on Earth…"
                                    className="flex-1 bg-transparent text-sm text-white placeholder:text-white/30 outline-none"
                                />
                                <button
                                    type="submit"
                                    className="rounded-full bg-sky-500 hover:bg-sky-400 px-3 py-1 text-xs font-semibold text-white transition-colors"
                                >
                                    Go
                                </button>
                                <button
                                    type="button"
                                    onClick={() => { setShowSearch(false); setSearchQuery(""); }}
                                    className="text-white/40 hover:text-white/70 text-xs transition-colors"
                                >
                                    ✕
                                </button>
                            </form>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* ── Left panel — presets (docked at bottom-left to prevent overlap with EE panel) ── */}
                {isLoaded && (
                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.4 }}
                        className="absolute left-4 bottom-24 flex flex-col gap-1.5 pointer-events-auto z-30"
                    >
                        {PRESETS.map(preset => (
                            <motion.button
                                key={preset.label}
                                whileHover={{ scale: 1.05, x: 3 }}
                                whileTap={{ scale: 0.97 }}
                                onClick={() => flyTo(preset)}
                                title={preset.label}
                                className={`flex items-center gap-2 rounded-xl px-3 py-2 text-left text-xs font-medium transition-all backdrop-blur-md border
                                    ${currentPreset === preset.label
                                        ? "bg-white/15 border-white/20 text-white shadow-lg"
                                        : "bg-black/50 border-white/[0.07] text-white/70 hover:bg-white/10 hover:text-white"
                                    }`}
                            >
                                <span className="text-sm leading-none">{preset.icon}</span>
                                <span className="hidden md:block">{preset.label}</span>
                            </motion.button>
                        ))}
                    </motion.div>
                )}

                {/* ── Bottom — imagery mode switcher + 3D Tiles ─────────── */}
                {isLoaded && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.4, delay: 0.1 }}
                        className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-2 pointer-events-auto"
                    >
                        {/* Imagery mode pills */}
                        <div className="flex items-center gap-1 rounded-full bg-black/60 border border-white/[0.08] backdrop-blur-xl p-1.5 shadow-2xl">
                            {MODES.map(mode => (
                                <motion.button
                                    key={mode.id}
                                    whileHover={{ scale: 1.03 }}
                                    whileTap={{ scale: 0.97 }}
                                    onClick={() => switchMode(mode.id)}
                                    disabled={tiles3DActive}
                                    className={`flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition-all
                                        ${tiles3DActive
                                            ? "text-white/20 cursor-not-allowed"
                                            : activeMode === mode.id
                                                ? "bg-white text-black shadow"
                                                : "text-white/60 hover:text-white/90"
                                        }`}
                                >
                                    <span>{mode.icon}</span>
                                    <span>{mode.label}</span>
                                </motion.button>
                            ))}
                        </div>

                        {/* Separator */}
                        <div className="h-8 w-px bg-white/10" />

                        {/* 3D Tiles toggle */}
                        <motion.button
                            whileHover={{ scale: 1.04 }}
                            whileTap={{ scale: 0.96 }}
                            onClick={toggle3DTiles}
                            disabled={tiles3DLoading}
                            className={`flex items-center gap-2 rounded-full px-4 py-2.5 text-sm font-semibold transition-all shadow-xl border backdrop-blur-xl
                                ${tiles3DActive
                                    ? "bg-gradient-to-r from-sky-500 to-blue-600 border-sky-400/40 text-white shadow-sky-500/30"
                                    : "bg-black/60 border-white/[0.08] text-white/70 hover:text-white hover:bg-white/10"
                                }`}
                        >
                            {tiles3DLoading ? (
                                <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
                                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" strokeOpacity="0.25" />
                                    <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                                </svg>
                            ) : (
                                <svg viewBox="0 0 20 20" fill="none" className="h-4 w-4">
                                    <path d="M10 2L2 6l8 4 8-4-8-4Z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
                                    <path d="M2 10l8 4 8-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                                    <path d="M2 14l8 4 8-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                                </svg>
                            )}
                            {tiles3DLoading ? "Loading…" : tiles3DActive ? "3D Tiles ON" : "3D Tiles"}
                        </motion.button>
                    </motion.div>
                )}

                {/* 3D Tiles error toast */}
                <AnimatePresence>
                    {tiles3DError && (
                        <motion.div
                            initial={{ opacity: 0, y: 12 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: 12 }}
                            className="absolute bottom-24 left-1/2 -translate-x-1/2 w-full max-w-md px-4 pointer-events-none"
                        >
                            <div className="rounded-2xl bg-red-500/10 border border-red-500/20 backdrop-blur-xl px-4 py-3 text-center">
                                <p className="text-xs text-red-300">{tiles3DError}</p>
                                {tiles3DError.includes("NEXT_PUBLIC") && (
                                    <code className="mt-1 block text-[10px] text-red-400/70 font-mono">
                                        NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=your_key
                                    </code>
                                )}
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* ── Branding ──────────────────────────────────────────── */}
                {isLoaded && (
                    <p className="absolute bottom-6 right-4 text-[10px] font-semibold text-white/25 tracking-widest uppercase">
                        GaiaLens · Explore Earth
                    </p>
                )}

                {/* ── Earth Engine Image Preview Panel (movable & collapsible) ─── */}
                <AnimatePresence>
                    {showImagePanel && (
                        <motion.div
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.95 }}
                            ref={imagePanelRef}
                            style={{ left: imagePanelPosRef.current.x, top: imagePanelPosRef.current.y }}
                            className="absolute w-64 pointer-events-auto z-50 select-none shadow-2xl"
                        >
                            <div className="rounded-2xl bg-neutral-950/90 border border-white/[0.12] backdrop-blur-2xl overflow-hidden shadow-2xl">
                                {/* Header / Drag Handle */}
                                <div
                                    onMouseDown={(e) => {
                                        if ((e.target as HTMLElement).closest("button")) return;
                                        e.preventDefault();
                                        imagePanelDragRef.current = {
                                            active: true,
                                            startX: e.clientX,
                                            startY: e.clientY,
                                            ox: imagePanelPosRef.current.x,
                                            oy: imagePanelPosRef.current.y,
                                        };
                                        document.body.style.userSelect = "none";
                                    }}
                                    className="flex items-center justify-between px-3 py-2.5 border-b border-white/[0.08] cursor-grab active:cursor-grabbing bg-white/[0.04] hover:bg-white/[0.06] transition-colors"
                                >
                                    <div className="flex items-center gap-2">
                                        <span className="text-white/40 text-xs cursor-grab">⠿</span>
                                        <div>
                                            <p className="text-[12px] font-semibold text-white leading-tight">Earth Engine</p>
                                            <p className="text-[9px] text-white/50">Past 4 Years · Sentinel-2</p>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-1">
                                        {/* Minimize / Maximize toggle */}
                                        <button
                                            onClick={() => setImagePanelMinimized(v => !v)}
                                            title={imagePanelMinimized ? "Expand" : "Collapse"}
                                            className="h-6 w-6 rounded-md text-white/50 hover:text-white hover:bg-white/10 flex items-center justify-center text-xs transition-colors"
                                        >
                                            {imagePanelMinimized ? "▼" : "▲"}
                                        </button>
                                        {/* Close */}
                                        <button
                                            onClick={() => setShowImagePanel(false)}
                                            title="Close panel"
                                            className="h-6 w-6 rounded-md text-white/50 hover:text-white hover:bg-white/10 flex items-center justify-center text-xs transition-colors"
                                        >
                                            ✕
                                        </button>
                                    </div>
                                </div>

                                {!imagePanelMinimized && (
                                    <>
                                        {/* 2×2 image grid */}
                                        <div className="p-2 grid grid-cols-2 gap-2">
                                            {geeLoading ? (
                                                Array.from({ length: 4 }).map((_, i) => (
                                                    <div key={i} className="aspect-square rounded-xl bg-white/[0.05] animate-pulse flex flex-col items-center justify-center gap-1 border border-white/[0.04]">
                                                        <div className="h-4 w-4 rounded-full border-2 border-sky-400 border-t-transparent animate-spin" />
                                                        <span className="text-[9px] text-white/30">Loading...</span>
                                                    </div>
                                                ))
                                            ) : geeImages.length > 0 ? (
                                                geeImages.map(img => (
                                                    <button
                                                        key={img.year}
                                                        onClick={() => openLightbox(img)}
                                                        className="relative aspect-square rounded-xl overflow-hidden border border-white/[0.08] hover:border-sky-400/60 hover:shadow-lg hover:shadow-sky-500/20 transition-all group cursor-pointer bg-neutral-900"
                                                    >
                                                        <img
                                                            src={img.url}
                                                            alt={`${img.year}`}
                                                            className="h-full w-full object-cover group-hover:scale-105 transition-transform duration-300"
                                                            onError={e => {
                                                                (e.target as HTMLImageElement).src =
                                                                    `https://placehold.co/200x200/0a0a0a/4a4a4a?text=${img.year}`;
                                                            }}
                                                        />
                                                        <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/85 via-black/40 to-transparent pt-3 pb-1 px-1.5 flex items-center justify-between">
                                                            <span className="text-[11px] font-bold text-white drop-shadow">{img.year}</span>
                                                            <span className="text-[8px] text-white/60 font-mono">S2-RGB</span>
                                                        </div>
                                                        {img.mock && (
                                                            <div className="absolute top-1 left-1 rounded bg-amber-500/90 px-1 text-[8px] text-white font-bold shadow">MOCK</div>
                                                        )}
                                                        {/* Hover zoom overlay */}
                                                        <div className="absolute inset-0 bg-sky-500/15 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                                            <div className="p-1.5 rounded-full bg-black/60 text-white backdrop-blur-sm border border-white/20">
                                                                <svg viewBox="0 0 16 16" fill="none" className="h-4 w-4">
                                                                    <path d="M10.5 10.5L14 14M6.5 11a4.5 4.5 0 1 0 0-9 4.5 4.5 0 0 0 0 9Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                                                                    <path d="M4.5 6.5h4M6.5 4.5v4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                                                                </svg>
                                                            </div>
                                                        </div>
                                                    </button>
                                                ))
                                            ) : (
                                                <div className="col-span-2 py-6 text-center text-[11px] text-white/40">
                                                    No satellite images returned
                                                </div>
                                            )}
                                        </div>

                                        {/* BBox coords & drag hint */}
                                        <div className="px-3 pb-2.5 pt-1 text-[9px] text-white/40 border-t border-white/[0.05] flex items-center justify-between">
                                            {selectedBBox ? (
                                                <span className="font-mono text-[9px]">
                                                    {selectedBBox.south.toFixed(2)}°, {selectedBBox.west.toFixed(2)}° → {selectedBBox.north.toFixed(2)}°, {selectedBBox.east.toFixed(2)}°
                                                </span>
                                            ) : (
                                                <span>Sentinel-2 L2A</span>
                                            )}
                                            <span className="text-[8px] text-white/25">⠿ Drag anywhere</span>
                                        </div>
                                    </>
                                )}
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* ── Expanded image lightbox with Scroll Zoom & Pan ─────────── */}
                <AnimatePresence>
                    {expandedImage && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            onClick={() => setExpandedImage(null)}
                            className="fixed inset-0 bg-black/85 backdrop-blur-md flex items-center justify-center pointer-events-auto z-[250]"
                        >
                            <motion.div
                                initial={{ scale: 0.9, y: 15 }}
                                animate={{ scale: 1, y: 0 }}
                                exit={{ scale: 0.9, y: 15 }}
                                onClick={e => e.stopPropagation()}
                                className="relative flex flex-col w-[92vw] max-w-4xl max-h-[90vh] rounded-3xl overflow-hidden border border-white/[0.15] shadow-2xl bg-neutral-950"
                            >
                                {/* Modal Header */}
                                <div className="flex items-center justify-between px-6 py-4 border-b border-white/[0.08] bg-white/[0.02]">
                                    <div className="flex items-center gap-3">
                                        <div className="px-2.5 py-1 rounded-lg bg-sky-500/20 border border-sky-400/30 text-sky-300 font-bold text-sm">
                                            {expandedImage.year}
                                        </div>
                                        <div>
                                            <h3 className="text-sm font-semibold text-white">
                                                Sentinel-2 Multi-spectral Imagery
                                            </h3>
                                            <p className="text-[11px] text-white/40">
                                                True Color (B4-Red, B3-Green, B2-Blue) · Harmonized L2A
                                            </p>
                                        </div>
                                    </div>

                                    {/* Action Buttons: Zoom HUD + Navigation + Close */}
                                    <div className="flex items-center gap-2">
                                        {/* Zoom Controls */}
                                        <div className="flex items-center gap-1 bg-black/60 border border-white/10 rounded-full px-2 py-1">
                                            <button
                                                onClick={() => setLightboxZoom(z => Math.max(1, +(z - 0.3).toFixed(2)))}
                                                disabled={lightboxZoom <= 1}
                                                className="h-7 w-7 rounded-full text-white/70 hover:text-white disabled:opacity-30 flex items-center justify-center text-sm font-bold transition-colors"
                                                title="Zoom out"
                                            >
                                                −
                                            </button>
                                            <span className="text-xs font-mono text-white/80 px-2 min-w-[50px] text-center">
                                                {Math.round(lightboxZoom * 100)}%
                                            </span>
                                            <button
                                                onClick={() => setLightboxZoom(z => Math.min(6, +(z + 0.3).toFixed(2)))}
                                                disabled={lightboxZoom >= 6}
                                                className="h-7 w-7 rounded-full text-white/70 hover:text-white disabled:opacity-30 flex items-center justify-center text-sm font-bold transition-colors"
                                                title="Zoom in"
                                            >
                                                +
                                            </button>
                                            {lightboxZoom > 1 && (
                                                <button
                                                    onClick={() => { setLightboxZoom(1); setLightboxPan({ x: 0, y: 0 }); }}
                                                    className="text-[10px] text-sky-400 hover:text-sky-300 px-2 font-medium"
                                                >
                                                    Reset
                                                </button>
                                            )}
                                        </div>

                                        {/* Year switchers */}
                                        {geeImages.length > 1 && (
                                            <div className="flex items-center gap-1 border-l border-white/10 pl-2">
                                                <button
                                                    onClick={() => {
                                                        const idx = geeImages.findIndex(i => i.year === expandedImage.year);
                                                        const prev = geeImages[(idx - 1 + geeImages.length) % geeImages.length];
                                                        openLightbox(prev);
                                                    }}
                                                    title="Previous year"
                                                    className="h-8 w-8 rounded-full bg-white/5 hover:bg-white/15 border border-white/10 text-white flex items-center justify-center text-sm transition-colors"
                                                >
                                                    ‹
                                                </button>
                                                <button
                                                    onClick={() => {
                                                        const idx = geeImages.findIndex(i => i.year === expandedImage.year);
                                                        const next = geeImages[(idx + 1) % geeImages.length];
                                                        openLightbox(next);
                                                    }}
                                                    title="Next year"
                                                    className="h-8 w-8 rounded-full bg-white/5 hover:bg-white/15 border border-white/10 text-white flex items-center justify-center text-sm transition-colors"
                                                >
                                                    ›
                                                </button>
                                            </div>
                                        )}

                                        {/* Close */}
                                        <button
                                            onClick={() => setExpandedImage(null)}
                                            className="h-8 w-8 rounded-full bg-white/10 hover:bg-white/20 text-white flex items-center justify-center text-sm transition-colors ml-2"
                                            title="Close (Esc)"
                                        >
                                            ✕
                                        </button>
                                    </div>
                                </div>

                                {/* Image Viewport with Scroll Zoom and Pan */}
                                <div
                                    onWheel={handleLightboxWheel}
                                    onMouseDown={handleLightboxPanStart}
                                    className={`relative flex-1 overflow-hidden flex items-center justify-center bg-black/60 min-h-[380px] max-h-[70vh] select-none ${
                                        lightboxZoom > 1 ? "cursor-grab active:cursor-grabbing" : "cursor-zoom-in"
                                    }`}
                                >
                                    <div
                                        style={{
                                            transform: `scale(${lightboxZoom}) translate(${lightboxPan.x / lightboxZoom}px, ${lightboxPan.y / lightboxZoom}px)`,
                                            transformOrigin: "center center",
                                            transition: lightboxDragRef.current.active ? "none" : "transform 0.15s ease-out",
                                        }}
                                        className="relative max-w-full max-h-full flex items-center justify-center"
                                    >
                                        <img
                                            src={expandedImage.url}
                                            alt={String(expandedImage.year)}
                                            draggable={false}
                                            className="max-w-full max-h-[68vh] object-contain rounded-lg shadow-2xl pointer-events-none"
                                            onError={e => {
                                                (e.target as HTMLImageElement).src =
                                                    `https://placehold.co/600x400/0a0a0a/4a4a4a?text=${expandedImage?.year}`;
                                            }}
                                        />
                                    </div>

                                    {/* Scroll zoom hint badge */}
                                    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2 bg-black/70 border border-white/10 rounded-full px-3 py-1 backdrop-blur-md text-[10px] text-white/50 pointer-events-none">
                                        <svg viewBox="0 0 20 20" fill="none" className="h-3.5 w-3.5 opacity-70">
                                            <circle cx="10" cy="10" r="7" stroke="currentColor" strokeWidth="1.5" />
                                            <path d="M10 6v3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                                        </svg>
                                        <span>Scroll mouse wheel to zoom in/out {lightboxZoom > 1 ? "• Drag to pan" : ""}</span>
                                    </div>
                                </div>

                                {/* Modal Footer */}
                                <div className="px-6 py-3 border-t border-white/[0.08] bg-white/[0.02] flex items-center justify-between text-xs text-white/50">
                                    <div className="flex items-center gap-4 font-mono text-[11px]">
                                        <span>Year: <b className="text-white">{expandedImage.year}</b></span>
                                        <span>Platform: <b className="text-white">Sentinel-2 MSI</b></span>
                                        {selectedBBox && (
                                            <span>
                                                BBox: <b className="text-white">[{selectedBBox.south.toFixed(3)}, {selectedBBox.west.toFixed(3)}] to [{selectedBBox.north.toFixed(3)}, {selectedBBox.east.toFixed(3)}]</b>
                                            </span>
                                        )}
                                    </div>
                                    {expandedImage.mock && (
                                        <span className="text-amber-400 font-medium text-[11px]">
                                            Mock Preview Image
                                        </span>
                                    )}
                                </div>
                            </motion.div>
                        </motion.div>
                    )}
                </AnimatePresence>

            </div>{/* end pointer-events-none overlay */}

            {/* ── Floating AI Chat Window (rendered outside the overlay div) ── */}
            {isLoaded && (
                <FloatingChatWindow
                    title="🌍 GaiaLens Assistant"
                    selectedBBox={selectedBBox}
                />
            )}
        </>
    );
}
