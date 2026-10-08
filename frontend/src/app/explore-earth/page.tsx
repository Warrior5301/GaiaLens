"use client";

import dynamic from "next/dynamic";

// ArcGIS is browser-only — never render on the server
const ArcGISViewer = dynamic(
    () => import("@/components/ExploreEarth/ArcGISViewer"),
    {
        ssr: false,
        loading: () => (
            <div className="flex h-screen w-full items-center justify-center bg-neutral-950">
                <div className="relative h-16 w-16">
                    <div className="absolute inset-0 rounded-full border-2 border-white/10" />
                    <div className="absolute inset-0 rounded-full border-t-2 border-sky-400 animate-spin" />
                </div>
            </div>
        ),
    }
);

export default function ExploreEarthPage() {
    return <ArcGISViewer />;
}

