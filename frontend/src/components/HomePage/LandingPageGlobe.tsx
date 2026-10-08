"use client";
import { Globe3D } from "@/ui/3d-globe";

export function Globe() {
    return (
        <Globe3D
            className="h-[620px] w-full"
            markers={[]}
            config={{
                atmosphereColor: "#4da6ff",
                atmosphereIntensity: 20,
                bumpScale: 5,
                autoRotateSpeed: 0.3,
                cameraDistanceMultiplier: 2.9,
            }}
        />
    );
}
