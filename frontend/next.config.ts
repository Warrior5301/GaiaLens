import type { NextConfig } from "next";

const nextConfig: NextConfig = {
    // Strict Mode causes useEffect to run twice, which creates two Cesium
    // Viewer instances on the same canvas — the root cause of globe flickering.
    reactStrictMode: false,
    // Keep the map libraries in one config; Next.js previously had competing JS/TS configs.
    transpilePackages: ["@arcgis/core", "resium", "cesium"],
    images: { unoptimized: true },
    turbopack: {},

    // @google/earthengine is a CommonJS package with native bindings;
    // it must run as-is in Node.js and must NOT be bundled by Turbopack.
    serverExternalPackages: ["@google/earthengine", "google-auth-library"],
};

export default nextConfig;
