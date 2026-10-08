import type { NextConfig } from "next";

const nextConfig: NextConfig = {
    // Strict Mode causes useEffect to run twice, which creates two Cesium
    // Viewer instances on the same canvas — the root cause of globe flickering.
    reactStrictMode: false,
    transpilePackages: ["resium", "cesium"],

    // @google/earthengine is a CommonJS package with native bindings;
    // it must run as-is in Node.js and must NOT be bundled by Turbopack.
    serverExternalPackages: ["@google/earthengine", "google-auth-library"],
};

export default nextConfig;
