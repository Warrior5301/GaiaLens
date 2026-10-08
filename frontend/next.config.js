/** @type {import('next').NextConfig} */
const nextConfig = {
  // Let Turbopack transpile the ArcGIS core ES modules
  transpilePackages: ["@arcgis/core"],

  // Allow external image URLs (Earth Engine tiles etc.)
  images: { unoptimized: true },

  // Empty turbopack config silences the webpack‑turbopack conflict warning
  turbopack: {},

  // Fallback: disable Turbopack if disk space is insufficient
  experimental: {
    turbo: false,
  },
};

module.exports = nextConfig;
