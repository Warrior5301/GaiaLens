"use client";

import { motion } from "motion/react";
import { ParallaxHeroImages } from "@/ui/parallax-hero-images";
import { Globe } from "./LandingPageGlobe";

// ─── Stepped connector lines ──────────────────────────────────────────────────
// viewBox matches a 16:9 (1440×810) reference screen so coordinates map 1:1
// to the actual parallax image positions (position styles × md card sizes).
//
// Card at md: w-80 (320px) × h-52 (208px). Globe centre = (720, 405).
// Both cards are vertically centered at y = 405 (50%).
// Left card right edge = 370. Right card left edge = 1070.

function SteppedConnectors() {
    const s = {
        fill: "none",
        stroke: "rgba(255,255,255,0.18)",
        strokeWidth: "1",
        strokeLinecap: "round" as const,
        strokeLinejoin: "round" as const,
    };

    const ease = "easeOut";

    return (
        <svg
            className="pointer-events-none absolute inset-0 h-full w-full z-[5]"
            viewBox="0 0 1440 810"
            preserveAspectRatio="none"
            aria-hidden="true"
        >
            {/* left (Explore Earth) → right=414, cy=405 → globe(720, 405) */}
            <motion.path
                d="M 414,405  H 720"
                style={s}
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: 1, opacity: 1 }}
                transition={{ duration: 1.1, delay: 0.3, ease }}
            />

            {/* right (General Use) → left=1026, cy=405 → globe(720, 405) */}
            <motion.path
                d="M 1026,405  H 720"
                style={s}
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: 1, opacity: 1 }}
                transition={{ duration: 1.1, delay: 0.3, ease }}
            />

            {/* Dot at globe centre */}
            <motion.circle
                cx="720"
                cy="405"
                r="3"
                fill="rgba(255,255,255,0.25)"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1.1, duration: 0.4 }}
            />
        </svg>
    );
}


// ─── Landing page ─────────────────────────────────────────────────────────────
export function LandingPage() {
    return (
        <div className="relative flex min-h-screen w-full items-center justify-center overflow-hidden bg-neutral-950">

            {/* Layer 1 — Parallax floating images, z-[9] so they sit above globe+vignette */}
            <div className="absolute inset-0 z-[9]">
                <ParallaxHeroImages images={images} hrefs={hrefs} labels={labels} />
            </div>

            {/* Layer 2 — Stepped connector lines */}
            <SteppedConnectors />

            {/* Layer 3 — 3-D Globe centered in the background */}
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-[6]">
                <Globe />
            </div>

            {/* Subtle vignette so text is readable over the globe */}
            <div className="pointer-events-none absolute inset-0 z-[7] bg-[radial-gradient(ellipse_65%_60%_at_50%_50%,transparent_25%,rgba(0,0,0,0.45)_100%)]" />
            <div className="pointer-events-none absolute inset-x-0 top-0 h-32 z-[7] bg-gradient-to-b from-neutral-950/80 to-transparent" />
            <div className="pointer-events-none absolute inset-x-0 bottom-0 h-32 z-[7] bg-gradient-to-t from-neutral-950/80 to-transparent" />

            {/* Layer 4 — Text overlay */}
            <div className="relative z-[8] mx-auto flex max-w-xl flex-col items-center gap-5 px-4 text-center pointer-events-none select-none -translate-y-12">

                {/* Heading slides in from top */}
                <motion.h1
                    initial={{ opacity: 0, y: -52 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.85, ease: [0.22, 1, 0.36, 1], delay: 0.15 }}
                    className="text-3xl font-bold tracking-tight text-white drop-shadow-[0_0_40px_rgba(0,0,0,0.95)] sm:text-4xl md:text-5xl lg:text-[3.25rem] lg:leading-[1.12]"
                >
                    Understand Earth,{" "}
                    <br className="hidden sm:block" />
                    through AI analysis,{" "}
                    <br className="hidden sm:block" />
                    <span className="bg-gradient-to-r from-sky-400 to-blue-400 bg-clip-text text-transparent">
                        from above.
                    </span>
                </motion.h1>

                {/* Paragraph slides in from bottom */}
                <motion.p
                    initial={{ opacity: 0, y: 52 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.85, ease: [0.22, 1, 0.36, 1], delay: 0.35 }}
                    className="max-w-md text-xs sm:text-sm text-neutral-300 drop-shadow-[0_2px_12px_rgba(0,0,0,1)] md:text-base leading-relaxed"
                >
                    Upload a satellite image, ask a question in natural language, and get visual answers,
                    grounded objects, and detailed insights.
                </motion.p>
            </div>
        </div>
    );
}

// 2 cards: Explore Earth on left, General Use on right
const images = [
    "/images/ExploreEarth.png",
    "/images/General-Purpose.png",
];

const hrefs = [
    "/explore-earth",
    "/general-use",
];

const labels = [
    "Explore Earth",
    "General Use",
];
