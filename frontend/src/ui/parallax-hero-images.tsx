"use client";
import React, { useEffect, useState, useMemo, useCallback, memo } from "react";
import Link from "next/link";
import { motion } from "motion/react";
import {
    useMotionValue,
    useSpring,
    useTransform,
    MotionValue,
} from "framer-motion";
import { cn } from "@/lib/utils";

type ImagePosition = {
    src: string;
    href?: string;
    label?: string;
    position:
    | "left"
    | "right"
    | "top-left"
    | "top-right"
    | "mid-left"
    | "mid-right"
    | "bottom-center"
    | "bottom-left"
    | "bottom-right"
    | "far-left"
    | "far-right";
    depth: number;
    delay: number;
};

const positionStyles: Record<
    ImagePosition["position"],
    { top?: string; bottom?: string; left?: string; right?: string; transform?: string }
> = {
    "left": { top: "50%", left: "6.5%", transform: "translateY(-50%)" },
    "right": { top: "50%", right: "6.5%", transform: "translateY(-50%)" },
    "top-left": { top: "7%", left: "7.5%" },
    "top-right": { top: "7%", right: "7.5%" },
    "mid-left": { top: "40%", left: "2%" },
    "mid-right": { top: "40%", right: "2%" },
    "bottom-center": { top: "calc(73% - 6px)", left: "50%", transform: "translateX(-50%)" },
    "bottom-left": { top: "68%", left: "4%" },
    "bottom-right": { top: "68%", right: "4%" },
    "far-left": { top: "52%", left: "2%" },
    "far-right": { top: "52%", right: "2%" },
};

const positionOrder: ImagePosition["position"][] = [
    "left",
    "right",
    "top-left",
    "top-right",
    "mid-left",
    "mid-right",
    "bottom-center",
    "bottom-left",
    "bottom-right",
    "far-left",
    "far-right",
];

type DepthVariant = "default" | "edge-focus";

const depthValuesByVariant: Record<DepthVariant, number[]> = {
    default: [0.5, 0.5, 0.35, 0.35, 0.85, 0.85, 0.4, 0.45],
    "edge-focus": [0.6, 0.6, 0.85, 0.85, 0.35, 0.35, 0.8, 0.85],
};

const SPRING_CONFIG = { damping: 25, stiffness: 120 };

export interface ParallaxHeroImagesProps {
    images: string[];
    hrefs?: string[];
    labels?: string[];
    className?: string;
    imageClassName?: string;
    variant?: DepthVariant;
}

export const ParallaxHeroImages = ({
    images,
    hrefs,
    labels,
    className,
    imageClassName,
    variant = "default",
}: ParallaxHeroImagesProps) => {
    const mouseX = useMotionValue(0);
    const mouseY = useMotionValue(0);

    const smoothMouseX = useSpring(mouseX, SPRING_CONFIG);
    const smoothMouseY = useSpring(mouseY, SPRING_CONFIG);

    const positions = useMemo(() => {
        const limitedImages = images.slice(0, 8);
        const depthValues = depthValuesByVariant[variant];
        return limitedImages.map((src, index) => ({
            src,
            href: hrefs?.[index],
            label: labels?.[index],
            position: positionOrder[index],
            depth: depthValues[index],
            delay: index * 0.12,
        }));
    }, [images, hrefs, labels, variant]);

    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            const x = (e.clientX / window.innerWidth) * 2 - 1;
            const y = (e.clientY / window.innerHeight) * 2 - 1;
            mouseX.set(x);
            mouseY.set(y);
        };

        window.addEventListener("mousemove", handleMouseMove);
        return () => window.removeEventListener("mousemove", handleMouseMove);
    }, [mouseX, mouseY]);

    return (
        <div
            className={cn(
                // pointer-events-auto so images are clickable
                "pointer-events-auto absolute inset-0 overflow-hidden",
                className,
            )}
        >
            {positions.map((pos, index) => (
                <ParallaxImage
                    key={`${pos.src}-${index}`}
                    src={pos.src}
                    href={pos.href}
                    label={pos.label}
                    position={pos.position}
                    depth={pos.depth}
                    delay={pos.delay}
                    imageClassName={imageClassName}
                    smoothMouseX={smoothMouseX}
                    smoothMouseY={smoothMouseY}
                />
            ))}
        </div>
    );
};

interface ParallaxImageProps extends ImagePosition {
    imageClassName?: string;
    smoothMouseX: MotionValue<number>;
    smoothMouseY: MotionValue<number>;
}

const ParallaxImage = memo(function ParallaxImage({
    src,
    href,
    label,
    position,
    depth,
    delay,
    imageClassName,
    smoothMouseX,
    smoothMouseY,
}: ParallaxImageProps) {
    const maxOffset = 40;

    const translateX = useTransform(
        smoothMouseX,
        [-1, 1],
        [-maxOffset * depth, maxOffset * depth],
    );

    const translateY = useTransform(
        smoothMouseY,
        [-1, 1],
        [-maxOffset * depth, maxOffset * depth],
    );

    const posStyle = positionStyles[position];

    const content = (
        <div className="group relative">
            <img
                src={src}
                alt={label ?? ""}
                loading="lazy"
                decoding="async"
                className={cn(
                    "aspect-4/3 h-20 w-32 rounded-xl object-cover shadow-2xl ring-1 ring-black/10 sm:h-36 sm:w-52 md:h-44 md:w-64 lg:h-48 lg:w-72 xl:h-52 xl:w-80 dark:ring-white/10",
                    "transition-all duration-300 group-hover:brightness-75 group-hover:scale-[1.03] group-hover:shadow-xl group-hover:ring-white/30",
                    imageClassName,
                )}
            />

            {/* Label overlay — appears on hover */}
            {label && (
                <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                    <span className="rounded-full bg-white/10 backdrop-blur-md border border-white/20 px-3 py-1 text-[10px] font-semibold uppercase tracking-widest text-white sm:text-xs">
                        {label}
                    </span>
                    <span className="mt-1.5 text-[9px] text-white/60 sm:text-[10px]">
                        Open →
                    </span>
                </div>
            )}

            {/* Subtle glow ring on hover */}
            <div className="pointer-events-none absolute inset-0 rounded-lg ring-2 ring-sky-400/0 group-hover:ring-sky-400/40 transition-all duration-300" />
        </div>
    );

    return (
        <div
            className="absolute"
            style={{
                top: posStyle.top,
                left: posStyle.left,
                right: posStyle.right,
                transform: posStyle.transform,
                zIndex: Math.round(depth * 10),
            }}
        >
            <motion.div
                style={{
                    x: translateX,
                    y: translateY,
                }}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{
                    duration: 0.6,
                    delay: delay,
                    ease: [0.25, 0.1, 0.25, 1],
                }}
            >
                {href ? (
                    <Link href={href} className="block cursor-pointer">
                        {content}
                    </Link>
                ) : (
                    content
                )}
            </motion.div>
        </div>
    );
});
