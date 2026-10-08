"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "motion/react";

export type ImageSlide = {
    src: string;
    name: string;
};

interface AnimatedImageStackProps {
    images: ImageSlide[];
    activeIndex: number;
    onNext: () => void;
    onPrev: () => void;
}

// Rotation offsets for the stack cards behind the active one
// Alternating left-right to create the natural "fanned deck" look
const STACK_ROTATIONS = [0, 4, -4, 8, -8];

export function AnimatedImageStack({
    images,
    activeIndex,
    onNext,
    onPrev,
}: AnimatedImageStackProps) {
    const [direction, setDirection]   = useState(1); // 1 = next, -1 = prev
    const [isAnimating, setIsAnimating] = useState(false);

    const handleNext = useCallback(() => {
        if (isAnimating || images.length <= 1) return;
        setDirection(1);
        setIsAnimating(true);
        onNext();
    }, [isAnimating, images.length, onNext]);

    const handlePrev = useCallback(() => {
        if (isAnimating || images.length <= 1) return;
        setDirection(-1);
        setIsAnimating(true);
        onPrev();
    }, [isAnimating, images.length, onPrev]);

    // Unlock after animation completes
    useEffect(() => {
        if (isAnimating) {
            const t = setTimeout(() => setIsAnimating(false), 500);
            return () => clearTimeout(t);
        }
    }, [isAnimating]);

    // Keyboard navigation
    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            if (e.key === "ArrowRight") handleNext();
            if (e.key === "ArrowLeft")  handlePrev();
        };
        window.addEventListener("keydown", handler);
        return () => window.removeEventListener("keydown", handler);
    }, [handleNext, handlePrev]);

    if (images.length === 0) return null;

    // Build a visible window: active card + up to 4 behind it
    // We show MAX 5 cards in the stack
    const visibleCount = Math.min(images.length, 5);
    const visibleIndices = Array.from({ length: visibleCount }, (_, i) =>
        (activeIndex + i) % images.length
    );

    return (
        <div className="relative h-full w-full flex flex-col select-none">

            {/* ── Card stack ─────────────────────────────────────────────── */}
            <div className="relative flex-1">
                <AnimatePresence initial={false} custom={direction}>
                    {visibleIndices.map((imgIdx, stackPos) => {
                        const img      = images[imgIdx];
                        const isActive = stackPos === 0;
                        const rotation = STACK_ROTATIONS[stackPos] ?? STACK_ROTATIONS[4];
                        const scale    = 1 - stackPos * 0.04;
                        const yOffset  = stackPos * -10; // cards appear behind / upward
                        const opacity  = isActive ? 1 : Math.max(0.35, 1 - stackPos * 0.15);
                        const zIndex   = visibleCount - stackPos;

                        return (
                            <motion.div
                                key={imgIdx + "-" + img.src}
                                className="absolute inset-0 overflow-hidden rounded-2xl"
                                style={{ zIndex, originX: 0.5, originY: 1 }}
                                custom={direction}
                                // Non-active cards just sit in their stack position
                                initial={isActive
                                    ? {
                                        x: direction > 0 ? "100%" : "-100%",
                                        opacity: 0,
                                        rotate: direction > 0 ? 12 : -12,
                                        scale: 0.85,
                                    }
                                    : false
                                }
                                animate={{
                                    x:       0,
                                    y:       yOffset,
                                    rotate:  rotation,
                                    scale,
                                    opacity,
                                }}
                                exit={isActive
                                    ? {
                                        x:       direction > 0 ? "-120%" : "120%",
                                        rotate:  direction > 0 ? -18 : 18,
                                        opacity: 0,
                                        scale:   0.9,
                                        transition: {
                                            duration: 0.45,
                                            ease: [0.32, 0, 0.67, 0],
                                        },
                                    }
                                    : { opacity: 0, scale: 0.8 }
                                }
                                transition={{
                                    duration: isActive ? 0.5 : 0.4,
                                    ease: [0.22, 1, 0.36, 1],
                                }}
                            >
                                <img
                                    src={img.src}
                                    alt={img.name}
                                    draggable={false}
                                    className="h-full w-full object-cover"
                                />

                                {/* Gradient + filename on the active card */}
                                {isActive && (
                                    <motion.div
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        transition={{ delay: 0.25, duration: 0.3 }}
                                        className="pointer-events-none absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-black/70 to-transparent flex items-end px-4 pb-3"
                                    >
                                        <span className="text-xs font-medium text-white/60 truncate max-w-[80%]">
                                            {img.name}
                                        </span>
                                        {images.length > 1 && (
                                            <span className="ml-auto text-[10px] text-white/40">
                                                {activeIndex + 1} / {images.length}
                                            </span>
                                        )}
                                    </motion.div>
                                )}

                                {/* Subtle edge shadow to give depth to stack */}
                                {!isActive && (
                                    <div className="pointer-events-none absolute inset-0 bg-black/20 rounded-2xl" />
                                )}
                            </motion.div>
                        );
                    })}
                </AnimatePresence>
            </div>

            {/* ── Dot indicators ─────────────────────────────────────────── */}
            {images.length > 1 && (
                <div className="flex shrink-0 items-center justify-center gap-1.5 pt-3 pb-1">
                    {images.map((_, i) => (
                        <motion.div
                            key={i}
                            animate={{
                                width:   i === activeIndex ? 20 : 6,
                                opacity: i === activeIndex ? 1 : 0.3,
                            }}
                            transition={{ duration: 0.3 }}
                            className="h-1.5 rounded-full bg-white"
                        />
                    ))}
                </div>
            )}
        </div>
    );
}
