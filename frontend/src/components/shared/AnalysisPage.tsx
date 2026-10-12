"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
    ZoomIn,
    ZoomOut,
    RotateCcw,
    Maximize2,
    ExternalLink,
    X,
    Image as ImageIcon,
    Paperclip,
    AlertCircle,
    Loader2,
} from "lucide-react";
import { AnimatedImageStack, ImageSlide } from "@/ui/animated-testimonials";
import { cn } from "@/lib/utils";
import { sendChat } from "@/lib/chat";

// ─── Types ────────────────────────────────────────────────────────────────────
export interface ChatAttachment {
    url: string;
    alt?: string;
    name?: string;
}

export interface ChatMessage {
    id: string;
    role: "user" | "assistant";
    content: string;
    imageUrl?: string;
    imageUrls?: string[];
    images?: (string | ChatAttachment)[];
    timestamp: Date;
}

interface AnalysisPageProps {
    title: string;
    subtitle: string;
    accentColor: string; // tailwind gradient class e.g. "from-emerald-500 to-teal-400"
    placeholder?: string;
}

interface LightboxItem {
    src: string;
    alt?: string;
    name?: string;
}

// ─── Helper: Parse text & extract embedded / attached images ──────────────────
function parseChatMessage(msg: ChatMessage): {
    textContent: string;
    extractedImages: ChatAttachment[];
} {
    const list: ChatAttachment[] = [];

    // 1. Explicit properties
    if (msg.imageUrl) {
        list.push({ url: msg.imageUrl, alt: "Attached Image" });
    }
    if (Array.isArray(msg.imageUrls)) {
        msg.imageUrls.forEach((u, i) => {
            if (u && !list.some(x => x.url === u)) {
                list.push({ url: u, alt: `Attached Image ${i + 1}` });
            }
        });
    }
    if (Array.isArray(msg.images)) {
        msg.images.forEach((item, i) => {
            if (typeof item === "string" && item) {
                if (!list.some(x => x.url === item)) {
                    list.push({ url: item, alt: `Attached Image ${i + 1}` });
                }
            } else if (item && typeof item === "object" && item.url) {
                if (!list.some(x => x.url === item.url)) {
                    list.push({ url: item.url, alt: item.alt || item.name || `Image ${i + 1}`, name: item.name });
                }
            }
        });
    }

    let text = msg.content || "";

    // 2. Markdown images: ![alt](url)
    const mdImgRegex = /!\[([^\]]*)\]\(([^)]+)\)/g;
    let mdMatch;
    while ((mdMatch = mdImgRegex.exec(text)) !== null) {
        const alt = mdMatch[1] || "Image";
        const url = mdMatch[2].trim();
        if (url && !list.some(x => x.url === url)) {
            list.push({ url, alt });
        }
    }
    text = text.replace(mdImgRegex, "").trim();

    // 3. HTML img tags: <img ... src="..." ... />
    const htmlImgRegex = /<img\s+[^>]*src=["']([^"']+)["'][^>]*\/?>/gi;
    let htmlMatch;
    while ((htmlMatch = htmlImgRegex.exec(text)) !== null) {
        const url = htmlMatch[1].trim();
        if (url && !list.some(x => x.url === url)) {
            list.push({ url, alt: "Satellite Image" });
        }
    }
    text = text.replace(htmlImgRegex, "").trim();

    // 4. Standalone direct image URLs or data URLs on their own line
    const lines = text.split("\n");
    const remainingLines: string[] = [];
    for (const line of lines) {
        const trimmed = line.trim();
        if (
            trimmed.startsWith("data:image/") ||
            /^https?:\/\/[^\s]+?\.(png|jpe?g|webp|gif|svg)(\?[^\s]*)?$/i.test(trimmed) ||
            /^\/images\/[^\s]+?\.(png|jpe?g|webp|gif|svg)$/i.test(trimmed)
        ) {
            if (!list.some(x => x.url === trimmed)) {
                list.push({ url: trimmed, alt: "Satellite Image" });
            }
        } else {
            remainingLines.push(line);
        }
    }
    text = remainingLines.join("\n").trim();

    return { textContent: text, extractedImages: list };
}

// ─── Chat Image Preview Card ──────────────────────────────────────────────────
function ChatImageAttachmentView({
    attachment,
    onOpenLightbox,
}: {
    attachment: ChatAttachment;
    onOpenLightbox: (img: LightboxItem) => void;
}) {
    const [isLoading, setIsLoading] = useState(true);
    const [hasError, setHasError] = useState(false);

    return (
        <div
            className="group relative mt-2 overflow-hidden rounded-xl border border-white/10 bg-black/40 shadow-lg cursor-pointer transition-all hover:border-cyan-400/60 hover:shadow-[0_0_20px_rgba(6,182,212,0.25)]"
            onClick={e => {
                e.stopPropagation();
                if (!hasError) {
                    onOpenLightbox({
                        src: attachment.url,
                        alt: attachment.alt,
                        name: attachment.name,
                    });
                }
            }}
        >
            {/* Loading shimmer skeleton */}
            {isLoading && !hasError && (
                <div className="flex h-44 w-full items-center justify-center bg-white/5 animate-pulse">
                    <Loader2 className="h-6 w-6 text-white/40 animate-spin" />
                </div>
            )}

            {/* Error fallback */}
            {hasError ? (
                <div className="flex h-32 w-full flex-col items-center justify-center gap-2 p-3 text-center bg-red-950/20 border border-red-500/20">
                    <AlertCircle className="h-6 w-6 text-red-400" />
                    <p className="text-xs text-red-300 font-medium">Image could not be loaded</p>
                    <p className="text-[10px] text-white/30 truncate max-w-[200px]">{attachment.url}</p>
                </div>
            ) : (
                <div className="relative overflow-hidden">
                    <img
                        src={attachment.url}
                        alt={attachment.alt || "Chat attachment"}
                        onLoad={() => setIsLoading(false)}
                        onError={() => {
                            setIsLoading(false);
                            setHasError(true);
                        }}
                        className={cn(
                            "max-h-56 w-full object-cover transition-transform duration-300 group-hover:scale-105",
                            isLoading ? "hidden" : "block"
                        )}
                    />

                    {/* Interactive overlay on hover with zoom badge */}
                    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2 backdrop-blur-[2px]">
                        <span className="flex items-center gap-1.5 rounded-full bg-black/80 border border-white/20 px-3 py-1.5 text-xs font-medium text-white shadow-xl">
                            <Maximize2 className="h-3.5 w-3.5 text-cyan-400" />
                            Click to view larger
                        </span>
                    </div>

                    {/* Bottom caption pill if available */}
                    {attachment.alt && attachment.alt !== "Image" && (
                        <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 to-transparent px-2.5 py-1.5">
                            <p className="text-[11px] text-white/80 truncate font-medium">
                                {attachment.alt}
                            </p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

// ─── Chat Message Bubble ──────────────────────────────────────────────────────
function ChatBubble({
    msg,
    onOpenLightbox,
}: {
    msg: ChatMessage;
    onOpenLightbox: (img: LightboxItem) => void;
}) {
    const isUser = msg.role === "user";
    const { textContent, extractedImages } = parseChatMessage(msg);

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.25 }}
            className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"}`}
        >
            {/* Avatar */}
            <div className={`h-7 w-7 shrink-0 rounded-full flex items-center justify-center text-[11px] font-bold overflow-hidden
                ${isUser ? "bg-white/10 text-white/70" : "border border-white/15 bg-neutral-900"}`}>
                {isUser ? (
                    "U"
                ) : (
                    <img src="/logo.png" alt="GaiaLens" className="h-full w-full object-cover" />
                )}
            </div>

            {/* Bubble */}
            <div className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed
                ${isUser
                    ? "bg-white/10 text-white/90 rounded-tr-sm"
                    : "bg-neutral-800 text-neutral-100 rounded-tl-sm border border-white/[0.06]"
                }`}>

                {/* Text content */}
                {textContent && (
                    <div className="whitespace-pre-wrap break-words">{textContent}</div>
                )}

                {/* Attached / extracted images */}
                {extractedImages.length > 0 && (
                    <div className={`space-y-2 ${textContent ? "mt-2" : ""}`}>
                        {extractedImages.map((img, idx) => (
                            <ChatImageAttachmentView
                                key={idx}
                                attachment={img}
                                onOpenLightbox={onOpenLightbox}
                            />
                        ))}
                    </div>
                )}
            </div>
        </motion.div>
    );
}

// ─── Image Lightbox Modal (Enlarged Full-Screen View) ─────────────────────────
function ImageLightboxModal({
    image,
    onClose,
}: {
    image: LightboxItem | null;
    onClose: () => void;
}) {
    const [zoom, setZoom] = useState(1);
    const [pan, setPan] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    const dragStartRef = useRef({ x: 0, y: 0, panX: 0, panY: 0 });

    // Reset zoom and pan whenever image changes
    useEffect(() => {
        setZoom(1);
        setPan({ x: 0, y: 0 });
    }, [image?.src]);

    // Keyboard shortcuts: Escape to close, +/- to zoom, 0 to reset
    useEffect(() => {
        if (!image) return;
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === "Escape") {
                onClose();
            } else if (e.key === "+" || e.key === "=") {
                setZoom(z => Math.min(z + 0.25, 4));
            } else if (e.key === "-") {
                setZoom(z => Math.max(z - 0.25, 0.5));
            } else if (e.key === "0") {
                setZoom(1);
                setPan({ x: 0, y: 0 });
            }
        };
        window.addEventListener("keydown", handleKeyDown);
        return () => window.removeEventListener("keydown", handleKeyDown);
    }, [image, onClose]);

    if (!image) return null;

    const handleWheel = (e: React.WheelEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setZoom(z => Math.min(4, Math.max(0.5, z - e.deltaY * 0.0015)));
    };

    const handleMouseDown = (e: React.MouseEvent) => {
        if (zoom <= 1) return;
        setIsDragging(true);
        dragStartRef.current = {
            x: e.clientX,
            y: e.clientY,
            panX: pan.x,
            panY: pan.y,
        };
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (!isDragging) return;
        setPan({
            x: dragStartRef.current.panX + (e.clientX - dragStartRef.current.x),
            y: dragStartRef.current.panY + (e.clientY - dragStartRef.current.y),
        });
    };

    const handleMouseUp = () => {
        setIsDragging(false);
    };

    const handleReset = () => {
        setZoom(1);
        setPan({ x: 0, y: 0 });
    };

    const handleDoubleClick = () => {
        if (zoom > 1.2) {
            handleReset();
        } else {
            setZoom(2);
        }
    };

    return (
        <AnimatePresence>
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 z-[100] flex flex-col bg-black/95 backdrop-blur-md select-none"
                onClick={onClose}
            >
                {/* Header Controls */}
                <div
                    className="flex h-16 items-center justify-between border-b border-white/10 px-6 shrink-0 bg-neutral-950/80 backdrop-blur-sm z-10"
                    onClick={e => e.stopPropagation()}
                >
                    <div className="flex items-center gap-3 min-w-0">
                        <div className="h-8 w-8 rounded-lg bg-white/10 flex items-center justify-center text-white/70 shrink-0">
                            <ImageIcon className="h-4 w-4" />
                        </div>
                        <div className="min-w-0">
                            <h3 className="text-sm font-semibold text-white truncate max-w-xs md:max-w-md">
                                {image.name || image.alt || "Satellite Image View"}
                            </h3>
                            <p className="text-[11px] text-white/40">
                                Double-click to toggle zoom · Drag to pan · Scroll to zoom
                            </p>
                        </div>
                    </div>

                    {/* Toolbar */}
                    <div className="flex items-center gap-2">
                        {/* Zoom buttons */}
                        <div className="flex items-center gap-1 rounded-lg bg-white/[0.07] border border-white/10 p-1">
                            <button
                                onClick={() => setZoom(z => Math.max(z - 0.25, 0.5))}
                                title="Zoom Out (-)"
                                className="h-7 w-7 rounded flex items-center justify-center text-white/70 hover:bg-white/10 hover:text-white transition-colors"
                            >
                                <ZoomOut className="h-3.5 w-3.5" />
                            </button>
                            <button
                                onClick={handleReset}
                                title="Reset Zoom (0)"
                                className="px-2 text-xs font-mono text-white/80 hover:text-white transition-colors min-w-[3.5rem] text-center"
                            >
                                {Math.round(zoom * 100)}%
                            </button>
                            <button
                                onClick={() => setZoom(z => Math.min(z + 0.25, 4))}
                                title="Zoom In (+)"
                                className="h-7 w-7 rounded flex items-center justify-center text-white/70 hover:bg-white/10 hover:text-white transition-colors"
                            >
                                <ZoomIn className="h-3.5 w-3.5" />
                            </button>
                        </div>

                        {/* Reset View */}
                        <button
                            onClick={handleReset}
                            title="Reset View"
                            className="h-8 w-8 rounded-lg bg-white/[0.07] border border-white/10 flex items-center justify-center text-white/70 hover:bg-white/10 hover:text-white transition-colors"
                        >
                            <RotateCcw className="h-3.5 w-3.5" />
                        </button>

                        {/* Open Original */}
                        <a
                            href={image.src}
                            target="_blank"
                            rel="noopener noreferrer"
                            title="Open original in new tab"
                            className="h-8 w-8 rounded-lg bg-white/[0.07] border border-white/10 flex items-center justify-center text-white/70 hover:bg-white/10 hover:text-white transition-colors"
                        >
                            <ExternalLink className="h-3.5 w-3.5" />
                        </a>

                        {/* Close button */}
                        <button
                            onClick={onClose}
                            title="Close (Esc)"
                            className="h-8 w-8 rounded-lg bg-red-500/20 border border-red-500/30 flex items-center justify-center text-red-200 hover:bg-red-500 hover:text-white transition-colors ml-2"
                        >
                            <X className="h-4 w-4" />
                        </button>
                    </div>
                </div>

                {/* Viewport Canvas */}
                <div
                    className="relative flex-1 flex items-center justify-center overflow-hidden p-4 cursor-default"
                    onWheel={handleWheel}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseUp}
                    onClick={e => {
                        if (e.target === e.currentTarget) onClose();
                    }}
                >
                    <motion.div
                        className="relative max-w-full max-h-full flex items-center justify-center"
                        style={{
                            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                            cursor: zoom > 1 ? (isDragging ? "grabbing" : "grab") : "zoom-in",
                            transition: isDragging ? "none" : "transform 0.15s ease-out",
                        }}
                        onClick={e => e.stopPropagation()}
                        onDoubleClick={handleDoubleClick}
                    >
                        <img
                            src={image.src}
                            alt={image.alt || "Satellite Image"}
                            className="max-h-[82vh] max-w-[88vw] object-contain rounded-lg shadow-2xl pointer-events-auto select-none border border-white/10"
                            draggable={false}
                        />
                    </motion.div>
                </div>

                {/* Footer Bar */}
                <div
                    className="h-10 flex items-center justify-between border-t border-white/10 px-6 text-[11px] text-white/40 shrink-0 bg-neutral-950/80 backdrop-blur-sm"
                    onClick={e => e.stopPropagation()}
                >
                    <span>High-Resolution View</span>
                    <span>Use mouse wheel to zoom · Drag image to pan · Double-click to toggle 2x · Press Esc to exit</span>
                    <button
                        onClick={onClose}
                        className="text-white/60 hover:text-white transition-colors"
                    >
                        Close [Esc]
                    </button>
                </div>
            </motion.div>
        </AnimatePresence>
    );
}

// ─── Main Shared Component ────────────────────────────────────────────────────
export function AnalysisPage({
    title,
    subtitle,
    accentColor,
    placeholder,
}: AnalysisPageProps) {
    const [lightboxImage, setLightboxImage] = useState<LightboxItem | null>(null);

    // ── Image State ───────────────────────────────────────────────────────────
    const [images, setImages] = useState<ImageSlide[]>([]);
    const [activeIndex, setActiveIndex] = useState(0);
    const [zoom, setZoom] = useState(1);
    const [isDroppingFile, setIsDroppingFile] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const dropZoneRef = useRef<HTMLDivElement>(null);

    // ── Pan/drag refs ─────────────────────────────────────────────────────────
    const imageWrapRef = useRef<HTMLDivElement>(null);
    const zoomRef = useRef(1);
    const panRef = useRef({ x: 0, y: 0 });
    const dragRef = useRef({ active: false, startX: 0, startY: 0, panX: 0, panY: 0 });
    const [isPanDragging, setIsPanDragging] = useState(false);

    const applyTransform = useCallback(() => {
        if (!imageWrapRef.current) return;
        const { x, y } = panRef.current;
        imageWrapRef.current.style.transform =
            `translate(${x}px, ${y}px) scale(${zoomRef.current})`;
    }, []);

    // ── Chat State ────────────────────────────────────────────────────────────
    const [messages, setMessages] = useState<ChatMessage[]>([
        {
            id: "welcome",
            role: "assistant",
            content: `Hi! I'm your GaiaLens analyst for ${title}. Upload a satellite image or ask me anything. You can also send or receive images directly in this chat!`,
            timestamp: new Date(),
        },
    ]);
    const [input, setInput] = useState("");
    const [isTyping, setIsTyping] = useState(false);
    const [pendingChatImage, setPendingChatImage] = useState<{ src: string; name: string; file: File } | null>(null);
    const chatFileInputRef = useRef<HTMLInputElement>(null);
    const chatBottomRef = useRef<HTMLDivElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    // ── Auto-scroll chat ──────────────────────────────────────────────────────
    useEffect(() => {
        chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, isTyping]);

    // ── Image upload handlers for workspace ───────────────────────────────────
    const handleFiles = useCallback((files: FileList | null) => {
        if (!files) return;
        const newSlides: ImageSlide[] = Array.from(files)
            .filter(f => f.type.startsWith("image/"))
            .map(f => ({ src: URL.createObjectURL(f), name: f.name }));
        if (!newSlides.length) return;
        setImages(prev => {
            const updated = [...prev, ...newSlides];
            setActiveIndex(updated.length - newSlides.length);
            return updated;
        });
        setZoom(1);
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDroppingFile(false);
        handleFiles(e.dataTransfer.files);
    }, [handleFiles]);

    const removeImage = useCallback((idx: number) => {
        setImages(prev => {
            const next = prev.filter((_, i) => i !== idx);
            setActiveIndex(i => Math.min(i, Math.max(0, next.length - 1)));
            return next;
        });
    }, []);

    // ── Zoom controls for workspace ───────────────────────────────────────────
    const setZoomAndApply = useCallback((updater: (z: number) => number) => {
        setZoom(prev => {
            const next = updater(prev);
            zoomRef.current = next;
            requestAnimationFrame(applyTransform);
            return next;
        });
    }, [applyTransform]);

    const zoomIn = () => setZoomAndApply(z => Math.min(z + 0.25, 4));
    const zoomOut = () => setZoomAndApply(z => Math.max(z - 0.25, 0.5));

    const zoomReset = useCallback(() => {
        panRef.current = { x: 0, y: 0 };
        zoomRef.current = 1;
        setZoom(1);
        if (imageWrapRef.current) {
            imageWrapRef.current.style.transition = "transform 0.35s cubic-bezier(0.22,1,0.36,1)";
            imageWrapRef.current.style.transform = "translate(0px, 0px) scale(1)";
            setTimeout(() => {
                if (imageWrapRef.current)
                    imageWrapRef.current.style.transition = "";
            }, 380);
        }
    }, []);

    const isViewModified = zoom !== 1;

    const handleWheel = useCallback((e: React.WheelEvent) => {
        e.preventDefault();
        setZoomAndApply(z => Math.min(4, Math.max(0.5, z - e.deltaY * 0.001)));
    }, [setZoomAndApply]);

    // ── Workspace pan / drag ──────────────────────────────────────────────────
    const handlePanMouseDown = useCallback((e: React.MouseEvent) => {
        if (e.button !== 0) return;
        dragRef.current = {
            active: true,
            startX: e.clientX,
            startY: e.clientY,
            panX: panRef.current.x,
            panY: panRef.current.y,
        };
        setIsPanDragging(true);
        e.preventDefault();
    }, []);

    useEffect(() => {
        const onMove = (e: MouseEvent) => {
            if (!dragRef.current.active) return;
            panRef.current = {
                x: dragRef.current.panX + (e.clientX - dragRef.current.startX),
                y: dragRef.current.panY + (e.clientY - dragRef.current.startY),
            };
            applyTransform();
        };
        const onUp = () => {
            if (!dragRef.current.active) return;
            dragRef.current.active = false;
            setIsPanDragging(false);
        };
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);
        return () => {
            window.removeEventListener("mousemove", onMove);
            window.removeEventListener("mouseup", onUp);
        };
    }, [applyTransform]);

    // ── Paste image directly into chat ────────────────────────────────────────
    const handleChatPaste = (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
        const items = e.clipboardData?.items;
        if (!items) return;
        for (let i = 0; i < items.length; i++) {
            if (items[i].type.startsWith("image/")) {
                const file = items[i].getAsFile();
                if (file) {
                    const src = URL.createObjectURL(file);
                    setPendingChatImage({ src, name: file.name || "Pasted satellite image", file });
                    e.preventDefault();
                    break;
                }
            }
        }
    };

    // ── Helper: Resolve image file from File or URL ───────────────────────────
    const resolveImageBlob = async (
        file?: File | null,
        src?: string | null,
        filename = "satellite-image.png"
    ): Promise<Blob | null> => {
        if (file) return file;
        if (!src) return null;
        try {
            const res = await fetch(src);
            if (!res.ok) throw new Error("Image download failed");
            const blob = await res.blob();
            return new File([blob], filename, { type: blob.type || "image/png" });
        } catch (err) {
            console.error("Failed to load image blob from URL:", src, err);
            return null;
        }
    };

    // ── Chat send ─────────────────────────────────────────────────────────────
    const sendMessage = useCallback(async () => {
        const text = input.trim();
        const attachedImg = pendingChatImage;
        if (isTyping || (!text && !attachedImg && !images.length)) return;

        setInput("");
        setPendingChatImage(null);
        const query = text || "Analyze the attached imagery.";
        const userMsg: ChatMessage = {
            id: Date.now().toString(), role: "user", content: query,
            imageUrl: attachedImg?.src, timestamp: new Date(),
        };
        setMessages(prev => [...prev, userMsg]);
        setIsTyping(true);
        try {
            // Send the selected image first, then the rest for comparison requests.
            const workspace = images.length
                ? [images[activeIndex], ...images.filter((_, index) => index !== activeIndex)]
                : [];
            const inputs: { src: string; name: string; file?: File }[] = attachedImg ? [attachedImg, ...workspace] : workspace;
            // Keep a chat-only attachment available for follow-up questions.
            if (!inputs.length) {
                const lastImage = [...messages].reverse().find(msg => msg.role === "user" && msg.imageUrl);
                if (lastImage?.imageUrl) inputs.push({ src: lastImage.imageUrl, name: "previous-upload.png" });
            }
            const files: File[] = [];
            for (const image of inputs) {
                const blob = await resolveImageBlob(image.file, image.src, image.name);
                if (!blob) throw new Error(`Could not load ${image.name || "the attached image"}. Please attach it again.`);
                files.push(new File([blob], image.name || "satellite.png", { type: blob.type }));
            }
            const reply = await sendChat(query, messages.filter(msg => msg.id !== "welcome"), files);
            setMessages(prev => [...prev, {
                id: `${Date.now()}-reply`, role: "assistant", content: reply.content,
                images: reply.images, timestamp: new Date(),
            }]);
        } catch (error) {
            setMessages(prev => [...prev, {
                id: `${Date.now()}-error`, role: "assistant",
                content: error instanceof Error ? error.message : "Unable to reach the assistant.",
                timestamp: new Date(),
            }]);
        } finally {
            setIsTyping(false);
        }
    }, [input, pendingChatImage, images, activeIndex, messages, isTyping]);

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    // ── Render ────────────────────────────────────────────────────────────────
    return (
        <div className="flex h-screen w-full overflow-hidden bg-neutral-950">

            {/* ═══════════════════════════════════════════════════════════════
                LEFT — Image workspace
            ═══════════════════════════════════════════════════════════════ */}
            <div className="flex flex-1 flex-col overflow-hidden border-r border-white/[0.06]">

                {/* Header (aligned with navbar level) */}
                <div className="flex h-[72px] items-center justify-between border-b border-white/[0.06] px-6 shrink-0 pt-3">
                    <div>
                        <h1 className={`text-lg font-bold bg-gradient-to-r ${accentColor} bg-clip-text text-transparent`}>
                            {title}
                        </h1>
                        <p className="text-xs text-neutral-500">{subtitle}</p>
                    </div>
                    {/* Zoom + reset controls */}
                    {images.length > 0 && (
                        <div className="flex items-center gap-2">
                            {/* Reset View button — only shown when view is modified */}
                            <motion.button
                                initial={false}
                                animate={{ opacity: isViewModified ? 1 : 0, scale: isViewModified ? 1 : 0.85 }}
                                transition={{ duration: 0.2 }}
                                onClick={zoomReset}
                                disabled={!isViewModified}
                                title="Reset zoom & position"
                                className="flex items-center gap-1.5 rounded-full bg-white/[0.05] border border-white/[0.08] px-3 py-1.5 text-xs text-white/50 hover:bg-white/10 hover:text-white transition-colors disabled:pointer-events-none"
                            >
                                <RotateCcw className="h-3 w-3" />
                                Reset View
                            </motion.button>

                            {/* Zoom pill */}
                            <div className="flex items-center gap-1 rounded-full bg-white/[0.05] border border-white/[0.08] p-1">
                                <button onClick={zoomOut} className="h-7 w-7 rounded-full flex items-center justify-center text-white/60 hover:bg-white/10 hover:text-white transition-colors text-sm font-bold">−</button>
                                <button onClick={zoomReset} className="px-2 text-xs font-mono text-white/50 hover:text-white transition-colors min-w-[3.5rem] text-center">
                                    {Math.round(zoom * 100)}%
                                </button>
                                <button onClick={zoomIn} className="h-7 w-7 rounded-full flex items-center justify-center text-white/60 hover:bg-white/10 hover:text-white transition-colors text-sm font-bold">+</button>
                            </div>
                        </div>
                    )}
                </div>

                {/* Image area */}
                <div className="relative flex flex-1 flex-col overflow-hidden">
                    {images.length === 0 ? (
                        /* Drop zone */
                        <div
                            ref={dropZoneRef}
                            onDrop={handleDrop}
                            onDragOver={e => { e.preventDefault(); setIsDroppingFile(true); }}
                            onDragLeave={() => setIsDroppingFile(false)}
                            onClick={() => fileInputRef.current?.click()}
                            className={`m-6 flex flex-1 cursor-pointer flex-col items-center justify-center gap-5 rounded-2xl border-2 border-dashed transition-all duration-300
                                ${isDroppingFile
                                    ? "border-sky-400/60 bg-sky-500/5"
                                    : "border-white/10 bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.04]"
                                }`}
                        >
                            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-white/[0.06] border border-white/[0.08]">
                                <svg viewBox="0 0 24 24" fill="none" className="h-8 w-8 text-white/40">
                                    <path d="M4 16l4-4 4 4 4-6 4 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                                    <rect x="3" y="3" width="18" height="18" rx="3" stroke="currentColor" strokeWidth="1.5" />
                                </svg>
                            </div>
                            <div className="text-center">
                                <p className="text-sm font-medium text-white/70">Drop satellite images here</p>
                                <p className="mt-1 text-xs text-white/30">or click to browse · PNG, JPG, GeoTIFF</p>
                            </div>
                            <div className={`rounded-full px-5 py-2 text-xs font-semibold text-white bg-gradient-to-r ${accentColor}`}>
                                Upload Image
                            </div>
                        </div>
                    ) : (
                        /* Animated image stack + zoom + pan */
                        <div className="relative flex flex-1 items-center justify-center overflow-hidden">
                            <div
                                ref={imageWrapRef}
                                style={{
                                    width: "min(560px, 92%)",
                                    height: "clamp(260px, 55vh, 440px)",
                                    transformOrigin: "center center",
                                    willChange: "transform",
                                    cursor: isPanDragging ? "grabbing" : zoom > 1 ? "grab" : "default",
                                }}
                                onWheel={handleWheel}
                                onMouseDown={handlePanMouseDown}
                            >
                                <AnimatedImageStack
                                    images={images}
                                    activeIndex={activeIndex}
                                    onNext={() => setActiveIndex(i => (i + 1) % images.length)}
                                    onPrev={() => setActiveIndex(i => (i - 1 + images.length) % images.length)}
                                />
                            </div>

                            {/* Nav arrows */}
                            {images.length > 1 && (
                                <>
                                    <button
                                        onClick={() => setActiveIndex(i => (i - 1 + images.length) % images.length)}
                                        className="absolute left-4 top-1/2 -translate-y-1/2 h-9 w-9 rounded-full bg-black/50 border border-white/10 flex items-center justify-center text-white/70 hover:bg-white/10 hover:text-white transition-all backdrop-blur-md z-20"
                                    >
                                        ‹
                                    </button>
                                    <button
                                        onClick={() => setActiveIndex(i => (i + 1) % images.length)}
                                        className="absolute right-4 top-1/2 -translate-y-1/2 h-9 w-9 rounded-full bg-black/50 border border-white/10 flex items-center justify-center text-white/70 hover:bg-white/10 hover:text-white transition-all backdrop-blur-md z-20"
                                    >
                                        ›
                                    </button>
                                </>
                            )}
                        </div>
                    )}

                    {/* Thumbnail strip + add more */}
                    {images.length > 0 && (
                        <div className="flex items-center gap-2 overflow-x-auto border-t border-white/[0.06] px-4 py-3 scrollbar-none">
                            {images.map((img, i) => (
                                <div
                                    key={i}
                                    className={`group relative h-14 w-20 shrink-0 cursor-pointer overflow-hidden rounded-lg border-2 transition-all
                                        ${i === activeIndex ? "border-sky-400" : "border-transparent hover:border-white/20"}`}
                                    onClick={() => setActiveIndex(i)}
                                >
                                    <img src={img.src} alt={img.name} className="h-full w-full object-cover" />
                                    <button
                                        onClick={e => { e.stopPropagation(); removeImage(i); }}
                                        className="absolute right-0.5 top-0.5 h-4 w-4 rounded-full bg-black/70 text-white/70 text-[9px] flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-500/80"
                                    >
                                        ✕
                                    </button>
                                </div>
                            ))}
                            {/* Add more button */}
                            <button
                                onClick={() => fileInputRef.current?.click()}
                                className="h-14 w-20 shrink-0 flex flex-col items-center justify-center gap-1 rounded-lg border-2 border-dashed border-white/10 text-white/30 hover:border-white/25 hover:text-white/50 transition-all"
                            >
                                <span className="text-lg leading-none">+</span>
                                <span className="text-[9px]">Add</span>
                            </button>
                        </div>
                    )}
                </div>

                <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    multiple
                    className="hidden"
                    onChange={e => handleFiles(e.target.files)}
                />
            </div>

            {/* ═══════════════════════════════════════════════════════════════
                RIGHT — Chat panel
            ═══════════════════════════════════════════════════════════════ */}
            <div className="flex w-[320px] lg:w-[340px] xl:w-[360px] shrink-0 flex-col bg-neutral-950">

                {/* Chat header (aligned with navbar level) */}
                <div className="flex h-[72px] items-center gap-3 border-b border-white/[0.06] px-5 shrink-0 pt-3">
                    <div className="h-8 w-8 rounded-full overflow-hidden border border-white/15 bg-neutral-900 flex items-center justify-center shadow-lg shrink-0">
                        <img src="/logo.png" alt="GaiaLens" className="h-full w-full object-cover" />
                    </div>
                    <div>
                        <p className="text-sm font-semibold text-white">GaiaLens Assistant</p>
                        <p className="text-[11px] text-neutral-500">{title} · VLM Analysis</p>
                    </div>
                    <div className="ml-auto flex items-center gap-1.5">
                        <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                        <span className="text-[11px] text-neutral-500">Online</span>
                    </div>
                </div>

                {/* Messages list */}
                <div className="flex-1 space-y-4 overflow-y-auto px-4 py-5 scrollbar-none">
                    {messages.map(msg => (
                        <ChatBubble
                            key={msg.id}
                            msg={msg}
                            onOpenLightbox={setLightboxImage}
                        />
                    ))}

                    {/* Typing indicator */}
                    <AnimatePresence>
                        {isTyping && (
                            <motion.div
                                initial={{ opacity: 0, y: 6 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0 }}
                                className="flex items-center gap-2"
                            >
                                <div className="h-7 w-7 rounded-full overflow-hidden border border-white/15 bg-neutral-900 flex items-center justify-center shrink-0">
                                    <img src="/logo.png" alt="GaiaLens" className="h-full w-full object-cover" />
                                </div>
                                <div className="flex gap-1 rounded-2xl rounded-tl-sm bg-neutral-800 px-4 py-3 border border-white/[0.06]">
                                    {[0, 1, 2].map(i => (
                                        <motion.span
                                            key={i}
                                            className="h-1.5 w-1.5 rounded-full bg-white/40"
                                            animate={{ y: [0, -4, 0] }}
                                            transition={{ duration: 0.6, delay: i * 0.15, repeat: Infinity }}
                                        />
                                    ))}
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                    <div ref={chatBottomRef} />
                </div>

                {/* Active workspace image pill */}
                {images.length > 0 && (
                    <div
                        onClick={() => setLightboxImage({ src: images[activeIndex]?.src, alt: images[activeIndex]?.name })}
                        className="mx-4 mb-2 flex items-center gap-2 rounded-xl bg-white/[0.04] border border-white/[0.06] px-3 py-2 cursor-pointer hover:bg-white/[0.07] transition-colors"
                        title="Click to view full size"
                    >
                        <img src={images[activeIndex]?.src} alt="" className="h-8 w-10 rounded object-cover" />
                        <div className="flex-1 min-w-0">
                            <p className="truncate text-[11px] font-medium text-white/70">{images[activeIndex]?.name}</p>
                            <p className="text-[10px] text-white/30">Active image · Click to view</p>
                        </div>
                        <Maximize2 className="h-3 w-3 text-white/30" />
                    </div>
                )}

                {/* Pending chat attachment preview pill */}
                {pendingChatImage && (
                    <div className="mx-4 mb-2 flex items-center justify-between gap-2 rounded-xl bg-white/[0.06] border border-white/15 px-3 py-1.5">
                        <div className="flex items-center gap-2 min-w-0">
                            <img src={pendingChatImage.src} alt="" className="h-7 w-7 rounded object-cover border border-white/20" />
                            <p className="truncate text-[11px] font-medium text-white/80">
                                Attached: {pendingChatImage.name}
                            </p>
                        </div>
                        <button
                            type="button"
                            onClick={() => setPendingChatImage(null)}
                            className="h-5 w-5 rounded-full hover:bg-white/10 flex items-center justify-center text-white/60 hover:text-white text-xs"
                            title="Remove attachment"
                        >
                            ✕
                        </button>
                    </div>
                )}

                {/* Input area */}
                <div className="border-t border-white/[0.06] px-4 py-4">
                    <div className="flex items-end gap-2 rounded-2xl bg-white/[0.05] border border-white/[0.08] px-3.5 py-2.5 focus-within:border-white/20 transition-colors">
                        {/* Attach image button */}
                        <button
                            type="button"
                            onClick={() => chatFileInputRef.current?.click()}
                            title="Attach image to chat"
                            className="h-8 w-8 shrink-0 rounded-full flex items-center justify-center text-white/40 hover:text-white hover:bg-white/10 transition-colors mb-0.5"
                        >
                            <Paperclip className="h-4 w-4" />
                        </button>
                        <input
                            ref={chatFileInputRef}
                            type="file"
                            accept="image/*"
                            className="hidden"
                            onChange={e => {
                                const file = e.target.files?.[0];
                                if (file) {
                                    setPendingChatImage({
                                        src: URL.createObjectURL(file),
                                        name: file.name,
                                        file,
                                    });
                                }
                                e.target.value = "";
                            }}
                        />

                        {/* Textarea */}
                        <textarea
                            ref={textareaRef}
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            onPaste={handleChatPaste}
                            rows={1}
                            placeholder={placeholder || "Ask anything: describe, detect, compare, or measure..."}
                            className="flex-1 resize-none bg-transparent text-sm text-white placeholder:text-white/25 outline-none max-h-32 scrollbar-none leading-relaxed py-1"
                            style={{ height: "auto" }}
                            onInput={e => {
                                const t = e.currentTarget;
                                t.style.height = "auto";
                                t.style.height = `${t.scrollHeight}px`;
                            }}
                        />

                        {/* Send button */}
                        <motion.button
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={sendMessage}
                            disabled={(!input.trim() && !pendingChatImage) || isTyping}
                            className={`h-8 w-8 shrink-0 rounded-full flex items-center justify-center transition-all mb-0.5
                                ${(input.trim() || pendingChatImage) && !isTyping
                                    ? `bg-gradient-to-br ${accentColor} text-white shadow-md`
                                    : "bg-white/[0.05] text-white/20"
                                }`}
                        >
                            <svg viewBox="0 0 20 20" fill="none" className="h-4 w-4 rotate-90">
                                <path d="M10 3L10 17M10 3L4 9M10 3L16 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                            </svg>
                        </motion.button>
                    </div>

                    {/* Mode Selector: Query | Caption | Detect */}
                    <p className="mt-2 text-center text-[10px] text-white/20">Enter to send · Shift+Enter for new line · Paste image supported</p>
                </div>
            </div>

            {/* ═══════════════════════════════════════════════════════════════
                IMAGE LIGHTBOX MODAL — Fullscreen Interactive Viewer
            ═══════════════════════════════════════════════════════════════ */}
            <ImageLightboxModal
                image={lightboxImage}
                onClose={() => setLightboxImage(null)}
            />
        </div>
    );
}
