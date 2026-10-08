"use client";

import { useRef, useState, useCallback, useEffect, KeyboardEvent } from "react";
import { motion, AnimatePresence } from "motion/react";

interface Message {
    id: string;
    role: "user" | "assistant";
    content: string;
    ts: Date;
}

interface FloatingChatWindowProps {
    onSend?: (message: string, history: Message[]) => Promise<string>;
    title?: string;
    selectedBBox?: { north: number; south: number; east: number; west: number } | null;
}

export default function FloatingChatWindow({
    onSend,
    title = "GaiaLens Assistant",
    selectedBBox,
}: FloatingChatWindowProps) {
    const [messages,  setMessages]  = useState<Message[]>([
        { id: "0", role: "assistant", content: "Hello! I'm your GaiaLens Assistant. Ask me anything about the globe, or select a region to analyse it.", ts: new Date() },
    ]);
    const [input,     setInput]     = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [minimized, setMinimized] = useState(false);

    const TOP_OFFSET = 78;
    const BOTTOM_GAP = 74;
    const CHAT_WIDTH = 360;
    const RIGHT_MARGIN = 24;

    // Refs for DOM-direct drag / resize — no React state for perf
    const panelRef  = useRef<HTMLDivElement>(null);
    const posRef    = useRef({ x: 0, y: TOP_OFFSET });
    const sizeRef   = useRef({ w: CHAT_WIDTH, h: 600 });
    const dragRef   = useRef({ active: false, startX: 0, startY: 0, ox: 0, oy: 0 });
    const resizeRef = useRef({ active: false, startX: 0, startY: 0, ow: 0, oh: 0 });

    const applyPosSize = useCallback(() => {
        const el = panelRef.current;
        if (!el) return;
        el.style.left  = `${posRef.current.x}px`;
        el.style.top   = `${posRef.current.y}px`;
        el.style.width = `${sizeRef.current.w}px`;
        if (!minimized) el.style.height = `${sizeRef.current.h}px`;
    }, [minimized]);

    // Set initial position & vertical height to span from below navbar to above bottom bar
    useEffect(() => {
        const initialH = Math.max(380, window.innerHeight - TOP_OFFSET - BOTTOM_GAP);
        const initialX = Math.max(10, window.innerWidth - CHAT_WIDTH - RIGHT_MARGIN);
        posRef.current = { x: initialX, y: TOP_OFFSET };
        sizeRef.current = { w: CHAT_WIDTH, h: initialH };
        applyPosSize();

        const onResize = () => {
            if (!dragRef.current.active && !resizeRef.current.active) {
                const newH = Math.max(380, window.innerHeight - TOP_OFFSET - BOTTOM_GAP);
                const newX = Math.max(10, window.innerWidth - CHAT_WIDTH - RIGHT_MARGIN);
                posRef.current.x = newX;
                sizeRef.current.h = newH;
                applyPosSize();
            }
        };
        window.addEventListener("resize", onResize);
        return () => window.removeEventListener("resize", onResize);
    }, [applyPosSize]);

    // Drag listeners
    const onDragStart = (e: React.MouseEvent) => {
        e.preventDefault();
        dragRef.current = { active: true, startX: e.clientX, startY: e.clientY, ox: posRef.current.x, oy: posRef.current.y };
    };
    useEffect(() => {
        const onMove = (e: MouseEvent) => {
            if (!dragRef.current.active) return;
            posRef.current = { x: dragRef.current.ox + (e.clientX - dragRef.current.startX), y: dragRef.current.oy + (e.clientY - dragRef.current.startY) };
            applyPosSize();
        };
        const onUp = () => { dragRef.current.active = false; };
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup",   onUp);
        return () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
    }, [applyPosSize]);

    // Resize listeners
    const onResizeStart = (e: React.MouseEvent) => {
        e.preventDefault(); e.stopPropagation();
        resizeRef.current = { active: true, startX: e.clientX, startY: e.clientY, ow: sizeRef.current.w, oh: sizeRef.current.h };
    };
    useEffect(() => {
        const onMove = (e: MouseEvent) => {
            if (!resizeRef.current.active) return;
            sizeRef.current = {
                w: Math.max(280, resizeRef.current.ow + (e.clientX - resizeRef.current.startX)),
                h: Math.max(300, resizeRef.current.oh + (e.clientY - resizeRef.current.startY)),
            };
            applyPosSize();
        };
        const onUp = () => { resizeRef.current.active = false; };
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup",   onUp);
        return () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
    }, [applyPosSize]);

    // Auto-scroll
    const listRef = useRef<HTMLDivElement>(null);
    useEffect(() => { listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" }); }, [messages]);

    // BBox notice
    useEffect(() => {
        if (!selectedBBox) return;
        const { north, south, east, west } = selectedBBox;
        setMessages(prev => [...prev, {
            id: Date.now() + "bbox",
            role: "assistant",
            content: `📍 Region selected: [${south.toFixed(4)}°N, ${west.toFixed(4)}°E] → [${north.toFixed(4)}°N, ${east.toFixed(4)}°E]\nEarth Engine imagery is being fetched. Ask me to analyse this area!`,
            ts: new Date(),
        }]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [selectedBBox]);

    const handleSend = useCallback(async () => {
        const text = input.trim();
        if (!text || isLoading) return;
        setInput("");
        const userMsg: Message = { id: Date.now().toString(), role: "user", content: text, ts: new Date() };
        setMessages(prev => [...prev, userMsg]);
        setIsLoading(true);
        try {
            let reply: string;
            if (onSend) {
                reply = await onSend(text, [...messages, userMsg]);
            } else {
                await new Promise(r => setTimeout(r, 900));
                reply = selectedBBox
                    ? `Analysing your selected region. Connect a Python VLM backend at /api/chat to get real analysis.`
                    : `You asked: "${text}". Wire up your Python backend to /api/chat for AI-powered answers.`;
            }
            setMessages(prev => [...prev, { id: Date.now() + "r", role: "assistant", content: reply, ts: new Date() }]);
        } catch {
            setMessages(prev => [...prev, { id: Date.now() + "e", role: "assistant", content: "Backend error — please try again.", ts: new Date() }]);
        } finally {
            setIsLoading(false);
        }
    }, [input, isLoading, messages, onSend, selectedBBox]);

    const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
    };

    return (
        <div
            ref={panelRef}
            className="fixed z-[200] flex flex-col rounded-2xl overflow-hidden shadow-2xl border border-white/[0.12] bg-neutral-950/90 backdrop-blur-2xl pointer-events-auto"
            style={{ left: 0, top: 78, width: 360, height: minimized ? "auto" : 600 }}
        >
            {/* Title bar — drag handle */}
            <div
                onMouseDown={onDragStart}
                className="flex shrink-0 items-center gap-2 border-b border-white/[0.08] px-4 py-3 cursor-grab active:cursor-grabbing select-none bg-white/[0.03]"
            >
                <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse shrink-0" />
                <span className="flex-1 text-sm font-semibold text-white truncate">{title}</span>
                <button
                    onMouseDown={e => e.stopPropagation()}
                    onClick={() => setMinimized(v => !v)}
                    className="h-6 w-6 rounded-md flex items-center justify-center text-white/40 hover:text-white hover:bg-white/10 transition-colors text-xs"
                >
                    {minimized ? "▲" : "▼"}
                </button>
            </div>

            <AnimatePresence initial={false}>
                {!minimized && (
                    <motion.div
                        key="body"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.15 }}
                        className="flex flex-1 flex-col overflow-hidden min-h-0"
                    >
                        {/* Messages list */}
                        <div ref={listRef} className="flex-1 overflow-y-auto p-3 space-y-3 min-h-0">
                            {messages.map(msg => (
                                <div key={msg.id} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                                    <div className={`max-w-[85%] rounded-xl px-3 py-2 text-xs leading-relaxed whitespace-pre-wrap ${
                                        msg.role === "user"
                                            ? "bg-sky-500/20 text-sky-100 border border-sky-500/20"
                                            : "bg-white/[0.06] text-white/80 border border-white/[0.07]"
                                    }`}>
                                        {msg.content}
                                    </div>
                                </div>
                            ))}
                            {isLoading && (
                                <div className="flex justify-start">
                                    <div className="bg-white/[0.06] border border-white/[0.07] rounded-xl px-4 py-2.5 flex gap-1 items-center">
                                        {[0, 1, 2].map(i => (
                                            <span key={i} className="h-1.5 w-1.5 rounded-full bg-white/40 animate-bounce" style={{ animationDelay: `${i * 0.15}s` }} />
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Input area */}
                        <div className="shrink-0 border-t border-white/[0.06] p-3 flex gap-2 items-end">
                            <textarea
                                value={input}
                                onChange={e => setInput(e.target.value)}
                                onKeyDown={onKeyDown}
                                placeholder="Ask about this region…"
                                rows={1}
                                className="flex-1 resize-none rounded-xl bg-white/[0.06] border border-white/[0.1] text-xs text-white placeholder:text-white/30 px-3 py-2 outline-none focus:border-sky-500/40 transition-colors max-h-24 overflow-y-auto"
                            />
                            <button
                                onClick={handleSend}
                                disabled={isLoading || !input.trim()}
                                className="h-8 w-8 rounded-xl bg-sky-500 hover:bg-sky-400 flex items-center justify-center shrink-0 disabled:opacity-30 transition-colors"
                            >
                                <svg viewBox="0 0 16 16" fill="none" className="h-3.5 w-3.5">
                                    <path d="M14 8L2 2l3 6-3 6 12-6Z" fill="white"/>
                                </svg>
                            </button>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Resize handle */}
            {!minimized && (
                <div
                    onMouseDown={onResizeStart}
                    className="absolute bottom-1 right-1 h-4 w-4 cursor-se-resize opacity-25 hover:opacity-60 transition-opacity"
                >
                    <svg viewBox="0 0 10 10" fill="none" className="h-full w-full">
                        <path d="M9 1L1 9M9 5L5 9" stroke="white" strokeWidth="1.2" strokeLinecap="round"/>
                    </svg>
                </div>
            )}
        </div>
    );
}
