"use client";

import Link from "next/link";
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "motion/react";

const navLinks = [
  { label: "Explore Earth", href: "/explore-earth" },
  { label: "General Use", href: "/general-use" },
];

export function Navbar() {
  // hoveredLink drives the sliding pill — null means no pill visible
  const [hoveredLink, setHoveredLink] = useState<string | null>(null);
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <>
      <motion.header
        initial={{ y: -80, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
        className="fixed top-0 left-0 right-0 z-50 flex items-center justify-center px-4 pt-4 pointer-events-none"
      >
        <nav
          className={`
            pointer-events-auto
            flex items-center justify-between gap-6 rounded-full px-4 py-2.5
            bg-[#111111]/90 backdrop-blur-md border border-white/[0.06]
            transition-shadow duration-300 w-full max-w-3xl
            ${scrolled ? "shadow-[0_8px_32px_rgba(0,0,0,0.5)]" : "shadow-[0_2px_12px_rgba(0,0,0,0.3)]"}
          `}
        >
          {/* Logo */}
          <Link
            href="/"
            className="flex items-center gap-2.5 shrink-0 group"
            aria-label="GaiaLens Home"
          >
            <div className="relative flex h-8 w-8 items-center justify-center rounded-lg overflow-hidden border border-white/10 bg-white/5 group-hover:border-white/20 transition-all duration-200">
              <img
                src="/logo.png"
                alt="GaiaLens Logo"
                className="h-full w-full object-contain p-0.5"
              />
            </div>
            <span className="text-sm font-semibold tracking-tight text-white">
              GaiaLens
            </span>
          </Link>

          {/* Desktop nav links — pill follows hover */}
          <div
            className="hidden md:flex items-center gap-1"
            onMouseLeave={() => setHoveredLink(null)}
          >
            {navLinks.map((link) => {
              const isHovered = hoveredLink === link.label;
              return (
                <Link
                  key={link.label}
                  href={link.href}
                  onMouseEnter={() => setHoveredLink(link.label)}
                  className="relative px-4 py-1.5 text-sm font-medium rounded-full"
                >
                  {/* Sliding pill — only renders when something is hovered */}
                  {isHovered && (
                    <motion.span
                      layoutId="nav-hover-pill"
                      className="absolute inset-0 rounded-full bg-white/10"
                      transition={{ type: "spring", stiffness: 500, damping: 35 }}
                    />
                  )}
                  <span className="relative z-10 text-white/60 hover:text-white transition-colors duration-150">
                    {link.label}
                  </span>
                </Link>
              );
            })}
          </div>

          {/* CTA + Hamburger */}
          <div className="flex items-center gap-3 shrink-0">
            <motion.div
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              className="hidden md:inline-flex rounded-full shadow-sm"
            >
              <Link
                href="/general-use"
                className="inline-flex items-center gap-1.5 rounded-full bg-white px-5 py-2 text-sm font-semibold text-black hover:bg-white/90 transition-colors duration-200"
              >
                Launch Workspace
                <svg
                  viewBox="0 0 16 16"
                  fill="none"
                  className="h-3.5 w-3.5"
                  aria-hidden="true"
                >
                  <path
                    d="M3 8h10M9 4l4 4-4 4"
                    stroke="currentColor"
                    strokeWidth="1.6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </Link>
            </motion.div>

            {/* Mobile hamburger */}
            <button
              className="md:hidden flex flex-col gap-1.5 p-2 rounded-lg hover:bg-white/10 transition-colors duration-200"
              onClick={() => setMenuOpen((v) => !v)}
              aria-label="Toggle menu"
              aria-expanded={menuOpen}
            >
              <motion.span
                animate={menuOpen ? { rotate: 45, y: 7 } : { rotate: 0, y: 0 }}
                className="block h-0.5 w-5 bg-white origin-center"
              />
              <motion.span
                animate={menuOpen ? { opacity: 0 } : { opacity: 1 }}
                className="block h-0.5 w-5 bg-white"
              />
              <motion.span
                animate={menuOpen ? { rotate: -45, y: -7 } : { rotate: 0, y: 0 }}
                className="block h-0.5 w-5 bg-white origin-center"
              />
            </button>
          </div>
        </nav>
      </motion.header>

      {/* Mobile dropdown */}
      <AnimatePresence>
        {menuOpen && (
          <motion.div
            key="mobile-menu"
            initial={{ opacity: 0, y: -12, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -12, scale: 0.97 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="fixed top-[72px] left-4 right-4 z-40 rounded-2xl bg-[#111111]/95 backdrop-blur-md border border-white/[0.06] p-3 shadow-2xl md:hidden"
          >
            <div className="flex flex-col gap-1">
              {navLinks.map((link) => (
                <Link
                  key={link.label}
                  href={link.href}
                  onClick={() => setMenuOpen(false)}
                  className="flex items-center gap-2 rounded-xl px-4 py-3 text-sm font-medium text-white/70 hover:bg-white/10 hover:text-white transition-colors duration-200"
                >
                  {link.label}
                </Link>
              ))}
              <div className="mt-2 border-t border-white/[0.06] pt-2">
                <Link
                  href="/general-use"
                  onClick={() => setMenuOpen(false)}
                  className="flex w-full items-center justify-center gap-1.5 rounded-xl bg-white px-4 py-2.5 text-sm font-semibold text-black hover:bg-white/90 transition-colors duration-200"
                >
                  Launch Workspace
                </Link>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
