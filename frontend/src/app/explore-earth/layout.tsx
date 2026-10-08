// Full‑width, full‑height layout container

export default function ExploreEarthLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <div className="relative w-full h-screen overflow-hidden bg-gray-900">
            {/* Optional global navigation */}
            {/* <Navbar /> */}
            {children}
        </div>
    );
}
