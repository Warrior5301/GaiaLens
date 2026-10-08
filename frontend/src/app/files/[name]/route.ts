import { NextRequest, NextResponse } from "next/server";

export async function GET(
    _req: NextRequest,
    context: { params: Promise<{ name: string }> }
) {
    const { name } = await context.params;
    const backendUrl = process.env.BACKEND_API_URL || "http://localhost:8000";

    try {
        const res = await fetch(`${backendUrl}/files/${encodeURIComponent(name)}`);

        if (!res.ok) {
            return new NextResponse("File not found or backend error", {
                status: res.status,
            });
        }

        const contentType = res.headers.get("content-type") || "image/png";
        const arrayBuffer = await res.arrayBuffer();

        return new NextResponse(arrayBuffer, {
            status: 200,
            headers: {
                "Content-Type": contentType,
                "Cache-Control": "public, max-age=3600",
            },
        });
    } catch (err: any) {
        console.error(`[files/${name}] Failed to fetch from backend:`, err);
        return new NextResponse("Backend service unreachable", { status: 502 });
    }
}
