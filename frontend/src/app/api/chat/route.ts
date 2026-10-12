import { NextRequest, NextResponse } from "next/server";

// Shared gateway entry point for General Use and Explore Earth.
// Forward multipart fields unchanged so FastAPI owns history and session handling.
export async function POST(req: NextRequest) {
    const backendUrl = process.env.BACKEND_API_URL || "http://localhost:8080";

    try {
        const incomingFormData = await req.formData();
        const outgoingFormData = new FormData();

        // Forward query
        const query = incomingFormData.get("query");
        if (query) {
            outgoingFormData.append("query", query);
        } else {
            return NextResponse.json(
                { error: "Query parameter is required" },
                { status: 400 }
            );
        }

        // Forward session_id if present
        const sessionId = incomingFormData.get("session_id");
        if (sessionId) {
            outgoingFormData.append("session_id", sessionId);
        }

        // Forward history if present
        const history = incomingFormData.get("history");
        if (history) {
            outgoingFormData.append("history", history);
        }

        // Forward all attached images
        const images = incomingFormData.getAll("images");
        for (const image of images) {
            if (image instanceof Blob) {
                outgoingFormData.append("images", image);
            }
        }

        // Send to FastAPI /chat endpoint
        const targetUrl = `${backendUrl}/chat`;
        const res = await fetch(targetUrl, {
            method: "POST",
            body: outgoingFormData,
        });

        if (!res.ok) {
            const errorText = await res.text();
            console.error(`[api/chat] Backend returned ${res.status}:`, errorText);
            return NextResponse.json(
                {
                    error: `Backend error (${res.status})`,
                    details: errorText,
                    response: `⚠️ Backend error (${res.status}): ${errorText.slice(0, 200)}`,
                },
                { status: res.status }
            );
        }

        const data = await res.json();
        return NextResponse.json(data);
    } catch (err: any) {
        console.error("[api/chat] Error forwarding to backend:", err);
        return NextResponse.json(
            {
                error: "Backend service unreachable",
                details: err?.message || String(err),
                response: `⚠️ Could not connect to GaiaLens backend at ${backendUrl}. Please ensure the Python FastAPI service (app.py) is running.`,
            },
            { status: 503 }
        );
    }
}
