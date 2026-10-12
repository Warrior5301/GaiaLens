import { NextRequest, NextResponse } from "next/server";

const EARTHMIND_API_URL = (
    process.env.EARTHMIND_API_URL ||
    "https://rajputvansh4391--earthmind-api-fastapi-app.modal.run"
)
    .trim()
    .replace(/\/docs(?:#.*)?\/?$/, "")
    .replace(/\/+$/, "");

// Adapt browser multipart uploads to EarthMind's JSON/base64 API.
// Legacy direct-model endpoint; interactive chats use /api/chat.
export async function POST(request: NextRequest) {
    try {
        const formData = await request.formData();
        const prompt = formData.get("prompt");
        const image = formData.get("image");

        if (typeof prompt !== "string" || !prompt.trim()) {
            return NextResponse.json({ error: "A prompt is required." }, { status: 400 });
        }

        if (image !== null && !(image instanceof Blob)) {
            return NextResponse.json({ error: "The image upload is invalid." }, { status: 400 });
        }

        const payload: { query_type: "normal"; prompt: string; image_base64?: string } = {
            query_type: "normal",
            prompt: prompt.trim(),
        };

        if (image instanceof Blob) {
            if (image.size > 12 * 1024 * 1024) {
                return NextResponse.json({ error: "The image must be 12 MiB or smaller." }, { status: 413 });
            }
            payload.image_base64 = Buffer.from(await image.arrayBuffer()).toString("base64");
        }

        const response = await fetch(`${EARTHMIND_API_URL}/run`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        const responseBody = await response.text();
        let data: Record<string, unknown>;
        try {
            data = JSON.parse(responseBody) as Record<string, unknown>;
        } catch {
            data = { error: responseBody || "EarthMind returned an unreadable response." };
        }

        if (!response.ok) {
            return NextResponse.json(data, { status: response.status });
        }

        return NextResponse.json(data);
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return NextResponse.json(
            { error: `Could not reach the EarthMind caption service: ${message}` },
            { status: 502 }
        );
    }
}
