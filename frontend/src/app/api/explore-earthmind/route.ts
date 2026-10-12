import { NextRequest, NextResponse } from "next/server";

const EARTHMIND_API_URL = (
    process.env.EARTHMIND_API_URL ||
    "https://rajputvansh4391--earthmind-api-fastapi-app.modal.run"
)
    .trim()
    .replace(/\/docs(?:#.*)?\/?$/, "")
    .replace(/\/+$/, "");

// Explore Earth sends four dated views of the same selected region as one comparison.
const IMAGE_COUNT = 4;
const MAX_IMAGE_BYTES = 12 * 1024 * 1024;

export async function POST(request: NextRequest) {
    try {
        const formData = await request.formData();
        const prompt = formData.get("prompt");
        const images = formData.getAll("images");

        if (typeof prompt !== "string" || !prompt.trim()) {
            return NextResponse.json({ error: "A prompt is required." }, { status: 400 });
        }

        if (prompt.length > 8000) {
            return NextResponse.json({ error: "The prompt must be 8,000 characters or fewer." }, { status: 400 });
        }

        if (images.length !== IMAGE_COUNT || images.some(image => !(image instanceof Blob))) {
            return NextResponse.json(
                { error: "Exactly four satellite images are required for Explore Earth analysis." },
                { status: 400 }
            );
        }

        const imageFiles = images as Blob[];
        if (imageFiles.some(image => image.size > MAX_IMAGE_BYTES)) {
            return NextResponse.json({ error: "Each image must be 12 MiB or smaller." }, { status: 413 });
        }

        const payload = {
            query_type: "temporal_analysis",
            prompt: prompt.trim(),
            images_base64: await Promise.all(
                imageFiles.map(async image =>
                    Buffer.from(await image.arrayBuffer()).toString("base64")
                )
            ),
        };

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

        return NextResponse.json(data, { status: response.status });
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return NextResponse.json(
            { error: `Could not reach the EarthMind temporal service: ${message}` },
            { status: 502 }
        );
    }
}
