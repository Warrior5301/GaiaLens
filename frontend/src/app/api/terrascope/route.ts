import { NextRequest, NextResponse } from "next/server";

const TERRASCOPE_API_URL = (
    process.env.TERRASCOPE_API_URL ||
    "https://rajputvansh4391--terrascope-bf16-test-fastapi-app.modal.run"
)
    .trim()
    .replace(/\/docs(?:#.*)?\/?$/, "")
    .replace(/\/+$/, "");

export async function POST(request: NextRequest) {
    try {
        const formData = await request.formData();
        const prompt = formData.get("prompt");
        const image = formData.get("image");
        const compareImage = formData.get("compare_image");

        if (typeof prompt !== "string" || !prompt.trim()) {
            return NextResponse.json({ error: "A query prompt is required." }, { status: 400 });
        }

        if (!(image instanceof Blob)) {
            return NextResponse.json({ error: "An image is required for TerraScope analysis." }, { status: 400 });
        }

        if (image.size > 12 * 1024 * 1024) {
            return NextResponse.json({ error: "The image must be 12 MiB or smaller." }, { status: 413 });
        }

        if (compareImage instanceof Blob && compareImage.size > 12 * 1024 * 1024) {
            return NextResponse.json({ error: "Each image must be 12 MiB or smaller." }, { status: 413 });
        }

        const payload = {
            image_base64: Buffer.from(await image.arrayBuffer()).toString("base64"),
            prompt: prompt.trim(),
            ...(compareImage instanceof Blob && {
                compare_image_base64: Buffer.from(await compareImage.arrayBuffer()).toString("base64"),
            }),
        };
        const response = await fetch(`${TERRASCOPE_API_URL}/run`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        const responseBody = await response.text();
        let data: Record<string, unknown>;
        try {
            data = JSON.parse(responseBody) as Record<string, unknown>;
        } catch {
            data = { error: responseBody || "TerraScope returned an unreadable response." };
        }

        if (!response.ok) {
            return NextResponse.json(data, { status: response.status });
        }

        return NextResponse.json(data);
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return NextResponse.json(
            { error: `Could not reach the TerraScope service: ${message}` },
            { status: 502 }
        );
    }
}
