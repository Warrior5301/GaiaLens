import { NextRequest, NextResponse } from "next/server";

const YOLOVR_API_URL = (
    process.env.YOLOVR_API_URL ||
    "https://rajputvansh4391--yolovr-inference-web-app.modal.run"
)
    .trim()
    .replace(/\/docs(?:#.*)?\/?$/, "")
    .replace(/\/+$/, "");

export async function POST(request: NextRequest) {
    try {
        let imageBase64 = "";
        let prompt = "objects";

        const contentType = request.headers.get("content-type") || "";
        if (contentType.includes("multipart/form-data")) {
            const formData = await request.formData();
            const image = formData.get("image");
            const promptField = formData.get("prompt");

            if (promptField) prompt = promptField.toString().trim() || prompt;

            if (image instanceof Blob) {
                if (image.size > 12 * 1024 * 1024) {
                    return NextResponse.json(
                        { error: "The image must be 12 MiB or smaller." },
                        { status: 413 }
                    );
                }
                imageBase64 = Buffer.from(await image.arrayBuffer()).toString("base64");
            } else if (typeof image === "string") {
                imageBase64 = image;
            }
        } else {
            const body = await request.json();
            imageBase64 = body.image_base64 || body.image || "";
            prompt = body.prompt?.trim() || prompt;
        }

        if (!imageBase64) {
            return NextResponse.json(
                { error: "Image data is required for detection." },
                { status: 400 }
            );
        }

        // Accept a base64 data URL from JSON callers as well as raw base64.
        if (imageBase64.startsWith("data:")) {
            const separator = imageBase64.indexOf(",");
            if (separator < 0) {
                return NextResponse.json(
                    { error: "Image data URL is invalid." },
                    { status: 400 }
                );
            }
            imageBase64 = imageBase64.slice(separator + 1);
        }

        const targetUrl = YOLOVR_API_URL.endsWith("/predict")
            ? YOLOVR_API_URL
            : `${YOLOVR_API_URL}/predict`;
        const upstreamResponse = await fetch(targetUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image_base64: imageBase64, prompt }),
        });

        const responseText = await upstreamResponse.text();
        let data: Record<string, unknown>;
        try {
            data = JSON.parse(responseText) as Record<string, unknown>;
        } catch {
            data = { error: responseText || "YolovR returned an unreadable response." };
        }

        if (!upstreamResponse.ok) {
            const detail = data.detail || data.error || `YolovR returned ${upstreamResponse.status}.`;
            return NextResponse.json(
                { error: `Detection service error (${upstreamResponse.status})`, details: detail, response: String(detail) },
                { status: upstreamResponse.status }
            );
        }

        const resultImage = data.image_base64 || data.image;
        if (typeof resultImage !== "string" || !resultImage) {
            return NextResponse.json(
                { error: "YolovR did not return an annotated image." },
                { status: 502 }
            );
        }

        const imageUrl = resultImage.startsWith("data:")
            ? resultImage
            : `data:image/jpeg;base64,${resultImage}`;

        return NextResponse.json({
            success: true,
            image: imageUrl,
            prompt,
            summary: `Detection complete for “${prompt}”. The annotated image is attached.`,
        });
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return NextResponse.json(
            {
                error: "Detection service unreachable",
                details: message,
                response: `Could not connect to the YolovR detection service: ${message}`,
            },
            { status: 502 }
        );
    }
}
