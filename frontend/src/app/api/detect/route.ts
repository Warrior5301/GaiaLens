import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
    const rawDetectUrl = process.env.DETECT_API_URL || "https://rajputvansh4391--o2-rtdetr-api-fastapi-app.modal.run";
    const detectApiUrl = rawDetectUrl
        .trim()
        .replace(/\/docs(#.*)?\/?$/, "")
        .replace(/\/+$/, "");

    try {
        let base64Image = "";
        let scoreThreshold = 0.3;
        let prompt = "";

        const contentType = req.headers.get("content-type") || "";

        if (contentType.includes("multipart/form-data")) {
            const formData = await req.formData();
            const imageFile = formData.get("image");
            const thresholdField = formData.get("score_threshold");
            const promptField = formData.get("prompt");

            if (thresholdField) {
                const parsed = parseFloat(thresholdField.toString());
                if (!isNaN(parsed)) scoreThreshold = parsed;
            }
            if (promptField) {
                prompt = promptField.toString();
            }

            if (imageFile instanceof Blob) {
                const arrayBuffer = await imageFile.arrayBuffer();
                const buffer = Buffer.from(arrayBuffer);
                base64Image = buffer.toString("base64");
            } else if (typeof imageFile === "string") {
                base64Image = imageFile;
            }
        } else {
            const json = await req.json();
            base64Image = json.image || "";
            if (json.score_threshold !== undefined) {
                scoreThreshold = Number(json.score_threshold);
            }
            if (json.prompt) {
                prompt = json.prompt;
            }
        }

        if (!base64Image) {
            return NextResponse.json(
                { error: "Image data is required for detection" },
                { status: 400 }
            );
        }

        // Clean any data URI prefix if present
        if (base64Image.includes(",")) {
            base64Image = base64Image.split(",")[1];
        }

        // Target detection endpoint: POST /run on O2-RTDETR service
        const targetUrl = detectApiUrl.endsWith("/run") ? detectApiUrl : `${detectApiUrl}/run`;

        const detectPayload = {
            image: base64Image,
            score_threshold: scoreThreshold,
        };

        const res = await fetch(targetUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(detectPayload),
        });

        if (!res.ok) {
            const errorText = await res.text();
            console.error(`[api/detect] Detection service returned ${res.status}:`, errorText);
            return NextResponse.json(
                {
                    error: `Detection service error (${res.status})`,
                    details: errorText,
                    response: `⚠️ Detection service returned an error (${res.status}): ${errorText.slice(0, 200)}`,
                },
                { status: res.status }
            );
        }

        const data = await res.json();

        // data format: { image: base64_annotated_png, count: number, detections: [...] }
        const count = data.count ?? (Array.isArray(data.detections) ? data.detections.length : 0);
        const detections = data.detections || [];

        // Count occurrences per label
        const labelCounts: Record<string, number> = {};
        for (const det of detections) {
            const label = det.label || "object";
            labelCounts[label] = (labelCounts[label] || 0) + 1;
        }

        const breakdown = Object.entries(labelCounts)
            .map(([lbl, num]) => `${num} ${lbl}${num > 1 ? "s" : ""}`)
            .join(", ");

        const summary = count > 0
            ? `Object detection complete. Identified ${count} objects${breakdown ? ` (${breakdown})` : ""}.`
            : "Object detection complete. No objects detected at the current confidence threshold.";

        const annotatedImageUrl = data.image
            ? (data.image.startsWith("data:") ? data.image : `data:image/png;base64,${data.image}`)
            : null;

        return NextResponse.json({
            success: true,
            count,
            detections,
            image: annotatedImageUrl,
            summary,
            response: summary,
        });
    } catch (err: any) {
        console.error("[api/detect] Error forwarding to detection service:", err);
        return NextResponse.json(
            {
                error: "Detection service unreachable",
                details: err?.message || String(err),
                response: `⚠️ Detection service at ${detectApiUrl} is currently unreachable. Please make sure the O2-RTDETR detection endpoint is running.`,
            },
            { status: 503 }
        );
    }
}
