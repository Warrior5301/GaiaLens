import { spawn } from "node:child_process";
import { delimiter, join, resolve } from "node:path";
import { existsSync } from "node:fs";
import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

const LOCATION_CONTEXT_API_URL = (
    process.env.LOCATION_CONTEXT_API_URL ||
    "https://rajputvansh4391--location-context-fastapi-fastapi-app.modal.run"
)
    .trim()
    .replace(/\/docs(?:#.*)?\/?$/, "")
    .replace(/\/+$/, "");

const MAX_IMAGE_BYTES = 10 * 1024 * 1024;

type LocalAgentInput = {
    query: string;
    location_context: Record<string, unknown>;
    images: string[];
    history: Array<{ role: string; content: string }>;
};

// This route needs a Node host with Python and the sibling backend directory.
// stdin/stdout carry one JSON request/response; Python sends diagnostic logs to stderr.
function runLocalOrchestrator(input: LocalAgentInput): Promise<string> {
    return new Promise((resolveAnswer, reject) => {
        const cwd = process.cwd();
        const repoRoot = existsSync(join(cwd, "backend", "app", "local_query_runner.py"))
            ? cwd
            : resolve(cwd, "..");
        const python = process.env.GAIALENS_PYTHON || (process.platform === "win32" ? "python" : "python3");
        const currentPythonPath = process.env.PYTHONPATH;
        const child = spawn(python, ["-m", "backend.app.local_query_runner"], {
            cwd: repoRoot,
            env: {
                ...process.env,
                PYTHONPATH: [repoRoot, currentPythonPath].filter(Boolean).join(delimiter),
            },
            stdio: ["pipe", "pipe", "pipe"],
        });

        let stdout = "";
        let stderr = "";
        let settled = false;
        const timeout = setTimeout(() => {
            child.kill();
            if (!settled) {
                settled = true;
                reject(new Error("The local orchestrator timed out."));
            }
        }, 15 * 60 * 1000);

        child.stdout.setEncoding("utf8");
        child.stderr.setEncoding("utf8");
        child.stdout.on("data", chunk => {
            stdout += chunk;
            if (stdout.length > 8 * 1024 * 1024) child.kill();
        });
        child.stderr.on("data", chunk => {
            stderr = (stderr + chunk).slice(-12_000);
        });
        child.on("error", error => {
            clearTimeout(timeout);
            if (!settled) {
                settled = true;
                reject(error);
            }
        });
        child.on("close", code => {
            clearTimeout(timeout);
            if (settled) return;
            settled = true;
            try {
                const result = JSON.parse(stdout) as { answer?: string; error?: string };
                if (code !== 0 || !result.answer) {
                    reject(new Error(result.error || stderr || "The local orchestrator returned no answer."));
                    return;
                }
                resolveAnswer(result.answer);
            } catch {
                reject(new Error(stderr || "Could not read the local orchestrator response."));
            }
        });

        child.stdin.on("error", error => {
            if (!settled) {
                clearTimeout(timeout);
                settled = true;
                reject(error);
            }
        });
        child.stdin.end(JSON.stringify(input));
    });
}

export async function POST(request: NextRequest) {
    try {
        const formData = await request.formData();
        const prompt = formData.get("prompt");
        const image = formData.get("image");
        const historyField = formData.get("history");

        if (typeof prompt !== "string" || !prompt.trim()) {
            return NextResponse.json({ error: "A query is required." }, { status: 400 });
        }
        if (!(image instanceof Blob)) {
            return NextResponse.json({ error: "Upload a satellite image to ask a question." }, { status: 400 });
        }
        if (image.size > MAX_IMAGE_BYTES) {
            return NextResponse.json({ error: "The image must be 10 MiB or smaller." }, { status: 413 });
        }

        let history: LocalAgentInput["history"] = [];
        if (typeof historyField === "string" && historyField) {
            try {
                const parsed = JSON.parse(historyField);
                if (Array.isArray(parsed)) {
                    history = parsed
                        .filter(turn => turn && (turn.role === "user" || turn.role === "assistant") && typeof turn.content === "string")
                        .slice(-6);
                }
            } catch {
                return NextResponse.json({ error: "Conversation history must be valid JSON." }, { status: 400 });
            }
        }

        const inputImageBase64 = Buffer.from(await image.arrayBuffer()).toString("base64");
        const locationResponse = await fetch(`${LOCATION_CONTEXT_API_URL}/location-explorer`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image_base64: inputImageBase64 }),
            signal: AbortSignal.timeout(5 * 60 * 1000),
        });
        const locationResult = await locationResponse.json().catch(() => ({})) as Record<string, any>;
        if (!locationResponse.ok) {
            const detail = locationResult.detail || locationResult.error || `HTTP ${locationResponse.status}`;
            return NextResponse.json(
                { error: `Location lookup failed: ${detail}` },
                { status: locationResponse.status >= 400 && locationResponse.status < 500 ? locationResponse.status : 502 }
            );
        }

        const satelliteImage = Array.isArray(locationResult.satellite_images)
            ? locationResult.satellite_images.find((entry: any) => typeof entry?.image_base64 === "string")
            : undefined;
        if (!satelliteImage?.image_base64) {
            return NextResponse.json({ error: "Location lookup did not return a satellite image." }, { status: 502 });
        }

        // Keep image bytes out of the textual prompt; pass them separately to the agent.
        const locationContext = { ...locationResult };
        delete locationContext.satellite_images;
        const satelliteImageMetadata = { ...satelliteImage };
        delete satelliteImageMetadata.image_base64;
        locationContext.satellite_image = satelliteImageMetadata;
        const answer = await runLocalOrchestrator({
            query: prompt.trim(),
            location_context: locationContext,
            images: [inputImageBase64, satelliteImage.image_base64],
            history,
        });

        return NextResponse.json({ answer, location_context: locationContext });
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return NextResponse.json(
            { error: `Query orchestration failed: ${message}` },
            { status: 502 }
        );
    }
}
