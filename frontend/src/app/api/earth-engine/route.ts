import { NextRequest, NextResponse } from "next/server";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";

/**
 * POST /api/earth-engine
 * Body: { north, south, east, west }
 *
 * Authentication — tried in priority order, first one that works wins:
 *
 *  1. GEE_REFRESH_TOKEN env var   ← paste your refresh_token from the credentials file
 *  2. ~/.config/earthengine/credentials  ← auto-populated by: earthengine authenticate
 *  3. Google Application Default Credentials  ← auto-populated by: gcloud auth application-default login
 *  4. GEE_SERVICE_ACCOUNT_KEY env var  ← service-account JSON (one-line)
 *  5. Mock fallback (placeholder images, no auth needed)
 */

// ── Singleton state ───────────────────────────────────────────────────────────
let eeReady = false;
let eeInitPromise: Promise<any> | null = null;

async function getEE() {
    const mod = await import("@google/earthengine");
    return (mod as any).default ?? mod;
}

// Helper to resolve project ID from env or .env.local
function getProjectId(): string {
    if (process.env.GEE_PROJECT_ID) return process.env.GEE_PROJECT_ID;
    try {
        const envLocalPath = path.join(process.cwd(), ".env.local");
        if (fs.existsSync(envLocalPath)) {
            const content = fs.readFileSync(envLocalPath, "utf8");
            const match = content.match(/GEE_PROJECT_ID\s*=\s*([^\r\n#]+)/);
            if (match && match[1].trim()) return match[1].trim();
        }
    } catch { /* ignore */ }
    return "";
}

// ── Helper: set access token and run ee.initialize ───────────────────────────
function initWithToken(ee: any, accessToken: string, expiresIn: number): Promise<void> {
    return new Promise((resolve, reject) => {
        ee.data.setAuthToken(
            "",             // clientId (not needed when we already have a token)
            "Bearer",
            accessToken,
            expiresIn,
            [],             // extraScopes
            () => {
                const project = getProjectId();
                console.log("[GEE] Initializing Earth Engine with project:", project || "(none)");
                ee.initialize(null, null, () => resolve(), reject, null, project || null);
            },
            false,          // opt_suppressDefaultScopes
        );
    });
}

// ── Helper: authenticate via service account private key ─────────────────────
function initWithServiceAccount(ee: any, privateKey: any): Promise<void> {
    return new Promise((resolve, reject) => {
        ee.data.authenticateViaPrivateKey(
            privateKey,
            () => {
                const project = getProjectId() || privateKey.project_id || "";
                ee.initialize(null, null, () => resolve(), reject, null, project || null);
            },
            reject,
        );
    });
}

// ── Well-known EE public OAuth client credentials ────────────────────────────
// Source: earthengine-api Python package (ee/oauth.py)
const EE_CLIENT_ID     = "517222506229-vsmmajv00ul0bs7p89v5m89qs8eb9359.apps.googleusercontent.com";
const EE_CLIENT_SECRET = "RUP0RZ6e0pPhDzsqIJ7KlNd1";

// ── Main init: tries all auth methods in order ────────────────────────────────
async function initEE(): Promise<any> {
    if (eeReady) return getEE();
    if (eeInitPromise) return eeInitPromise;

    eeInitPromise = (async () => {
        try {
        const ee = await getEE();


        // ── Method 1: GEE_REFRESH_TOKEN env var ───────────────────────────────
        // Set this to the refresh_token value from your earthengine credentials file.
        const envRefreshToken = process.env.GEE_REFRESH_TOKEN;
        if (envRefreshToken && !envRefreshToken.includes("REPLACE_ME")) {
            console.log("[GEE] Using GEE_REFRESH_TOKEN env var");
            const clientId     = process.env.GEE_CLIENT_ID     ?? EE_CLIENT_ID;
            const clientSecret = process.env.GEE_CLIENT_SECRET ?? EE_CLIENT_SECRET;
            try {
                const { access_token, expires_in } = await refreshOAuthToken(envRefreshToken, clientId, clientSecret);
                await initWithToken(ee, access_token, expires_in);
                eeReady = true;
                return ee;
            } catch (e) {
                console.warn("[GEE] GEE_REFRESH_TOKEN failed:", e);
            }
        }

        // ── Method 2: earthengine credentials file ────────────────────────────
        // Populated by running:  earthengine authenticate
        // File location: ~/.config/earthengine/credentials
        // New format (post-2023) uses PKCE and omits client_id/client_secret.
        const eeCredPath = path.join(os.homedir(), ".config", "earthengine", "credentials");
        if (fs.existsSync(eeCredPath)) {
            try {
                const creds = JSON.parse(fs.readFileSync(eeCredPath, "utf8"));
                console.log("[GEE] Using earthengine credentials file at", eeCredPath);

                const clientId     = creds.client_id     ?? EE_CLIENT_ID;
                const clientSecret = creds.client_secret ?? EE_CLIENT_SECRET;

                if (creds.refresh_token) {
                    // Use refresh token to obtain fresh access token
                    const { access_token, expires_in } = await refreshOAuthToken(
                        creds.refresh_token,
                        clientId,
                        clientSecret,
                    );
                    await initWithToken(ee, access_token, expires_in);
                } else if (creds.access_token) {
                    // Directly use existing access token (may be short-lived)
                    const expiresIn = creds.expires_in ?? 3600;
                    await initWithToken(ee, creds.access_token, expiresIn);
                } else {
                    throw new Error("No token information in credentials file");
                }
                eeReady = true;
                return ee;
            } catch (e) {
                console.warn("[GEE] earthengine credentials file failed:", e);
            }
        }

        // ── Method 3: Google Application Default Credentials (ADC) ────────────
        // Populated by running:  gcloud auth application-default login
        try {
            console.log("[GEE] Trying Google Application Default Credentials (ADC)");
            const { GoogleAuth } = await import("google-auth-library");
            const auth   = new GoogleAuth({ scopes: ["https://www.googleapis.com/auth/earthengine"] });
            const client = await auth.getClient();
            const res    = await client.getAccessToken();
            const token  = typeof res === "string" ? res : (res as any)?.token ?? (res as any)?.access_token;
            if (token) {
                await initWithToken(ee, token, 3600);
                eeReady = true;
                return ee;
            }
        } catch (e) {
            console.warn("[GEE] ADC failed:", (e as any)?.message ?? e);
        }

        // ── Method 4: Service account key env var ─────────────────────────────
        const keyStr = process.env.GEE_SERVICE_ACCOUNT_KEY;
        if (keyStr && keyStr.startsWith("{") && !keyStr.includes("REPLACE_ME")) {
            try {
                console.log("[GEE] Using GEE_SERVICE_ACCOUNT_KEY");
                const privateKey = JSON.parse(keyStr);
                await initWithServiceAccount(ee, privateKey);
                eeReady = true;
                return ee;
            } catch (e) {
                console.warn("[GEE] Service account key failed:", e);
            }
        }

        throw new Error(
            "No Earth Engine credentials found. " +
            "Run one of:\n" +
            "  • earthengine authenticate   (free personal account)\n" +
            "  • gcloud auth application-default login\n" +
            "Or set GEE_REFRESH_TOKEN in .env.local",
        );
        } catch (err) {
            // Reset so the next request retries rather than re-throwing cached failure
            eeInitPromise = null;
            throw err;
        }
    })();

    return eeInitPromise;
}

// ── Refresh an OAuth2 access token using a refresh token ─────────────────────
async function refreshOAuthToken(
    refreshToken: string,
    clientId: string,
    clientSecret: string,
): Promise<{ access_token: string; expires_in: number }> {
    const params: Record<string, string> = {
        client_id:     clientId,
        refresh_token: refreshToken,
        grant_type:    "refresh_token",
    };
    // Only include client_secret if non-empty (omitting it is correct for pure PKCE public clients)
    if (clientSecret) params.client_secret = clientSecret;

    const res = await fetch("https://oauth2.googleapis.com/token", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams(params),
    });
    const data = await res.json();
    if (!data.access_token) throw new Error(`Token refresh failed: ${JSON.stringify(data)}`);
    return { access_token: data.access_token, expires_in: data.expires_in ?? 3600 };
}

// ── Promisify ee.Image.getThumbURL with timeout ──────────────────────────────
function getThumbUrl(ee: any, image: any, params: any, timeoutMs = 20_000): Promise<string> {
    return Promise.race([
        new Promise<string>((resolve, reject) => {
            image.getThumbURL(params, (url: string, err: any) => {
                if (err) reject(new Error(String(err)));
                else     resolve(url);
            });
        }),
        new Promise<never>((_, reject) =>
            setTimeout(() => reject(new Error("getThumbURL timed out")), timeoutMs)
        ),
    ]);
}

// ── Route handler ─────────────────────────────────────────────────────────────
export async function POST(req: NextRequest) {
    let body: any;
    try {
        body = await req.json();
    } catch {
        return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
    }

    const { north, south, east, west } = body ?? {};
    if ([north, south, east, west].some(v => v == null)) {
        return NextResponse.json({ error: "Missing bbox fields (north, south, east, west)" }, { status: 400 });
    }

    // Quick check: are any credentials configured at all?
    const hasAnyCredentials = (
        (process.env.GEE_REFRESH_TOKEN && !process.env.GEE_REFRESH_TOKEN.includes("REPLACE_ME")) ||
        fs.existsSync(path.join(os.homedir(), ".config", "earthengine", "credentials")) ||
        fs.existsSync(path.join(os.homedir(), ".config", "gcloud", "application_default_credentials.json")) ||
        (process.env.GEE_SERVICE_ACCOUNT_KEY?.startsWith("{") && !process.env.GEE_SERVICE_ACCOUNT_KEY.includes("REPLACE_ME"))
    );

    if (!hasAnyCredentials) {
        console.warn("[GEE] No credentials found — returning mock data");
        return NextResponse.json({ mock: true, images: mockImages({ north, south, east, west }) });
    }

    try {
        const ee = await initEE();

        // Ensure ROI has non-zero width/height by adding a buffer based on bbox size
        const minDelta = 0.001;
        const width = east - west;
        const height = north - south;
        console.log('[GEE] BBox received:', { north, south, east, west });
        console.log('[GEE] width, height:', width, height);
        const delta = Math.max(
            minDelta,
            Math.abs(width) * 0.02,
            Math.abs(height) * 0.02,
        );
        const roi = ee.Geometry.Rectangle([
            west - delta,
            south - delta,
            east + delta,
            north + delta,
        ], null, false); // geodesic false to avoid reprojection edge issues
        const currentYear = new Date().getFullYear();


        const images = await Promise.all(
            Array.from({ length: 4 }, async (_, i) => {
                const year  = currentYear - 3 + i;
                try {
                    // Limit collection size + filter clouds aggressively for speed
                    const col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                        .filterBounds(roi)
                        .filterDate(`${year}-03-01`, `${year}-10-31`)
                        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
                        .select(["B4", "B3", "B2"])
                        .limit(10);

                    // Median composite is faster than .first() on large collections
                    // Create a median composite and apply visualization parameters
                    const image = col.median();
                    const viz = image.visualize({
                        bands: ["B4", "B3", "B2"],
                        min: 0,
                        max: 4000,
                        gamma: 1.4,
                    });
                    const url = await getThumbUrl(ee, viz, {
                        region: roi,
                        dimensions: 256,
                        format: "png",
                    }, 25_000);

                    return { year, date: `${year}-06-01`, url };
                } catch (imgErr) {
                    console.warn(`[GEE] S2 failed for ${year}:`, imgErr);
                    const cx = ((east + west)   / 2).toFixed(5);
                    const cy = ((north + south) / 2).toFixed(5);
                    return { year, date: `${year}-06-01`, mock: true,
                        url: `https://staticmap.openstreetmap.de/staticmap.php?center=${cy},${cx}&zoom=13&size=400x400` };
                }
            }),
        );

        return NextResponse.json({ images });
    } catch (err: any) {
        const msg = String(err?.message ?? err);
        console.error("[/api/earth-engine] Error:", msg);
        // Always return mock data on error so the UI never crashes
        return NextResponse.json({ error: msg, mock: true, images: mockImages({ north, south, east, west }) });
    }
}

// ── Mock fallback ─────────────────────────────────────────────────────────────
function mockImages({ north, south, east, west }: { north: number; south: number; east: number; west: number }) {
    const currentYear = new Date().getFullYear();
    const cx = ((east + west)   / 2).toFixed(5);
    const cy = ((north + south) / 2).toFixed(5);
    return Array.from({ length: 4 }, (_, i) => {
        const year = currentYear - 3 + i;
        return {
            year,
            date:  `${year}-06-01`,
            url:   `https://staticmap.openstreetmap.de/staticmap.php?center=${cy},${cx}&zoom=13&size=400x400`,
            mock:  true,
        };
    });
}
