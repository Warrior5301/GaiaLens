export interface ChatTurn {
    role: "user" | "assistant";
    content: string;
}

export interface ChatImage {
    url: string;
    name?: string;
}

export interface ChatReply {
    content: string;
    images: ChatImage[];
}

// Both experiences use the same gateway contract. The orchestrator, rather than
// a browser task selector, decides which models and tools answer the request.
export async function sendChat(
    query: string,
    history: ChatTurn[],
    images: File[] = [],
): Promise<ChatReply> {
    const form = new FormData();
    form.append("query", query);
    form.append("history", JSON.stringify(history.slice(-20).map(({ role, content }) => ({ role, content }))));
    for (const image of images) form.append("images", image);

    const response = await fetch("/api/chat", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) {
        throw new Error(typeof data.detail === "string" ? data.detail : data.error || "The assistant could not complete this request.");
    }
    if (typeof data.response !== "string") throw new Error("The assistant returned an invalid response.");
    return {
        content: data.response,
        images: Array.isArray(data.images)
            ? data.images.filter((image: ChatImage) => typeof image?.url === "string" && image.url.startsWith("/files/"))
            : [],
    };
}
