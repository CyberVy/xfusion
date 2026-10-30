import type {
    Txt2ImgParams,
    Img2ImgParams,
    InpaintParams,
    SystemInfo,
    GenerationResponse,
    GenerationProgress,
    StreamGenerationEvent
} from "./types"

export class ApiClient {
    private static clean_base_url(url: string): string {
        return url.trim().replace(/\/+$/, "")
    }

    public static async check_health(base_url: string): Promise<boolean> {
        try {
            const url = `${this.clean_base_url(base_url)}/api/v1/system/health`
            const res = await fetch(url, { method: "GET", signal: AbortSignal.timeout(10_000) })
            if (!res.ok) return false
            const data = await res.json()
            return data.status === "healthy"
        } catch {
            return false
        }
    }

    public static async get_system_info(base_url: string): Promise<SystemInfo> {
        const url = `${this.clean_base_url(base_url)}/api/v1/system/info`
        const res = await fetch(url, { signal: AbortSignal.timeout(15_000) })
        if (!res.ok) throw new Error(`Failed to fetch system info: ${res.statusText}`)
        return res.json()
    }

    public static async load_model(base_url: string, model: string, model_version: string = "xl"): Promise<any> {
        const url = `${this.clean_base_url(base_url)}/api/v1/models/load`
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model, model_version }),
            signal: AbortSignal.timeout(600_000)
        })
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }))
            throw new Error(err.detail || "Failed to load model")
        }
        return res.json()
    }

    public static async load_lora(
        base_url: string,
        lora_url: string,
        lora_name: string,
        strength: number = 0.4
    ): Promise<any> {
        const url = `${this.clean_base_url(base_url)}/api/v1/lora/load`
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url: lora_url, lora_name, strength })
        })
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }))
            throw new Error(err.detail || "Failed to load LoRA")
        }
        return res.json()
    }

    public static async delete_lora(base_url: string, lora_name: string): Promise<any> {
        const url = `${this.clean_base_url(base_url)}/api/v1/lora/delete`
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ lora_name })
        })
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }))
            throw new Error(err.detail || "Failed to delete LoRA")
        }
        return res.json()
    }

    public static async generate_txt2img(
        base_url: string,
        params: Txt2ImgParams,
        signal?: AbortSignal
    ): Promise<GenerationResponse> {
        const url = `${this.clean_base_url(base_url)}/api/v1/generate/txt2img`
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params),
            signal: signal ?? AbortSignal.timeout(600_000)
        })
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }))
            throw new Error(err.detail || "Failed to generate image")
        }
        return res.json()
    }

    public static async generate_img2img(
        base_url: string,
        params: Img2ImgParams,
        signal?: AbortSignal
    ): Promise<GenerationResponse> {
        const url = `${this.clean_base_url(base_url)}/api/v1/generate/img2img`
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params),
            signal: signal ?? AbortSignal.timeout(600_000)
        })
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }))
            throw new Error(err.detail || "Failed to generate image")
        }
        return res.json()
    }

    public static async generate_inpaint(
        base_url: string,
        params: InpaintParams,
        signal?: AbortSignal
    ): Promise<GenerationResponse> {
        const url = `${this.clean_base_url(base_url)}/api/v1/generate/inpaint`
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params),
            signal: signal ?? AbortSignal.timeout(600_000)
        })
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }))
            throw new Error(err.detail || "Failed to generate image")
        }
        return res.json()
    }

    public static async *generate_txt2img_stream(
        base_url: string,
        params: Txt2ImgParams,
        signal?: AbortSignal
    ): AsyncGenerator<StreamGenerationEvent, void, unknown> {
        yield* this.consume_sse_stream(
            `${this.clean_base_url(base_url)}/api/v1/generate/txt2img/stream`,
            params,
            signal
        )
    }

    public static async *generate_img2img_stream(
        base_url: string,
        params: Img2ImgParams,
        signal?: AbortSignal
    ): AsyncGenerator<StreamGenerationEvent, void, unknown> {
        yield* this.consume_sse_stream(
            `${this.clean_base_url(base_url)}/api/v1/generate/img2img/stream`,
            params,
            signal
        )
    }

    public static async *generate_inpaint_stream(
        base_url: string,
        params: InpaintParams,
        signal?: AbortSignal
    ): AsyncGenerator<StreamGenerationEvent, void, unknown> {
        yield* this.consume_sse_stream(
            `${this.clean_base_url(base_url)}/api/v1/generate/inpaint/stream`,
            params,
            signal
        )
    }

    public static async get_progress(base_url: string, signal?: AbortSignal): Promise<GenerationProgress> {
        const url = `${this.clean_base_url(base_url)}/api/v1/generate/progress`
        const res = await fetch(url, { signal: signal ?? AbortSignal.timeout(5_000) })
        if (!res.ok) throw new Error("Failed to read generation progress")
        return res.json()
    }

    public static async cancel_generation(base_url: string): Promise<boolean> {
        try {
            const url = `${this.clean_base_url(base_url)}/api/v1/generate/cancel`
            const res = await fetch(url, { method: "POST", signal: AbortSignal.timeout(3_000) })
            return res.ok
        } catch {
            return false
        }
    }

    public static set_lora_strength(base_url: string, lora_name: string, strength: number): Promise<any> {
        return this.post_json(base_url, "/api/v1/lora/strength", { lora_name, strength })
    }

    public static set_loras_enabled(base_url: string, enabled: boolean): Promise<any> {
        return this.post_json(base_url, "/api/v1/lora/enabled", { enabled })
    }

    public static load_ip_adapter(
        base_url: string,
        uri: string,
        subfolder: string,
        weight_name: string,
        image_encoder_folder: string,
        strength: number
    ): Promise<any> {
        return this.post_json(base_url, "/api/v1/ip-adapter/load", {
            uri,
            subfolder,
            weight_name,
            image_encoder_folder,
            strength
        })
    }

    public static set_ip_adapter_strength(base_url: string, strength: number): Promise<any> {
        return this.post_json(base_url, "/api/v1/ip-adapter/strength", { strength })
    }

    public static delete_ip_adapter(base_url: string): Promise<any> {
        return this.post_json(base_url, "/api/v1/ip-adapter/delete")
    }

    public static load_controlnet(base_url: string, model: string): Promise<any> {
        return this.post_json(base_url, "/api/v1/controlnet/load", { model: model || null })
    }

    public static offload_controlnet(base_url: string): Promise<any> {
        return this.post_json(base_url, "/api/v1/controlnet/offload")
    }

    public static preview_canny(
        base_url: string,
        image: string,
        low_threshold: number,
        high_threshold: number
    ): Promise<{ image_base64: string }> {
        return this.post_json(base_url, "/api/v1/controlnet/preview", { image, low_threshold, high_threshold })
    }

    private static async post_json(base_url: string, path: string, body?: unknown): Promise<any> {
        const res = await fetch(`${this.clean_base_url(base_url)}${path}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: body === undefined ? undefined : JSON.stringify(body)
        })
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }))
            const detail = err.detail
            throw new Error(typeof detail === "string" ? detail : "Request failed")
        }
        return res.json()
    }

    private static async *consume_sse_stream(
        url: string,
        body: unknown,
        signal?: AbortSignal
    ): AsyncGenerator<StreamGenerationEvent, void, unknown> {
        const res = await fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Accept: "text/event-stream"
            },
            body: JSON.stringify(body),
            signal: signal ?? AbortSignal.timeout(600_000)
        })

        if (res.status === 404 && url.endsWith("/stream")) {
            const fallback_url = url.slice(0, -7)
            const fallback_res = await fetch(fallback_url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
                signal: signal ?? AbortSignal.timeout(600_000)
            })
            if (!fallback_res.ok) {
                const err = await fallback_res.json().catch(() => ({ detail: fallback_res.statusText }))
                throw new Error(err.detail || "Generation request failed")
            }
            const data: GenerationResponse = await fallback_res.json()
            for (let i = 0; i < data.images.length; i++) {
                const img = data.images[i]
                yield {
                    type: "image",
                    data: {
                        image_base64: img.image_base64,
                        seed: img.seed,
                        width: img.width,
                        height: img.height,
                        index: i,
                        elapsed_seconds: data.elapsed_seconds
                    }
                }
            }
            yield {
                type: "done",
                data: {
                    status: data.status,
                    total_images: data.images.length,
                    elapsed_seconds: data.elapsed_seconds
                }
            }
            return
        }

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }))
            throw new Error(err.detail || "Generation request failed")
        }

        if (!res.body) {
            throw new Error("Response body is not readable")
        }

        const reader = res.body.getReader()
        const decoder = new TextDecoder("utf-8")
        let buffer = ""

        try {
            while (true) {
                const { done, value } = await reader.read()
                if (done) break
                buffer += decoder.decode(value, { stream: true })

                let boundary_index = buffer.indexOf("\n\n")
                while (boundary_index !== -1) {
                    const block = buffer.slice(0, boundary_index)
                    buffer = buffer.slice(boundary_index + 2)

                    const event = this.parse_sse_block(block)
                    if (event) {
                        yield event
                    }
                    boundary_index = buffer.indexOf("\n\n")
                }
            }

            if (buffer.trim()) {
                const event = this.parse_sse_block(buffer)
                if (event) {
                    yield event
                }
            }
        } finally {
            reader.releaseLock()
        }
    }

    private static parse_sse_block(block: string): StreamGenerationEvent | null {
        let event_type = "message"
        const data_lines: string[] = []

        const lines = block.split(/\r?\n/)
        for (const line of lines) {
            if (line.startsWith("event:")) {
                event_type = line.slice(6).trim()
            } else if (line.startsWith("data:")) {
                data_lines.push(line.slice(5).trim())
            }
        }

        if (data_lines.length === 0) return null
        const raw_json = data_lines.join("\n")

        try {
            const parsed = JSON.parse(raw_json)
            if (event_type === "image") {
                return { type: "image", data: parsed }
            }
            if (event_type === "done") {
                return { type: "done", data: parsed }
            }
            if (event_type === "progress") {
                return { type: "progress", data: parsed }
            }
            if (event_type === "error") {
                throw new Error(parsed.detail || "Stream error during generation")
            }
        } catch (err: any) {
            if (event_type === "error") throw err
            return null
        }
        return null
    }
}
