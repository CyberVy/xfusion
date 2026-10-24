import type {
    Txt2ImgParams,
    Img2ImgParams,
    InpaintParams,
    SystemInfo,
    GenerationResponse,
    GenerationProgress
} from "./types"

export class ApiClient {
    private static clean_base_url(url: string): string {
        return url.trim().replace(/\/+$/, "")
    }

    public static async check_health(base_url: string): Promise<boolean> {
        try {
            const url = `${this.clean_base_url(base_url)}/api/v1/system/health`
            const res = await fetch(url, { method: "GET" })
            if (!res.ok) return false
            const data = await res.json()
            return data.status === "healthy"
        } catch {
            return false
        }
    }

    public static async get_system_info(base_url: string): Promise<SystemInfo> {
        const url = `${this.clean_base_url(base_url)}/api/v1/system/info`
        const res = await fetch(url)
        if (!res.ok) throw new Error(`Failed to fetch system info: ${res.statusText}`)
        return res.json()
    }

    public static async load_model(base_url: string, model: string, model_version: string = "xl"): Promise<any> {
        const url = `${this.clean_base_url(base_url)}/api/v1/models/load`
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model, model_version })
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

    public static async generate_txt2img(base_url: string, params: Txt2ImgParams): Promise<GenerationResponse> {
        const url = `${this.clean_base_url(base_url)}/api/v1/generate/txt2img`
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params)
        })
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }))
            throw new Error(err.detail || "Failed to generate image")
        }
        return res.json()
    }

    public static async generate_img2img(base_url: string, params: Img2ImgParams): Promise<GenerationResponse> {
        const url = `${this.clean_base_url(base_url)}/api/v1/generate/img2img`
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params)
        })
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }))
            throw new Error(err.detail || "Failed to generate image")
        }
        return res.json()
    }

    public static async generate_inpaint(base_url: string, params: InpaintParams): Promise<GenerationResponse> {
        const url = `${this.clean_base_url(base_url)}/api/v1/generate/inpaint`
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params)
        })
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }))
            throw new Error(err.detail || "Failed to generate image")
        }
        return res.json()
    }

    public static async get_progress(base_url: string): Promise<GenerationProgress> {
        const url = `${this.clean_base_url(base_url)}/api/v1/generate/progress`
        const res = await fetch(url)
        if (!res.ok) throw new Error("Failed to read generation progress")
        return res.json()
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
}
