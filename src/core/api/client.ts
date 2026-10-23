import type {
    Txt2ImgParams,
    Img2ImgParams,
    InpaintParams,
    SystemInfo
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

    public static async load_model(
        base_url: string,
        model: string,
        model_version: string = "xl"
    ): Promise<any> {
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
        if (!res.ok) throw new Error(await res.text())
        return res.json()
    }

    public static async delete_lora(base_url: string, lora_name: string): Promise<any> {
        const url = `${this.clean_base_url(base_url)}/api/v1/lora/delete`
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ lora_name })
        })
        if (!res.ok) throw new Error(await res.text())
        return res.json()
    }

    public static async set_scheduler(
        base_url: string,
        scheduler: string,
        pipeline_type: string = "txt2img"
    ): Promise<any> {
        const url = `${this.clean_base_url(base_url)}/api/v1/scheduler/set`
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ scheduler, pipeline_type })
        })
        if (!res.ok) throw new Error(await res.text())
        return res.json()
    }

    public static async generate_txt2img(base_url: string, params: Txt2ImgParams): Promise<any> {
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

    public static async generate_img2img(base_url: string, params: Img2ImgParams): Promise<any> {
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

    public static async generate_inpaint(base_url: string, params: InpaintParams): Promise<any> {
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
}
