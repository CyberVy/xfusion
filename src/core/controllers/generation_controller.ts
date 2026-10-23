import { BaseController } from "@/infra/base_controller"
import { LocalForageMap } from "@/infra/storage/db.client"
import { create_logger } from "@/infra/logger"
import { ApiClient } from "../api/client"
import { settings_controller } from "./settings_controller"
import type {
    GeneratedImageItem,
    SystemInfo,
    Txt2ImgParams,
    Img2ImgParams,
    InpaintParams
} from "../api/types"

const logger = create_logger("GenerationController")

export interface GenerationState {
    is_generating: boolean
    error: string | null
    system_info: SystemInfo | null
    history: GeneratedImageItem[]
    selected_image: GeneratedImageItem | null

    // Default Generation Parameters
    prompt: string
    negative_prompt: string
    guidance_scale: number
    num_inference_steps: number
    clip_skip: number
    width: number
    height: number
    seed: number | null
    scheduler: string
}

export const INITIAL_GENERATION_STATE: GenerationState = {
    is_generating: false,
    error: null,
    system_info: null,
    history: [],
    selected_image: null,

    prompt: "",
    negative_prompt: "bad hands, low resolution, blurry, distorted, disfigured, text, watermark",
    guidance_scale: 7.0,
    num_inference_steps: 25,
    clip_skip: 0,
    width: 1024,
    height: 1024,
    seed: null,
    scheduler: "DPM++ 2M KARRAS"
}

export class GenerationController extends BaseController<GenerationState> {
    private history_store = new LocalForageMap<GeneratedImageItem>("history", "xfusion")

    constructor() {
        super()
        this._state = { ...INITIAL_GENERATION_STATE }
        this.init().catch((err) => {
            logger.error("Failed to initialize generation controller", err)
        })
    }

    private async init(): Promise<void> {
        try {
            const keys = await this.history_store.keys()
            const items: GeneratedImageItem[] = []
            for (const key of keys) {
                const item = await this.history_store.get(key)
                if (item) items.push(item)
            }
            // Sort by creation time descending
            items.sort((a, b) => b.created_at - a.created_at)

            this.update_state({
                history: items,
                selected_image: items[0] || null
            })
            this.dispatchEvent(new CustomEvent("history_change"))
            this.dispatchEvent(new CustomEvent("change"))

            await this.refresh_system_info()
        } catch (e) {
            logger.error("Error in init", e)
        }
    }

    public async refresh_system_info(): Promise<void> {
        const api_url = settings_controller.get_value("api_url", "http://localhost:8000")
        try {
            const info = await ApiClient.get_system_info(api_url)
            this.update_state({ system_info: info })
            this.dispatchEvent(new CustomEvent("system_info_change"))
            this.dispatchEvent(new CustomEvent("change"))
        } catch (err: any) {
            logger.warn("Could not fetch system info", err)
        }
    }

    public set_params(partial: Partial<GenerationState>): void {
        this.update_state(partial)
        this.dispatchEvent(new CustomEvent("params_change"))
        this.dispatchEvent(new CustomEvent("change"))
    }

    public select_image(item: GeneratedImageItem | null): void {
        this.update_state({ selected_image: item })
        this.dispatchEvent(new CustomEvent("selection_change"))
        this.dispatchEvent(new CustomEvent("change"))
    }

    public async delete_image(id: string): Promise<void> {
        await this.history_store.delete(id)
        const updated = this.state.history.filter((x) => x.id !== id)
        const selected = this.state.selected_image?.id === id ? updated[0] || null : this.state.selected_image
        this.update_state({ history: updated, selected_image: selected })
        this.dispatchEvent(new CustomEvent("history_change"))
        this.dispatchEvent(new CustomEvent("change"))
    }

    public async clear_history(): Promise<void> {
        await this.history_store.clear()
        this.update_state({ history: [], selected_image: null })
        this.dispatchEvent(new CustomEvent("history_change"))
        this.dispatchEvent(new CustomEvent("change"))
    }

    public async generate_txt2img(): Promise<void> {
        const api_url = settings_controller.get_value("api_url", "http://localhost:8000")
        const s = this.state

        if (!s.prompt.trim()) {
            this.update_state({ error: "Please enter a prompt" })
            this.dispatchEvent(new CustomEvent("change"))
            return
        }

        this.update_state({ is_generating: true, error: null })
        this.dispatchEvent(new CustomEvent("status_change"))
        this.dispatchEvent(new CustomEvent("change"))

        try {
            const params: Txt2ImgParams = {
                prompt: s.prompt,
                negative_prompt: s.negative_prompt,
                guidance_scale: s.guidance_scale,
                num_inference_steps: s.num_inference_steps,
                clip_skip: s.clip_skip,
                width: s.width,
                height: s.height,
                seed: s.seed,
                scheduler: s.scheduler
            }

            const res = await ApiClient.generate_txt2img(api_url, params)
            const new_items: GeneratedImageItem[] = []

            for (const img of res.images) {
                const item: GeneratedImageItem = {
                    id: `img_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
                    image_base64: img.image_base64,
                    seed: img.seed,
                    prompt: s.prompt,
                    negative_prompt: s.negative_prompt,
                    width: s.width,
                    height: s.height,
                    created_at: Date.now(),
                    type: "txt2img",
                    elapsed_seconds: res.elapsed_seconds
                }
                await this.history_store.set(item.id, item)
                new_items.push(item)
            }

            const updated_history = [...new_items, ...this.state.history]
            this.update_state({
                is_generating: false,
                history: updated_history,
                selected_image: new_items[0] || this.state.selected_image
            })
            this.dispatchEvent(new CustomEvent("history_change"))
            this.dispatchEvent(new CustomEvent("status_change"))
            this.dispatchEvent(new CustomEvent("change"))
        } catch (err: any) {
            this.update_state({ is_generating: false, error: err.message || "Generation failed" })
            this.dispatchEvent(new CustomEvent("status_change"))
            this.dispatchEvent(new CustomEvent("change"))
        }
    }

    public async generate_img2img(image_base64: string, strength: number): Promise<void> {
        const api_url = settings_controller.get_value("api_url", "http://localhost:8000")
        const s = this.state

        this.update_state({ is_generating: true, error: null })
        this.dispatchEvent(new CustomEvent("status_change"))
        this.dispatchEvent(new CustomEvent("change"))

        try {
            const params: Img2ImgParams = {
                image: image_base64,
                prompt: s.prompt,
                negative_prompt: s.negative_prompt,
                strength,
                guidance_scale: s.guidance_scale,
                num_inference_steps: s.num_inference_steps,
                clip_skip: s.clip_skip,
                width: s.width,
                height: s.height,
                seed: s.seed,
                scheduler: s.scheduler
            }

            const res = await ApiClient.generate_img2img(api_url, params)
            const new_items: GeneratedImageItem[] = []

            for (const img of res.images) {
                const item: GeneratedImageItem = {
                    id: `img_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
                    image_base64: img.image_base64,
                    seed: img.seed,
                    prompt: s.prompt,
                    negative_prompt: s.negative_prompt,
                    width: s.width,
                    height: s.height,
                    created_at: Date.now(),
                    type: "img2img",
                    elapsed_seconds: res.elapsed_seconds
                }
                await this.history_store.set(item.id, item)
                new_items.push(item)
            }

            const updated_history = [...new_items, ...this.state.history]
            this.update_state({
                is_generating: false,
                history: updated_history,
                selected_image: new_items[0] || this.state.selected_image
            })
            this.dispatchEvent(new CustomEvent("history_change"))
            this.dispatchEvent(new CustomEvent("status_change"))
            this.dispatchEvent(new CustomEvent("change"))
        } catch (err: any) {
            this.update_state({ is_generating: false, error: err.message || "Generation failed" })
            this.dispatchEvent(new CustomEvent("status_change"))
            this.dispatchEvent(new CustomEvent("change"))
        }
    }

    public async generate_inpaint(image_base64: string, mask_base64: string, strength: number): Promise<void> {
        const api_url = settings_controller.get_value("api_url", "http://localhost:8000")
        const s = this.state

        this.update_state({ is_generating: true, error: null })
        this.dispatchEvent(new CustomEvent("status_change"))
        this.dispatchEvent(new CustomEvent("change"))

        try {
            const params: InpaintParams = {
                image: image_base64,
                mask_image: mask_base64,
                prompt: s.prompt,
                negative_prompt: s.negative_prompt,
                strength,
                guidance_scale: s.guidance_scale,
                num_inference_steps: s.num_inference_steps,
                clip_skip: s.clip_skip,
                width: s.width,
                height: s.height,
                seed: s.seed,
                scheduler: s.scheduler
            }

            const res = await ApiClient.generate_inpaint(api_url, params)
            const new_items: GeneratedImageItem[] = []

            for (const img of res.images) {
                const item: GeneratedImageItem = {
                    id: `img_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
                    image_base64: img.image_base64,
                    seed: img.seed,
                    prompt: s.prompt,
                    negative_prompt: s.negative_prompt,
                    width: s.width,
                    height: s.height,
                    created_at: Date.now(),
                    type: "inpaint",
                    elapsed_seconds: res.elapsed_seconds
                }
                await this.history_store.set(item.id, item)
                new_items.push(item)
            }

            const updated_history = [...new_items, ...this.state.history]
            this.update_state({
                is_generating: false,
                history: updated_history,
                selected_image: new_items[0] || this.state.selected_image
            })
            this.dispatchEvent(new CustomEvent("history_change"))
            this.dispatchEvent(new CustomEvent("status_change"))
            this.dispatchEvent(new CustomEvent("change"))
        } catch (err: any) {
            this.update_state({ is_generating: false, error: err.message || "Generation failed" })
            this.dispatchEvent(new CustomEvent("status_change"))
            this.dispatchEvent(new CustomEvent("change"))
        }
    }
}

export const generation_controller = new GenerationController()
