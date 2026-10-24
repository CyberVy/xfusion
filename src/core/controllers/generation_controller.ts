import { BaseController } from "@/infra/base_controller"
import { LocalForageMap } from "@/infra/storage/db.client"
import { create_logger } from "@/infra/logger"
import { ApiClient } from "../api/client"
import { settings_controller } from "./settings_controller"
import { model_controller } from "./model_controller"
import type { GeneratedImageItem, GenerationResponse } from "../api/types"

const logger = create_logger("GenerationController")

export interface GenerationParams {
    prompt: string
    negative_prompt: string
    guidance_scale: number
    num_inference_steps: number
    clip_skip: number
    width: number
    height: number
    seed: number | null
    scheduler: string
    img2img_strength: number
    inpaint_strength: number
}

export interface GenerationState extends GenerationParams {
    is_generating: boolean
    error: string | null
    history: GeneratedImageItem[]
    selected_image: GeneratedImageItem | null
}

export const INITIAL_GENERATION_STATE: GenerationState = {
    is_generating: false,
    error: null,
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
    scheduler: "DPM++ 2M KARRAS",
    img2img_strength: 0.75,
    inpaint_strength: 0.8
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
        const keys = await this.history_store.keys()
        const items: GeneratedImageItem[] = []
        for (const key of keys) {
            const item = await this.history_store.get(key)
            if (item) items.push(item)
        }
        items.sort((a, b) => b.created_at - a.created_at)

        this.update_state({
            history: items,
            selected_image: items[0] || null
        })
        this.dispatchEvent(new CustomEvent("history_change"))
        this.dispatchEvent(new CustomEvent("change"))
    }

    public set_params(partial: Partial<GenerationParams>): void {
        this.update_state(partial)
        this.dispatchEvent(new CustomEvent("params_change"))
        this.dispatchEvent(new CustomEvent("change"))
    }

    public clear_error(): void {
        if (this.state.error === null) return
        this.update_state({ error: null })
        this.dispatchEvent(new CustomEvent("change"))
    }

    public reuse_settings(item: GeneratedImageItem): void {
        this.set_params({
            prompt: item.prompt,
            negative_prompt: item.negative_prompt ?? this.state.negative_prompt,
            seed: item.seed,
            ...(item.type === "txt2img" ? { width: item.width, height: item.height } : {})
        })
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

    public generate_txt2img(): Promise<void> {
        if (!this.state.prompt.trim()) {
            this.update_state({ error: "Please enter a prompt" })
            this.dispatchEvent(new CustomEvent("change"))
            return Promise.resolve()
        }
        const { width, height } = this.state
        return this.run_generation("txt2img", (api_url, common) =>
            ApiClient.generate_txt2img(api_url, { ...common, width, height })
        )
    }

    public generate_img2img(image_base64: string): Promise<void> {
        const strength = this.state.img2img_strength
        return this.run_generation("img2img", (api_url, common) =>
            ApiClient.generate_img2img(api_url, { ...common, image: image_base64, strength })
        )
    }

    public generate_inpaint(image_base64: string, mask_base64: string): Promise<void> {
        const strength = this.state.inpaint_strength
        return this.run_generation("inpaint", (api_url, common) =>
            ApiClient.generate_inpaint(api_url, {
                ...common,
                image: image_base64,
                mask_image: mask_base64,
                strength
            })
        )
    }

    private common_request_params() {
        const s = this.state
        const schedulers = model_controller.state.system_info?.schedulers ?? []
        return {
            prompt: s.prompt,
            negative_prompt: s.negative_prompt,
            guidance_scale: s.guidance_scale,
            num_inference_steps: s.num_inference_steps,
            clip_skip: s.clip_skip,
            seed: s.seed,
            scheduler: schedulers.includes(s.scheduler) ? s.scheduler : null
        }
    }

    private async run_generation(
        type: GeneratedImageItem["type"],
        request: (
            api_url: string,
            common: ReturnType<GenerationController["common_request_params"]>
        ) => Promise<GenerationResponse>
    ): Promise<void> {
        if (this.state.is_generating) return
        const api_url = settings_controller.get_value("api_url", "http://localhost:8000")
        const { prompt, negative_prompt } = this.state

        this.update_state({ is_generating: true, error: null })
        this.dispatchEvent(new CustomEvent("status_change"))
        this.dispatchEvent(new CustomEvent("change"))

        try {
            const res = await request(api_url, this.common_request_params())
            const new_items: GeneratedImageItem[] = []

            for (const img of res.images) {
                const item: GeneratedImageItem = {
                    id: `img_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
                    image_base64: img.image_base64,
                    seed: img.seed,
                    prompt,
                    negative_prompt,
                    width: img.width,
                    height: img.height,
                    created_at: Date.now(),
                    type,
                    elapsed_seconds: res.elapsed_seconds
                }
                await this.history_store.set(item.id, item)
                new_items.push(item)
            }

            this.update_state({
                is_generating: false,
                history: [...new_items, ...this.state.history],
                selected_image: new_items[0] || this.state.selected_image
            })
            this.dispatchEvent(new CustomEvent("history_change"))
        } catch (err: any) {
            this.update_state({ is_generating: false, error: err.message || "Generation failed" })
        }
        this.dispatchEvent(new CustomEvent("status_change"))
        this.dispatchEvent(new CustomEvent("change"))
    }
}

export const generation_controller = new GenerationController()
