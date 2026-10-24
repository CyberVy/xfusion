import { BaseController } from "@/infra/base_controller"
import { create_logger } from "@/infra/logger"
import { ApiClient } from "../api/client"
import { settings_controller } from "./settings_controller"
import type { SystemInfo } from "../api/types"

const logger = create_logger("ModelController")

export interface ModelState {
    system_info: SystemInfo | null
    is_loading_model: boolean
    is_loading_lora: boolean
    is_loading_ip_adapter: boolean
    is_loading_controlnet: boolean
    model_error: string | null
    lora_error: string | null
    ip_adapter_error: string | null
    controlnet_error: string | null
}

export class ModelController extends BaseController<ModelState> {
    constructor() {
        super()
        this._state = {
            system_info: null,
            is_loading_model: false,
            is_loading_lora: false,
            is_loading_ip_adapter: false,
            is_loading_controlnet: false,
            model_error: null,
            lora_error: null,
            ip_adapter_error: null,
            controlnet_error: null
        }
        settings_controller.addEventListener("set:connection", (event) => {
            if ((event as CustomEvent<{ is_connected: boolean }>).detail.is_connected) {
                this.refresh_system_info()
            } else {
                this.commit({ system_info: null })
            }
        })
    }

    private get api_url(): string {
        return settings_controller.get_value("api_url", "http://localhost:8000")
    }

    private commit(partial: Partial<ModelState>): void {
        this.update_state(partial)
        this.dispatchEvent(new CustomEvent("change"))
    }

    public async refresh_system_info(): Promise<void> {
        try {
            this.commit({ system_info: await ApiClient.get_system_info(this.api_url) })
        } catch (err) {
            logger.warn("Could not fetch system info", err)
            this.commit({ system_info: null })
        }
    }

    public async load_model(model: string, model_version: string): Promise<void> {
        this.commit({ is_loading_model: true, model_error: null })
        try {
            await ApiClient.load_model(this.api_url, model, model_version)
            await this.refresh_system_info()
            this.commit({ is_loading_model: false })
        } catch (err: any) {
            this.commit({ is_loading_model: false, model_error: err.message || "Failed to load model" })
        }
    }

    public async load_lora(lora_url: string, lora_name: string, strength: number): Promise<boolean> {
        this.commit({ is_loading_lora: true, lora_error: null })
        try {
            await ApiClient.load_lora(this.api_url, lora_url, lora_name, strength)
            await this.refresh_system_info()
            this.commit({ is_loading_lora: false })
            return true
        } catch (err: any) {
            this.commit({ is_loading_lora: false, lora_error: err.message || "Failed to load LoRA" })
            return false
        }
    }

    public async set_lora_strength(lora_name: string, strength: number): Promise<void> {
        this.commit({ lora_error: null })
        try {
            await ApiClient.set_lora_strength(this.api_url, lora_name, strength)
            await this.refresh_system_info()
        } catch (err: any) {
            this.commit({ lora_error: err.message || "Failed to update LoRA strength" })
        }
    }

    public async set_loras_enabled(enabled: boolean): Promise<void> {
        this.commit({ lora_error: null })
        try {
            await ApiClient.set_loras_enabled(this.api_url, enabled)
            await this.refresh_system_info()
        } catch (err: any) {
            this.commit({ lora_error: err.message || "Failed to update LoRA" })
        }
    }

    public async load_ip_adapter(
        uri: string,
        subfolder: string,
        weight_name: string,
        image_encoder_folder: string,
        strength: number
    ): Promise<boolean> {
        this.commit({ is_loading_ip_adapter: true, ip_adapter_error: null })
        try {
            await ApiClient.load_ip_adapter(this.api_url, uri, subfolder, weight_name, image_encoder_folder, strength)
            await this.refresh_system_info()
            this.commit({ is_loading_ip_adapter: false })
            return true
        } catch (err: any) {
            this.commit({ is_loading_ip_adapter: false, ip_adapter_error: err.message || "Failed to load IP-Adapter" })
            return false
        }
    }

    public async set_ip_adapter_strength(strength: number): Promise<void> {
        this.commit({ ip_adapter_error: null })
        try {
            await ApiClient.set_ip_adapter_strength(this.api_url, strength)
            await this.refresh_system_info()
        } catch (err: any) {
            this.commit({ ip_adapter_error: err.message || "Failed to update IP-Adapter strength" })
        }
    }

    public async delete_ip_adapter(): Promise<void> {
        this.commit({ ip_adapter_error: null })
        try {
            await ApiClient.delete_ip_adapter(this.api_url)
            await this.refresh_system_info()
        } catch (err: any) {
            this.commit({ ip_adapter_error: err.message || "Failed to remove IP-Adapter" })
        }
    }

    public async load_controlnet(model: string): Promise<void> {
        this.commit({ is_loading_controlnet: true, controlnet_error: null })
        try {
            await ApiClient.load_controlnet(this.api_url, model)
            await this.refresh_system_info()
            this.commit({ is_loading_controlnet: false })
        } catch (err: any) {
            this.commit({ is_loading_controlnet: false, controlnet_error: err.message || "Failed to load ControlNet" })
        }
    }

    public async offload_controlnet(): Promise<void> {
        this.commit({ is_loading_controlnet: true, controlnet_error: null })
        try {
            await ApiClient.offload_controlnet(this.api_url)
            await this.refresh_system_info()
            this.commit({ is_loading_controlnet: false })
        } catch (err: any) {
            this.commit({
                is_loading_controlnet: false,
                controlnet_error: err.message || "Failed to offload ControlNet"
            })
        }
    }

    public async preview_canny(image: string, low_threshold: number, high_threshold: number): Promise<string> {
        const res = await ApiClient.preview_canny(this.api_url, image, low_threshold, high_threshold)
        return res.image_base64
    }

    public async delete_lora(lora_name: string): Promise<void> {
        this.commit({ lora_error: null })
        try {
            await ApiClient.delete_lora(this.api_url, lora_name)
            await this.refresh_system_info()
        } catch (err: any) {
            this.commit({ lora_error: err.message || "Failed to delete LoRA" })
        }
    }
}

export const model_controller = new ModelController()
