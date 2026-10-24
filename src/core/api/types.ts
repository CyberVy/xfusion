export interface GenerationExtras {
    control_image?: string | null
    controlnet_conditioning_scale?: number
    low_threshold?: number
    high_threshold?: number
    ip_adapter_image?: string | null
}

export interface Txt2ImgParams extends GenerationExtras {
    prompt: string
    negative_prompt?: string
    guidance_scale?: number
    num_inference_steps?: number
    clip_skip?: number
    width?: number
    height?: number
    seed?: number | null
    num?: number
    scheduler?: string | null
}

export interface Img2ImgParams extends GenerationExtras {
    image: string // Base64 data URL or HTTP URL
    prompt: string
    negative_prompt?: string
    strength?: number
    guidance_scale?: number
    num_inference_steps?: number
    clip_skip?: number
    width?: number
    height?: number
    seed?: number | null
    num?: number
    scheduler?: string | null
}

export interface InpaintParams extends GenerationExtras {
    image: string
    mask_image: string
    prompt: string
    negative_prompt?: string
    strength?: number
    guidance_scale?: number
    num_inference_steps?: number
    clip_skip?: number
    width?: number
    height?: number
    seed?: number | null
    num?: number
    scheduler?: string | null
}

export interface GenerationProgress {
    active: boolean
    step: number
    total_steps: number
    image_index: number
    image_count: number
}

export interface IpAdapterInfo {
    uri: string
    subfolder: string
    weight_name: string
    image_encoder_folder: string
    strength: number
}

export interface GenerationResponse {
    status: string
    images: { image_base64: string; seed: number; width: number; height: number }[]
    elapsed_seconds: number
}

export interface GeneratedImageItem {
    id: string
    image_base64: string
    seed: number
    prompt: string
    negative_prompt?: string
    width: number
    height: number
    created_at: number
    type: "txt2img" | "img2img" | "inpaint"
    elapsed_seconds?: number
}

export interface SystemInfo {
    status: string
    gpu_count: number
    gpu_names: string[]
    current_model: string | null
    model_version: string | null
    loras: Record<string, number>
    loras_enabled?: boolean
    schedulers: string[]
    controlnet_loaded?: boolean
    ip_adapter?: IpAdapterInfo | null
    python_version?: string
    os_platform?: string
}
