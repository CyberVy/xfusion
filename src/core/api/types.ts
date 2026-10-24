export interface Txt2ImgParams {
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

export interface Img2ImgParams {
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

export interface InpaintParams {
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
    schedulers: string[]
}
