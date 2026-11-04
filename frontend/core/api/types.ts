export interface GenerationBody {
    prompt: string
    negative_prompt?: string
    guidance?: number | null
    steps?: number
    clip_skip?: number | null
    width?: number | null
    height?: number | null
    seed?: number | null
    image_count?: number
    scheduler?: string | null
    image?: string
    mask?: string
    strength?: number
    control_image?: string | null
    control_scale?: number | null
    canny_low?: number
    canny_high?: number
    reference_image?: string | null
}

export interface GenerationProgress {
    active: boolean
    step: number
    total_steps: number
    image_index: number
    image_count: number
}

export interface IpAdapterInfo {
    source: string
    folder: string
    weight: string
    encoder_folder: string
    strength: number
}

export interface GenerationResponse {
    status: string
    images: { image_base64: string; seed: number; width: number; height: number }[]
    elapsed_seconds: number
}

export interface StreamImageEvent {
    image_base64: string
    seed: number
    width: number
    height: number
    index: number
    elapsed_seconds: number
}

export interface StreamDoneEvent {
    status: string
    total_images: number
    elapsed_seconds: number
}

export type StreamGenerationEvent =
    | { type: "image"; data: StreamImageEvent }
    | { type: "done"; data: StreamDoneEvent }
    | { type: "progress"; data: GenerationProgress }

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

export interface ModelLoadOptions {
    model_version?: string | null
    backbone?: string
    text_encoder?: string
    memory?: string
    delete_file_after_load?: boolean
}

export interface SystemInfo {
    status: string
    gpu_count: number
    gpu_names: string[]
    current_model: string | null
    model_version: string | null
    backbone?: string
    text_encoder?: string
    memory?: string
    adapters?: string[]
    default_size?: number
    default_guidance?: number
    accepts_negative_prompt?: boolean
    accepts_clip_skip?: boolean
    loras: Record<string, number>
    loras_enabled?: boolean
    schedulers: string[]
    controlnet_loaded?: boolean
    ip_adapter?: IpAdapterInfo | null
    python_version?: string
    os_platform?: string
}
