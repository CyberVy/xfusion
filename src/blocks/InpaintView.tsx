"use client"

import { useState, useRef, useEffect } from "react"
import { NaiveButton } from "@/components"
import { BrushIcon, WandIcon, RefreshIcon, DownloadIcon, TrashIcon } from "./icons"
import { generation_controller } from "@/core/controllers"
import { useController } from "@/components/hooks"

export function InpaintView() {
    const state = useController(generation_controller, {
        events: ["change", "status_change"]
    })

    const [source_image, set_source_image] = useState<string | null>(null)
    const [brush_size, set_brush_size] = useState(30)
    const [is_drawing, set_is_drawing] = useState(false)

    const file_input_ref = useRef<HTMLInputElement | null>(null)
    const canvas_ref = useRef<HTMLCanvasElement | null>(null)
    const image_ref = useRef<HTMLImageElement | null>(null)

    const handle_file_change = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (!file) return

        const reader = new FileReader()
        reader.onload = (event) => {
            const base64 = event.target?.result as string
            set_source_image(base64)
        }
        reader.readAsDataURL(file)
    }

    // Set canvas dimensions to match image when loaded
    useEffect(() => {
        if (!source_image || !canvas_ref.current) return
        const canvas = canvas_ref.current
        const ctx = canvas.getContext("2d")
        if (!ctx) return

        const img = new Image()
        img.onload = () => {
            canvas.width = img.width
            canvas.height = img.height
            ctx.clearRect(0, 0, canvas.width, canvas.height)
        }
        img.src = source_image
    }, [source_image])

    const start_drawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
        set_is_drawing(true)
        draw(e)
    }

    const stop_drawing = () => {
        set_is_drawing(false)
    }

    const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
        if (!is_drawing && e.type !== "mousedown") return
        const canvas = canvas_ref.current
        if (!canvas) return
        const ctx = canvas.getContext("2d")
        if (!ctx) return

        const rect = canvas.getBoundingClientRect()
        const scale_x = canvas.width / rect.width
        const scale_y = canvas.height / rect.height

        const x = (e.clientX - rect.left) * scale_x
        const y = (e.clientY - rect.top) * scale_y

        ctx.fillStyle = "rgba(255, 255, 255, 1.0)" // pure white mask
        ctx.beginPath()
        ctx.arc(x, y, brush_size, 0, Math.PI * 2)
        ctx.fill()
    }

    const clear_mask = () => {
        const canvas = canvas_ref.current
        if (!canvas) return
        const ctx = canvas.getContext("2d")
        if (!ctx) return
        ctx.clearRect(0, 0, canvas.width, canvas.height)
    }

    const handle_generate = () => {
        if (!source_image || !canvas_ref.current) {
            generation_controller.set_params({ error: "Please upload an image and draw a mask" })
            return
        }

        const mask_base64 = canvas_ref.current.toDataURL("image/png")
        generation_controller.generate_inpaint(source_image, mask_base64, 0.8)
    }

    const latest_image = state.history.find((x) => x.type === "inpaint") || state.history[0]

    return (
        <div className="w-full max-w-4xl mx-auto px-4 py-6 pb-28">
            <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
                {/* Left Controls */}
                <div className="md:col-span-7 flex flex-col gap-4">
                    <div className="flex items-center justify-between">
                        <label className="text-xs font-semibold text-black/60 dark:text-white/60 uppercase tracking-wider">
                            Inpainting Canvas (Draw Mask)
                        </label>
                        {source_image && (
                            <div className="flex items-center gap-2">
                                <span className="text-[11px] text-black/50 dark:text-white/50">Brush: {brush_size}px</span>
                                <input
                                    type="range"
                                    min={5}
                                    max={80}
                                    value={brush_size}
                                    onChange={(e) => set_brush_size(Number(e.target.value))}
                                    className="w-20 accent-black dark:accent-white cursor-pointer"
                                />
                                <NaiveButton
                                    className="p-1 text-black/50 dark:text-white/50 hover:text-rose-500"
                                    on_click={clear_mask}
                                    title="Clear Mask"
                                >
                                    <TrashIcon width={14} height={14} />
                                </NaiveButton>
                            </div>
                        )}
                    </div>

                    <input
                        type="file"
                        accept="image/*"
                        ref={file_input_ref}
                        onChange={handle_file_change}
                        className="hidden"
                    />

                    <div className="relative w-full h-64 rounded-2xl border border-black/15 dark:border-white/15 overflow-hidden bg-black/5 dark:bg-white/5 flex items-center justify-center">
                        {source_image ? (
                            <div className="relative w-full h-full flex items-center justify-center select-none">
                                <img
                                    ref={image_ref}
                                    src={source_image}
                                    alt="Source canvas"
                                    className="max-h-full max-w-full object-contain pointer-events-none"
                                />
                                <canvas
                                    ref={canvas_ref}
                                    onMouseDown={start_drawing}
                                    onMouseUp={stop_drawing}
                                    onMouseMove={draw}
                                    onMouseLeave={stop_drawing}
                                    className="absolute inset-0 w-full h-full object-contain cursor-crosshair opacity-75"
                                />
                            </div>
                        ) : (
                            <div
                                onClick={() => file_input_ref.current?.click()}
                                className="cursor-pointer text-center text-black/40 dark:text-white/40 text-xs p-6"
                            >
                                <BrushIcon width={32} height={32} className="mx-auto mb-2 opacity-60" />
                                <p className="font-medium">Upload image to start inpainting</p>
                            </div>
                        )}
                    </div>

                    {/* Prompt Box */}
                    <div className="flex flex-col gap-1.5">
                        <label className="text-xs font-semibold text-black/60 dark:text-white/60 uppercase tracking-wider">
                            Inpainting Prompt
                        </label>
                        <textarea
                            value={state.prompt}
                            onChange={(e) => generation_controller.set_params({ prompt: e.target.value })}
                            placeholder="Describe what to fill in the masked area..."
                            rows={3}
                            className="w-full p-3.5 text-sm rounded-2xl bg-black/5 dark:bg-white/5 border border-black/10 dark:border-white/10 focus:outline-none focus:ring-1 focus:ring-black/30 dark:focus:ring-white/30 resize-none transition"
                        />
                    </div>

                    {state.error && (
                        <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-600 dark:text-rose-400 text-xs">
                            {state.error}
                        </div>
                    )}

                    <NaiveButton
                        className="w-full py-3.5 rounded-2xl bg-black dark:bg-white text-white dark:text-black font-semibold text-sm shadow-md hover:opacity-90 disabled:opacity-50"
                        on_click={handle_generate}
                        disabled={state.is_generating || !source_image}
                    >
                        {state.is_generating ? (
                            <span className="flex items-center gap-2">
                                <RefreshIcon width={16} height={16} className="animate-spin" />
                                Inpainting on GPU...
                            </span>
                        ) : (
                            <span className="flex items-center gap-2">
                                <WandIcon width={18} height={18} />
                                Inpaint
                            </span>
                        )}
                    </NaiveButton>
                </div>

                {/* Right Output */}
                <div className="md:col-span-5 flex flex-col items-center justify-center">
                    <div className="w-full aspect-square rounded-3xl bg-black/5 dark:bg-white/5 border border-black/10 dark:border-white/10 flex items-center justify-center overflow-hidden relative group">
                        {state.is_generating ? (
                            <div className="flex flex-col items-center gap-3">
                                <RefreshIcon width={32} height={32} className="animate-spin text-black/40 dark:text-white/40" />
                                <span className="text-xs text-black/50 dark:text-white/50 animate-pulse">
                                    Inpainting on GPU...
                                </span>
                            </div>
                        ) : latest_image ? (
                            <>
                                <img
                                    src={latest_image.image_base64}
                                    alt="Inpainted output"
                                    className="w-full h-full object-cover select-none"
                                />
                                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition flex items-end p-4 justify-between">
                                    <div className="text-white text-xs truncate max-w-[200px]">
                                        <p className="font-medium truncate">{latest_image.prompt}</p>
                                        <p className="text-[10px] opacity-70">Seed: {latest_image.seed}</p>
                                    </div>
                                    <a
                                        href={latest_image.image_base64}
                                        download={`xfusion_inpaint_${latest_image.seed}.png`}
                                        className="p-2 rounded-full bg-white text-black hover:scale-105 transition"
                                        title="Download"
                                    >
                                        <DownloadIcon width={16} height={16} />
                                    </a>
                                </div>
                            </>
                        ) : (
                            <div className="text-center p-6 text-black/30 dark:text-white/30 text-xs">
                                <BrushIcon width={40} height={40} className="mx-auto mb-2 opacity-40" />
                                <p>Inpainted image will appear here</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}
