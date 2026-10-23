"use client"

import { useState, useRef } from "react"
import { NaiveButton } from "@/components"
import { ImageIcon, WandIcon, RefreshIcon, DownloadIcon } from "./icons"
import { generation_controller } from "@/core/controllers"
import { useController } from "@/components/hooks"

export function Img2ImgView() {
    const state = useController(generation_controller, {
        events: ["change", "status_change"]
    })

    const [input_image, set_input_image] = useState<string | null>(null)
    const [strength, set_strength] = useState(0.75)
    const file_input_ref = useRef<HTMLInputElement | null>(null)

    const handle_file_change = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (!file) return

        const reader = new FileReader()
        reader.onload = (event) => {
            const base64 = event.target?.result as string
            set_input_image(base64)
        }
        reader.readAsDataURL(file)
    }

    const handle_generate = () => {
        if (!input_image) {
            generation_controller.set_params({ error: "Please upload an initial image" })
            return
        }
        generation_controller.generate_img2img(input_image, strength)
    }

    const latest_image = state.history.find((x) => x.type === "img2img") || state.history[0]

    return (
        <div className="w-full max-w-4xl mx-auto px-4 py-6 pb-28">
            <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
                {/* Left Controls */}
                <div className="md:col-span-7 flex flex-col gap-4">
                    {/* Image Upload Area */}
                    <div className="flex flex-col gap-1.5">
                        <label className="text-xs font-semibold text-black/60 dark:text-white/60 uppercase tracking-wider">
                            Initial Image
                        </label>
                        <input
                            type="file"
                            accept="image/*"
                            ref={file_input_ref}
                            onChange={handle_file_change}
                            className="hidden"
                        />
                        <div
                            onClick={() => file_input_ref.current?.click()}
                            className="w-full h-44 rounded-2xl border-2 border-dashed border-black/15 dark:border-white/15 hover:border-black/30 dark:hover:border-white/30 cursor-pointer flex flex-col items-center justify-center p-3 overflow-hidden bg-black/5 dark:bg-white/5 transition"
                        >
                            {input_image ? (
                                <img
                                    src={input_image}
                                    alt="Input source"
                                    className="h-full object-contain rounded-xl select-none"
                                />
                            ) : (
                                <div className="text-center text-black/40 dark:text-white/40 text-xs">
                                    <ImageIcon width={32} height={32} className="mx-auto mb-1.5 opacity-60" />
                                    <p className="font-medium">Click or Drag & Drop to upload image</p>
                                    <p className="text-[10px] opacity-60 mt-0.5">PNG, JPG, WEBP</p>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Denoising Strength */}
                    <div className="flex flex-col gap-1.5">
                        <div className="flex justify-between text-xs">
                            <span className="font-semibold text-black/60 dark:text-white/60 uppercase tracking-wider">
                                Denoising Strength
                            </span>
                            <span className="font-mono text-xs">{strength}</span>
                        </div>
                        <input
                            type="range"
                            min={0.1}
                            max={0.95}
                            step={0.05}
                            value={strength}
                            onChange={(e) => set_strength(Number(e.target.value))}
                            className="w-full accent-black dark:accent-white cursor-pointer"
                        />
                        <div className="flex justify-between text-[10px] text-black/40 dark:text-white/40">
                            <span>More like original (0.1)</span>
                            <span>More creative (0.95)</span>
                        </div>
                    </div>

                    {/* Prompt Box */}
                    <div className="flex flex-col gap-1.5">
                        <label className="text-xs font-semibold text-black/60 dark:text-white/60 uppercase tracking-wider">
                            Prompt
                        </label>
                        <textarea
                            value={state.prompt}
                            onChange={(e) => generation_controller.set_params({ prompt: e.target.value })}
                            placeholder="Describe how to transform this image..."
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
                        disabled={state.is_generating || !input_image}
                    >
                        {state.is_generating ? (
                            <span className="flex items-center gap-2">
                                <RefreshIcon width={16} height={16} className="animate-spin" />
                                Processing Image...
                            </span>
                        ) : (
                            <span className="flex items-center gap-2">
                                <WandIcon width={18} height={18} />
                                Generate Image to Image
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
                                    Transforming on GPU...
                                </span>
                            </div>
                        ) : latest_image ? (
                            <>
                                <img
                                    src={latest_image.image_base64}
                                    alt="Generated output"
                                    className="w-full h-full object-cover select-none"
                                />
                                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition flex items-end p-4 justify-between">
                                    <div className="text-white text-xs truncate max-w-[200px]">
                                        <p className="font-medium truncate">{latest_image.prompt}</p>
                                        <p className="text-[10px] opacity-70">Seed: {latest_image.seed}</p>
                                    </div>
                                    <a
                                        href={latest_image.image_base64}
                                        download={`xfusion_img2img_${latest_image.seed}.png`}
                                        className="p-2 rounded-full bg-white text-black hover:scale-105 transition"
                                        title="Download"
                                    >
                                        <DownloadIcon width={16} height={16} />
                                    </a>
                                </div>
                            </>
                        ) : (
                            <div className="text-center p-6 text-black/30 dark:text-white/30 text-xs">
                                <ImageIcon width={40} height={40} className="mx-auto mb-2 opacity-40" />
                                <p>Img2Img output will appear here</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}
