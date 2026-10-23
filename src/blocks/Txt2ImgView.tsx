"use client"

import { useState } from "react"
import { NaiveButton } from "@/components"
import { WandIcon, DownloadIcon, RefreshIcon } from "./icons"
import { generation_controller } from "@/core/controllers"
import { useController } from "@/components/hooks"

const ASPECT_RATIOS = [
    { label: "1:1 Square", width: 1024, height: 1024 },
    { label: "9:16 Portrait", width: 768, height: 1344 },
    { label: "16:9 Landscape", width: 1344, height: 768 },
    { label: "3:4 Classic", width: 896, height: 1152 },
    { label: "4:3 Standard", width: 1152, height: 896 },
    { label: "512 SD1.5", width: 512, height: 512 }
]

export function Txt2ImgView() {
    const state = useController(generation_controller, {
        events: ["change", "status_change", "params_change"]
    })

    const [show_advanced, set_show_advanced] = useState(false)

    const handle_generate = () => {
        generation_controller.generate_txt2img()
    }

    const handle_aspect_change = (w: number, h: number) => {
        generation_controller.set_params({ width: w, height: h })
    }

    const randomize_seed = () => {
        generation_controller.set_params({ seed: Math.floor(Math.random() * 2147483647) })
    }

    const latest_image = state.history.find((x) => x.type === "txt2img") || state.history[0]

    return (
        <div className="w-full max-w-4xl mx-auto px-4 py-6 pb-28">
            <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
                {/* Generation Controls Panel */}
                <div className="md:col-span-7 flex flex-col gap-4">
                    {/* Prompt Box */}
                    <div className="flex flex-col gap-1.5">
                        <label className="text-xs font-semibold text-black/60 dark:text-white/60 uppercase tracking-wider">
                            Prompt
                        </label>
                        <textarea
                            value={state.prompt}
                            onChange={(e) => generation_controller.set_params({ prompt: e.target.value })}
                            placeholder="A cinematic shot of a majestic cyberpunk city in neon rain, 8k, photorealistic..."
                            rows={4}
                            className="w-full p-3.5 text-sm rounded-2xl bg-black/5 dark:bg-white/5 border border-black/10 dark:border-white/10 focus:outline-none focus:ring-1 focus:ring-black/30 dark:focus:ring-white/30 resize-none transition"
                        />
                    </div>

                    {/* Negative Prompt (Collapsible) */}
                    <div>
                        <button
                            type="button"
                            onClick={() => set_show_advanced((prev) => !prev)}
                            className="text-xs text-black/50 dark:text-white/50 hover:text-black dark:hover:text-white flex items-center gap-1 transition select-none"
                        >
                            <span>{show_advanced ? "Hide Negative Prompt & Settings" : "+ Advanced Settings"}</span>
                        </button>

                        {show_advanced && (
                            <div className="mt-3 p-4 rounded-2xl bg-black/5 dark:bg-white/5 border border-black/10 dark:border-white/10 flex flex-col gap-4">
                                <div className="flex flex-col gap-1">
                                    <label className="text-[11px] font-semibold text-black/50 dark:text-white/50 uppercase">
                                        Negative Prompt
                                    </label>
                                    <textarea
                                        value={state.negative_prompt}
                                        onChange={(e) =>
                                            generation_controller.set_params({ negative_prompt: e.target.value })
                                        }
                                        rows={2}
                                        className="w-full p-2.5 text-xs rounded-xl bg-white dark:bg-[#1a1a1a] border border-black/10 dark:border-white/10 focus:outline-none resize-none"
                                    />
                                </div>

                                <div className="grid grid-cols-2 gap-3 text-xs">
                                    <div>
                                        <div className="flex justify-between text-[11px] mb-1">
                                            <span>Steps: {state.num_inference_steps}</span>
                                        </div>
                                        <input
                                            type="range"
                                            min={10}
                                            max={50}
                                            value={state.num_inference_steps}
                                            onChange={(e) =>
                                                generation_controller.set_params({
                                                    num_inference_steps: Number(e.target.value)
                                                })
                                            }
                                            className="w-full accent-black dark:accent-white cursor-pointer"
                                        />
                                    </div>

                                    <div>
                                        <div className="flex justify-between text-[11px] mb-1">
                                            <span>CFG Scale: {state.guidance_scale}</span>
                                        </div>
                                        <input
                                            type="range"
                                            min={1.0}
                                            max={15.0}
                                            step={0.5}
                                            value={state.guidance_scale}
                                            onChange={(e) =>
                                                generation_controller.set_params({
                                                    guidance_scale: Number(e.target.value)
                                                })
                                            }
                                            className="w-full accent-black dark:accent-white cursor-pointer"
                                        />
                                    </div>
                                </div>

                                <div className="flex items-center gap-2">
                                    <input
                                        type="number"
                                        value={state.seed ?? ""}
                                        placeholder="Seed (Random)"
                                        onChange={(e) =>
                                            generation_controller.set_params({
                                                seed: e.target.value ? Number(e.target.value) : null
                                            })
                                        }
                                        className="flex-1 py-1.5 px-3 text-xs rounded-lg bg-white dark:bg-[#1a1a1a] border border-black/10 dark:border-white/10"
                                    />
                                    <NaiveButton
                                        className="px-3 py-1.5 rounded-lg border border-black/10 dark:border-white/10 text-xs bg-white dark:bg-[#1a1a1a]"
                                        on_click={randomize_seed}
                                    >
                                        <RefreshIcon width={12} height={12} className="mr-1" />
                                        Random
                                    </NaiveButton>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Aspect Ratio Selection */}
                    <div className="flex flex-col gap-1.5">
                        <label className="text-xs font-semibold text-black/60 dark:text-white/60 uppercase tracking-wider">
                            Aspect Ratio
                        </label>
                        <div className="grid grid-cols-3 gap-2">
                            {ASPECT_RATIOS.map((item) => {
                                const active = state.width === item.width && state.height === item.height
                                return (
                                    <button
                                        key={item.label}
                                        type="button"
                                        onClick={() => handle_aspect_change(item.width, item.height)}
                                        className={`py-2 px-3 rounded-xl text-xs font-medium border transition cursor-pointer text-left ${
                                            active
                                                ? "bg-black text-white dark:bg-white dark:text-black border-transparent shadow-sm"
                                                : "bg-black/5 dark:bg-white/5 border-black/10 dark:border-white/10 hover:bg-black/10 dark:hover:bg-white/10"
                                        }`}
                                    >
                                        <div className="font-semibold">{item.label}</div>
                                        <div className="text-[10px] opacity-60">
                                            {item.width} × {item.height}
                                        </div>
                                    </button>
                                )
                            })}
                        </div>
                    </div>

                    {/* Error display */}
                    {state.error && (
                        <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-600 dark:text-rose-400 text-xs">
                            {state.error}
                        </div>
                    )}

                    {/* Generate Button */}
                    <NaiveButton
                        className="w-full py-3.5 rounded-2xl bg-black dark:bg-white text-white dark:text-black font-semibold text-sm shadow-md hover:opacity-90 disabled:opacity-50"
                        on_click={handle_generate}
                        disabled={state.is_generating}
                    >
                        {state.is_generating ? (
                            <span className="flex items-center gap-2">
                                <RefreshIcon width={16} height={16} className="animate-spin" />
                                Generating Artwork...
                            </span>
                        ) : (
                            <span className="flex items-center gap-2">
                                <WandIcon width={18} height={18} />
                                Generate
                            </span>
                        )}
                    </NaiveButton>
                </div>

                {/* Preview / Result Output Panel */}
                <div className="md:col-span-5 flex flex-col items-center justify-center">
                    <div className="w-full aspect-square rounded-3xl bg-black/5 dark:bg-white/5 border border-black/10 dark:border-white/10 flex items-center justify-center overflow-hidden relative group">
                        {state.is_generating ? (
                            <div className="flex flex-col items-center gap-3">
                                <RefreshIcon width={32} height={32} className="animate-spin text-black/40 dark:text-white/40" />
                                <span className="text-xs text-black/50 dark:text-white/50 animate-pulse">
                                    Denoising on GPU...
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
                                        <p className="text-[10px] opacity-70">
                                            Seed: {latest_image.seed} • {latest_image.elapsed_seconds}s
                                        </p>
                                    </div>
                                    <a
                                        href={latest_image.image_base64}
                                        download={`xfusion_${latest_image.seed}.png`}
                                        className="p-2 rounded-full bg-white text-black hover:scale-105 transition"
                                        title="Download"
                                    >
                                        <DownloadIcon width={16} height={16} />
                                    </a>
                                </div>
                            </>
                        ) : (
                            <div className="text-center p-6 text-black/30 dark:text-white/30 text-xs">
                                <WandIcon width={40} height={40} className="mx-auto mb-2 opacity-40" />
                                <p>Generated artwork will appear here</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}
