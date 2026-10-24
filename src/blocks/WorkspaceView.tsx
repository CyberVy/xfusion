"use client"

import { useState } from "react"
import type { ReactNode } from "react"
import { NaiveButton } from "@/components"
import { useController } from "@/components/hooks"
import { generation_controller, model_controller } from "@/core/controllers"
import { ImageSourceInput } from "./ImageSourceInput"
import { MaskPainter } from "./MaskPainter"
import { ResultPanel } from "./ResultPanel"
import { RefreshIcon, WandIcon } from "./icons"

type WorkspaceMode = "txt2img" | "img2img" | "inpaint"

const MODES: { id: WorkspaceMode; label: string }[] = [
    { id: "txt2img", label: "Text to Image" },
    { id: "img2img", label: "Image to Image" },
    { id: "inpaint", label: "Inpaint" }
]

const SIZE_PRESETS = [
    { label: "1:1", width: 1024, height: 1024 },
    { label: "3:4", width: 896, height: 1152 },
    { label: "4:3", width: 1152, height: 896 },
    { label: "9:16", width: 768, height: 1344 },
    { label: "16:9", width: 1344, height: 768 },
    { label: "512", width: 512, height: 512 }
]

const FIELD_CLASS =
    "w-full rounded-xl bg-black/4 dark:bg-white/6 border border-black/8 dark:border-white/10 focus:outline-none focus:border-black/30 dark:focus:border-white/30 transition"

function Section({ title, children }: { title: string; children: ReactNode }) {
    return (
        <section className="flex flex-col gap-2.5">
            <h3 className="text-xs font-medium text-black/50 dark:text-white/50">{title}</h3>
            {children}
        </section>
    )
}

type ParamSliderProps = {
    label: string
    value: number
    min: number
    max: number
    step: number
    on_change: (value: number) => void
}

function ParamSlider({ label, value, min, max, step, on_change }: ParamSliderProps) {
    return (
        <label className="flex flex-col gap-1.5">
            <span className="flex justify-between text-sm">
                <span>{label}</span>
                <span className="font-mono text-black/60 dark:text-white/60">{value}</span>
            </span>
            <input
                type="range"
                min={min}
                max={max}
                step={step}
                value={value}
                onChange={(e) => on_change(Number(e.target.value))}
                className="w-full accent-black dark:accent-white cursor-pointer"
            />
        </label>
    )
}

export function WorkspaceView() {
    const state = useController(generation_controller, { events: ["change"] })
    const model_state = useController(model_controller, { events: ["change"] })

    const [mode, set_mode] = useState<WorkspaceMode>("txt2img")
    const [source_image, set_source_image] = useState<string | null>(null)
    const [mask_image, set_mask_image] = useState<string | null>(null)

    const is_flux = model_state.system_info?.model_version === "flux"
    const schedulers = model_state.system_info?.schedulers ?? []
    const needs_source = mode !== "txt2img"
    const can_generate =
        !state.is_generating && (!needs_source || source_image !== null) && (mode !== "inpaint" || mask_image !== null)

    const handle_mode_change = (next_mode: WorkspaceMode) => {
        set_mode(next_mode)
        generation_controller.clear_error()
    }

    const handle_source_change = (image: string | null) => {
        set_source_image(image)
        set_mask_image(null)
    }

    const handle_generate = () => {
        if (mode === "txt2img") {
            generation_controller.generate_txt2img()
        } else if (mode === "img2img" && source_image) {
            generation_controller.generate_img2img(source_image)
        } else if (mode === "inpaint" && source_image && mask_image) {
            generation_controller.generate_inpaint(source_image, mask_image)
        }
    }

    let generate_hint: string | null = null
    if (needs_source && !source_image) generate_hint = "Add a source image to continue"
    else if (mode === "inpaint" && !mask_image) generate_hint = "Paint over the area you want to change"

    return (
        <div className="w-full max-w-7xl mx-auto px-4 md:px-6 pt-4 md:pt-6 pb-32">
            <div className="grid grid-cols-1 md:grid-cols-[minmax(340px,420px)_1fr] gap-6 md:gap-8 items-start">
                <div className="md:order-last md:sticky md:top-6">
                    <ResultPanel />
                </div>

                <div className="flex flex-col gap-6">
                    <div className="grid grid-cols-3 p-1 rounded-full bg-black/5 dark:bg-white/8">
                        {MODES.map((item) => (
                            <button
                                key={item.id}
                                type="button"
                                onClick={() => handle_mode_change(item.id)}
                                className={`h-9 rounded-full text-xs sm:text-sm font-medium cursor-pointer transition ${
                                    mode === item.id
                                        ? "bg-white text-black shadow-sm dark:bg-white/15 dark:text-white"
                                        : "text-black/55 dark:text-white/55 hover:text-black dark:hover:text-white"
                                }`}
                            >
                                {item.label}
                            </button>
                        ))}
                    </div>

                    {needs_source && (
                        <Section title={mode === "inpaint" ? "Source image and mask" : "Source image"}>
                            <ImageSourceInput image={source_image} on_change={handle_source_change}>
                                {source_image &&
                                    (mode === "inpaint" ? (
                                        <MaskPainter
                                            key={source_image}
                                            image_src={source_image}
                                            on_mask_change={set_mask_image}
                                        />
                                    ) : (
                                        <img
                                            src={source_image}
                                            alt="Source"
                                            className="max-w-full max-h-[40vh] rounded-xl object-contain select-none"
                                        />
                                    ))}
                            </ImageSourceInput>
                        </Section>
                    )}

                    <Section title="Prompt">
                        <textarea
                            value={state.prompt}
                            onChange={(e) => generation_controller.set_params({ prompt: e.target.value })}
                            placeholder={
                                mode === "inpaint"
                                    ? "Describe what should appear in the painted area"
                                    : "Describe the image you want"
                            }
                            rows={4}
                            className={`${FIELD_CLASS} p-3.5 text-sm leading-relaxed resize-y`}
                        />
                        {!is_flux && (
                            <textarea
                                value={state.negative_prompt}
                                onChange={(e) => generation_controller.set_params({ negative_prompt: e.target.value })}
                                placeholder="Negative prompt"
                                rows={2}
                                className={`${FIELD_CLASS} p-3 text-xs leading-relaxed text-black/70 dark:text-white/70 resize-y`}
                            />
                        )}
                    </Section>

                    <div className="flex flex-col gap-2">
                        <NaiveButton
                            className="w-full h-12 rounded-2xl bg-black text-white dark:bg-white dark:text-black text-sm font-semibold hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed"
                            on_click={handle_generate}
                            disabled={!can_generate}
                        >
                            {state.is_generating ? (
                                <span className="flex items-center gap-2">
                                    <RefreshIcon width={16} height={16} className="animate-spin" />
                                    Generating
                                </span>
                            ) : (
                                <span className="flex items-center gap-2">
                                    <WandIcon width={18} height={18} />
                                    Generate
                                </span>
                            )}
                        </NaiveButton>
                        {generate_hint && !state.is_generating && (
                            <p className="text-xs text-center text-black/45 dark:text-white/45">{generate_hint}</p>
                        )}
                        {state.error && (
                            <p className="p-3 rounded-xl text-xs leading-relaxed bg-black/5 dark:bg-white/8 border border-black/15 dark:border-white/20 break-words">
                                {state.error}
                            </p>
                        )}
                    </div>

                    {mode === "txt2img" ? (
                        <Section title="Size">
                            <div className="grid grid-cols-6 gap-1.5">
                                {SIZE_PRESETS.map((item) => {
                                    const active = state.width === item.width && state.height === item.height
                                    return (
                                        <button
                                            key={item.label}
                                            type="button"
                                            onClick={() =>
                                                generation_controller.set_params({
                                                    width: item.width,
                                                    height: item.height
                                                })
                                            }
                                            title={`${item.width} × ${item.height}`}
                                            className={`h-9 rounded-lg text-xs font-medium cursor-pointer transition ${
                                                active
                                                    ? "bg-black text-white dark:bg-white dark:text-black"
                                                    : "bg-black/4 dark:bg-white/6 hover:bg-black/8 dark:hover:bg-white/12"
                                            }`}
                                        >
                                            {item.label}
                                        </button>
                                    )
                                })}
                            </div>
                            <p className="text-xs text-black/45 dark:text-white/45 font-mono">
                                {state.width} × {state.height}
                            </p>
                        </Section>
                    ) : (
                        <Section title="Strength">
                            <ParamSlider
                                label={mode === "inpaint" ? "Inpaint strength" : "Denoising strength"}
                                value={mode === "inpaint" ? state.inpaint_strength : state.img2img_strength}
                                min={0.05}
                                max={1}
                                step={0.05}
                                on_change={(value) =>
                                    generation_controller.set_params(
                                        mode === "inpaint" ? { inpaint_strength: value } : { img2img_strength: value }
                                    )
                                }
                            />
                            <p className="text-xs text-black/45 dark:text-white/45">
                                Lower keeps more of the source. Output keeps the source aspect ratio.
                            </p>
                        </Section>
                    )}

                    <Section title="Sampling">
                        <ParamSlider
                            label="Steps"
                            value={state.num_inference_steps}
                            min={1}
                            max={50}
                            step={1}
                            on_change={(value) => generation_controller.set_params({ num_inference_steps: value })}
                        />
                        <ParamSlider
                            label="Guidance scale"
                            value={state.guidance_scale}
                            min={0}
                            max={15}
                            step={0.5}
                            on_change={(value) => generation_controller.set_params({ guidance_scale: value })}
                        />
                        {schedulers.length > 0 && (
                            <label className="flex flex-col gap-1.5">
                                <span className="text-sm">Scheduler</span>
                                <select
                                    value={schedulers.includes(state.scheduler) ? state.scheduler : schedulers[0]}
                                    onChange={(e) => generation_controller.set_params({ scheduler: e.target.value })}
                                    className={`${FIELD_CLASS} h-10 px-3 text-sm cursor-pointer`}
                                >
                                    {schedulers.map((name) => (
                                        <option key={name} value={name}>
                                            {name}
                                        </option>
                                    ))}
                                </select>
                            </label>
                        )}
                        <div className="flex flex-col gap-1.5">
                            <span className="text-sm">Seed</span>
                            <div className="flex gap-2">
                                <input
                                    type="number"
                                    value={state.seed ?? ""}
                                    placeholder="Random"
                                    onChange={(e) =>
                                        generation_controller.set_params({
                                            seed: e.target.value ? Number(e.target.value) : null
                                        })
                                    }
                                    className={`${FIELD_CLASS} flex-1 h-10 px-3 text-sm font-mono`}
                                />
                                <NaiveButton
                                    className="h-10 px-3.5 rounded-xl text-xs font-medium bg-black/5 dark:bg-white/8 hover:bg-black/10 dark:hover:bg-white/12"
                                    on_click={() =>
                                        generation_controller.set_params({
                                            seed: Math.floor(Math.random() * 2147483646) + 1
                                        })
                                    }
                                >
                                    Pick
                                </NaiveButton>
                                <NaiveButton
                                    className="h-10 px-3.5 rounded-xl text-xs font-medium bg-black/5 dark:bg-white/8 hover:bg-black/10 dark:hover:bg-white/12 disabled:opacity-40"
                                    on_click={() => generation_controller.set_params({ seed: null })}
                                    disabled={state.seed === null}
                                >
                                    Random
                                </NaiveButton>
                            </div>
                        </div>
                    </Section>
                </div>
            </div>
        </div>
    )
}
