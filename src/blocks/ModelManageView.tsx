"use client"

import { useState } from "react"
import { NaiveButton, StringInput } from "@/components"
import { CubeIcon, RefreshIcon, TrashIcon } from "./icons"
import { generation_controller, settings_controller } from "@/core/controllers"
import { ApiClient } from "@/core/api/client"
import { useController } from "@/components/hooks"

const MODEL_VERSIONS = [
    { label: "SDXL", value: "xl" },
    { label: "Stable Diffusion 1.5", value: "1.5" },
    { label: "Stable Diffusion 2.1", value: "2" },
    { label: "Stable Diffusion 3 / 3.5", value: "3" },
    { label: "FLUX", value: "flux" }
]

export function ModelManageView() {
    const state = useController(generation_controller, {
        events: ["system_info_change", "change"]
    })

    const [model_url, set_model_url] = useState("eramth/realism-sdxl")
    const [model_version, set_model_version] = useState("xl")
    const [is_loading_model, set_is_loading_model] = useState(false)
    const [model_msg, set_model_msg] = useState<string | null>(null)

    // LoRA inputs
    const [lora_url, set_lora_url] = useState("")
    const [lora_name, set_lora_name] = useState("")
    const [lora_strength, set_lora_strength] = useState(0.4)
    const [is_loading_lora, set_is_loading_lora] = useState(false)

    const api_url = settings_controller.get_value("api_url", "http://localhost:8000")

    const handle_load_model = async () => {
        if (!model_url.trim()) return
        set_is_loading_model(true)
        set_model_msg(null)
        try {
            await ApiClient.load_model(api_url, model_url, model_version)
            await generation_controller.refresh_system_info()
            set_model_msg("Model loaded successfully!")
        } catch (err: any) {
            set_model_msg(`Error: ${err.message}`)
        } finally {
            set_is_loading_model(false)
        }
    }

    const handle_load_lora = async () => {
        if (!lora_url.trim() || !lora_name.trim()) return
        set_is_loading_lora(true)
        try {
            await ApiClient.load_lora(api_url, lora_url, lora_name, lora_strength)
            await generation_controller.refresh_system_info()
            set_lora_url("")
            set_lora_name("")
        } catch (err: any) {
            alert(`Failed to load LoRA: ${err.message}`)
        } finally {
            set_is_loading_lora(false)
        }
    }

    const handle_delete_lora = async (name: string) => {
        try {
            await ApiClient.delete_lora(api_url, name)
            await generation_controller.refresh_system_info()
        } catch (err: any) {
            alert(`Failed to delete LoRA: ${err.message}`)
        }
    }

    const handle_scheduler_change = async (scheduler: string) => {
        generation_controller.set_params({ scheduler })
        try {
            await ApiClient.set_scheduler(api_url, scheduler)
        } catch {
            // Ignore if backend isn't loaded yet
        }
    }

    const active_loras = state.system_info?.loras ? Object.entries(state.system_info.loras) : []
    const available_schedulers = state.system_info?.schedulers || [
        "DPM++ 2M", "DPM++ 2M KARRAS", "DPM++ 2M SDE", "DPM++ 2M SDE KARRAS",
        "EULER", "EULER A", "DEIS", "UNIPC"
    ]

    return (
        <div className="w-full max-w-3xl mx-auto px-4 py-6 pb-28 flex flex-col gap-6">
            {/* Base Model Management */}
            <div className="p-5 rounded-3xl bg-black/5 dark:bg-white/5 border border-black/10 dark:border-white/10 flex flex-col gap-4">
                <div className="flex items-center justify-between">
                    <div>
                        <h3 className="text-sm font-semibold">Diffusion Base Model</h3>
                        <p className="text-xs text-black/50 dark:text-white/50">
                            Current: <span className="font-mono text-black dark:text-white">{state.system_info?.current_model || "None Loaded"}</span>
                        </p>
                    </div>
                    <span className="text-xs uppercase font-mono px-2 py-0.5 rounded-md bg-black/10 dark:bg-white/10">
                        {state.system_info?.model_version || "N/A"}
                    </span>
                </div>

                <div className="flex flex-col gap-3">
                    <StringInput
                        value={model_url}
                        on_change={set_model_url}
                        placeholder="Model URL or Hugging Face repo ID (e.g. eramth/realism-sdxl)"
                        className="py-2.5 px-3 bg-white dark:bg-[#1a1a1a] border border-black/10 dark:border-white/10 rounded-xl text-xs"
                    />

                    <div className="flex gap-2">
                        <select
                            value={model_version}
                            onChange={(e) => set_model_version(e.target.value)}
                            className="flex-1 py-2 px-3 rounded-xl bg-white dark:bg-[#1a1a1a] border border-black/10 dark:border-white/10 text-xs outline-none cursor-pointer"
                        >
                            {MODEL_VERSIONS.map((v) => (
                                <option key={v.value} value={v.value} className="bg-white dark:bg-neutral-900">
                                    {v.label}
                                </option>
                            ))}
                        </select>

                        <NaiveButton
                            className="px-5 py-2 rounded-xl bg-black dark:bg-white text-white dark:text-black text-xs font-semibold hover:opacity-90 disabled:opacity-50"
                            on_click={handle_load_model}
                            disabled={is_loading_model}
                        >
                            {is_loading_model ? (
                                <span className="flex items-center gap-1.5">
                                    <RefreshIcon width={14} height={14} className="animate-spin" />
                                    Loading...
                                </span>
                            ) : (
                                "Load Model"
                            )}
                        </NaiveButton>
                    </div>

                    {model_msg && (
                        <div
                            className={`p-2.5 rounded-xl text-xs ${
                                model_msg.startsWith("Error")
                                    ? "bg-rose-500/10 text-rose-500"
                                    : "bg-emerald-500/10 text-emerald-500"
                            }`}
                        >
                            {model_msg}
                        </div>
                    )}
                </div>
            </div>

            {/* LoRA Management */}
            <div className="p-5 rounded-3xl bg-black/5 dark:bg-white/5 border border-black/10 dark:border-white/10 flex flex-col gap-4">
                <div>
                    <h3 className="text-sm font-semibold">LoRA Adapters</h3>
                    <p className="text-xs text-black/50 dark:text-white/50">
                        Attach and manage lightweight fine-tuning weights
                    </p>
                </div>

                <div className="flex flex-col gap-3">
                    <StringInput
                        value={lora_url}
                        on_change={set_lora_url}
                        placeholder="LoRA download URL or local file path"
                        className="py-2.5 px-3 bg-white dark:bg-[#1a1a1a] border border-black/10 dark:border-white/10 rounded-xl text-xs"
                    />

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                        <StringInput
                            value={lora_name}
                            on_change={set_lora_name}
                            placeholder="Adapter identifier name"
                            className="py-2 px-3 bg-white dark:bg-[#1a1a1a] border border-black/10 dark:border-white/10 rounded-xl text-xs"
                        />

                        <div className="flex items-center gap-2 px-3 py-2 bg-white dark:bg-[#1a1a1a] border border-black/10 dark:border-white/10 rounded-xl">
                            <span className="text-[11px] text-black/50 dark:text-white/50">Scale:</span>
                            <input
                                type="range"
                                min={0}
                                max={1}
                                step={0.05}
                                value={lora_strength}
                                onChange={(e) => set_lora_strength(Number(e.target.value))}
                                className="flex-1 accent-black dark:accent-white cursor-pointer"
                            />
                            <span className="font-mono text-xs">{lora_strength}</span>
                        </div>

                        <NaiveButton
                            className="py-2 rounded-xl bg-black dark:bg-white text-white dark:text-black text-xs font-semibold hover:opacity-90 disabled:opacity-50"
                            on_click={handle_load_lora}
                            disabled={is_loading_lora}
                        >
                            {is_loading_lora ? "Loading..." : "Add LoRA"}
                        </NaiveButton>
                    </div>
                </div>

                {/* Active LoRA List */}
                <div className="mt-2 flex flex-col gap-1.5">
                    <span className="text-[11px] font-semibold text-black/50 dark:text-white/50 uppercase tracking-wider">
                        Active Adapters ({active_loras.length})
                    </span>
                    {active_loras.length === 0 ? (
                        <div className="text-center py-4 text-xs text-black/30 dark:text-white/30 border border-dashed border-black/10 dark:border-white/10 rounded-2xl">
                            No LoRA adapters currently loaded
                        </div>
                    ) : (
                        active_loras.map(([name, weight]) => (
                            <div
                                key={name}
                                className="flex items-center justify-between p-3 rounded-xl bg-white dark:bg-[#1a1a1a] border border-black/10 dark:border-white/10 text-xs"
                            >
                                <div className="flex items-center gap-2">
                                    <CubeIcon width={16} height={16} className="text-black/40 dark:text-white/40" />
                                    <span className="font-medium">{name}</span>
                                    <span className="font-mono text-[10px] opacity-60">weight: {weight}</span>
                                </div>
                                <NaiveButton
                                    className="p-1 rounded text-black/40 dark:text-white/40 hover:text-rose-500"
                                    on_click={() => handle_delete_lora(name)}
                                    title="Delete adapter"
                                >
                                    <TrashIcon width={14} height={14} />
                                </NaiveButton>
                            </div>
                        ))
                    )}
                </div>
            </div>

            {/* Sampler / Scheduler Selector */}
            <div className="p-5 rounded-3xl bg-black/5 dark:bg-white/5 border border-black/10 dark:border-white/10 flex flex-col gap-3">
                <div>
                    <h3 className="text-sm font-semibold">Active Sampler / Scheduler</h3>
                    <p className="text-xs text-black/50 dark:text-white/50">
                        Choose the numerical solver used for denoising steps
                    </p>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                    {available_schedulers.map((s) => {
                        const active = state.scheduler === s
                        return (
                            <button
                                key={s}
                                type="button"
                                onClick={() => handle_scheduler_change(s)}
                                className={`p-2.5 rounded-xl text-xs font-medium border transition cursor-pointer text-center truncate ${
                                    active
                                        ? "bg-black text-white dark:bg-white dark:text-black border-transparent shadow-sm"
                                        : "bg-white dark:bg-[#1a1a1a] border-black/10 dark:border-white/10 hover:border-black/30 dark:hover:border-white/30"
                                }`}
                                title={s}
                            >
                                {s}
                            </button>
                        )
                    })}
                </div>
            </div>
        </div>
    )
}
