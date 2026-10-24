"use client"

import { useState } from "react"
import type { ReactNode } from "react"
import { NaiveButton, StringInput } from "@/components"
import { CubeIcon, RefreshIcon, TrashIcon } from "./icons"
import { model_controller } from "@/core/controllers"
import { useController } from "@/components/hooks"

const MODEL_VERSIONS = [
    { label: "Auto", value: "" },
    { label: "SD 1.5", value: "1.5" },
    { label: "SD 2", value: "2" },
    { label: "SDXL", value: "xl" },
    { label: "Pony", value: "pony" },
    { label: "SD 3", value: "3" },
    { label: "SD 3.5", value: "3.5" },
    { label: "FLUX", value: "flux" }
]

const FIELD_CLASS =
    "rounded-xl bg-black/4 dark:bg-white/6 border border-black/8 dark:border-white/10 focus-within:border-black/30 dark:focus-within:border-white/30 transition"

function Panel({ title, description, children }: { title: string; description: string; children: ReactNode }) {
    return (
        <section className="flex flex-col gap-4 py-6 border-b border-black/8 dark:border-white/10 last:border-b-0">
            <div>
                <h3 className="text-base font-semibold">{title}</h3>
                <p className="text-sm text-black/50 dark:text-white/50">{description}</p>
            </div>
            {children}
        </section>
    )
}

function LoraAdapterRow({ name, weight }: { name: string; weight: number }) {
    const [strength, set_strength] = useState(weight)
    const [prev_weight, set_prev_weight] = useState(weight)
    if (weight !== prev_weight) {
        set_prev_weight(weight)
        set_strength(weight)
    }

    return (
        <div className="flex items-center gap-3 px-3.5 py-2.5 rounded-xl bg-black/4 dark:bg-white/6">
            <CubeIcon width={16} height={16} className="shrink-0 text-black/40 dark:text-white/40" />
            <span className="text-sm font-medium truncate min-w-0 flex-1">{name}</span>
            <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={strength}
                onChange={(e) => set_strength(Number(e.target.value))}
                onPointerUp={() => model_controller.set_lora_strength(name, strength)}
                className="w-24 accent-black dark:accent-white cursor-pointer"
                aria-label={`${name} strength`}
            />
            <span className="w-8 text-right font-mono text-xs">{strength}</span>
            <NaiveButton
                className="size-8 rounded-full text-black/45 dark:text-white/45 hover:text-black dark:hover:text-white hover:bg-black/5 dark:hover:bg-white/8"
                on_click={() => model_controller.delete_lora(name)}
                title="Remove adapter"
            >
                <TrashIcon width={15} height={15} />
            </NaiveButton>
        </div>
    )
}

function ErrorText({ message }: { message: string | null }) {
    if (!message) return null
    return (
        <p className="p-3 rounded-xl text-xs leading-relaxed bg-black/5 dark:bg-white/8 border border-black/15 dark:border-white/20 wrap-break-word">
            {message}
        </p>
    )
}

export function ModelManageView() {
    const state = useController(model_controller, { events: ["change"] })

    const [model_url, set_model_url] = useState("eramth/realism-sdxl")
    const [model_version, set_model_version] = useState("xl")
    const [lora_url, set_lora_url] = useState("")
    const [lora_name, set_lora_name] = useState("")
    const [lora_strength, set_lora_strength] = useState(0.4)
    const [ip_uri, set_ip_uri] = useState("h94/IP-Adapter")
    const [ip_subfolder, set_ip_subfolder] = useState("")
    const [ip_name, set_ip_name] = useState("")
    const [ip_encoder, set_ip_encoder] = useState("image_encoder")
    const [ip_strength, set_ip_strength] = useState(0.5)
    const [controlnet_model, set_controlnet_model] = useState("")

    const handle_load_lora = async () => {
        if (!lora_url.trim() || !lora_name.trim()) return
        if (await model_controller.load_lora(lora_url, lora_name, lora_strength)) {
            set_lora_url("")
            set_lora_name("")
        }
    }

    const active_loras = state.system_info?.loras ? Object.entries(state.system_info.loras) : []
    const loras_enabled = state.system_info?.loras_enabled !== false
    const is_flux = state.system_info?.model_version === "flux"
    const ip_adapter = state.system_info?.ip_adapter

    return (
        <div className="w-full max-w-2xl mx-auto px-4 md:px-6 pt-2 pb-32">
            <Panel title="Base model" description="Hugging Face repo ID, local path, or download URL">
                <div className="flex items-center justify-between gap-3 p-3.5 rounded-2xl bg-black/4 dark:bg-white/6">
                    <div className="min-w-0">
                        <p className="text-xs text-black/50 dark:text-white/50">Currently loaded</p>
                        <p className="text-sm font-mono truncate">{state.system_info?.current_model || "None"}</p>
                    </div>
                    <span className="shrink-0 text-xs uppercase font-mono px-2 py-1 rounded-md bg-black/8 dark:bg-white/10">
                        {state.system_info?.model_version || "N/A"}
                    </span>
                </div>

                <StringInput
                    value={model_url}
                    on_change={set_model_url}
                    placeholder="e.g. eramth/realism-sdxl"
                    className={`h-11 ${FIELD_CLASS}`}
                />

                <div className="flex gap-2">
                    <select
                        value={model_version}
                        onChange={(e) => set_model_version(e.target.value)}
                        className={`flex-1 h-11 px-3 text-sm outline-none cursor-pointer ${FIELD_CLASS}`}
                    >
                        {MODEL_VERSIONS.map((v) => (
                            <option key={v.value} value={v.value}>
                                {v.label}
                            </option>
                        ))}
                    </select>

                    <NaiveButton
                        className="h-11 px-5 rounded-xl bg-black text-white dark:bg-white dark:text-black text-sm font-semibold hover:opacity-90 disabled:opacity-40"
                        on_click={() => model_url.trim() && model_controller.load_model(model_url, model_version)}
                        disabled={state.is_loading_model}
                    >
                        {state.is_loading_model ? (
                            <span className="flex items-center gap-2">
                                <RefreshIcon width={14} height={14} className="animate-spin" />
                                Loading
                            </span>
                        ) : (
                            "Load"
                        )}
                    </NaiveButton>
                </div>
                <ErrorText message={state.model_error} />
            </Panel>

            <Panel title="LoRA adapters" description="Drag a loaded adapter's strength, then release to apply it">
                <StringInput
                    value={lora_url}
                    on_change={set_lora_url}
                    placeholder="LoRA URL or local path"
                    className={`h-11 ${FIELD_CLASS}`}
                />

                <div className="grid grid-cols-1 sm:grid-cols-[1fr_1fr_auto] gap-2">
                    <StringInput
                        value={lora_name}
                        on_change={set_lora_name}
                        placeholder="Adapter name"
                        className={`h-11 ${FIELD_CLASS}`}
                    />

                    <label className={`h-11 flex items-center gap-2 px-3 ${FIELD_CLASS}`}>
                        <span className="text-xs text-black/50 dark:text-white/50">Strength</span>
                        <input
                            type="range"
                            min={0}
                            max={1}
                            step={0.05}
                            value={lora_strength}
                            onChange={(e) => set_lora_strength(Number(e.target.value))}
                            className="flex-1 min-w-0 accent-black dark:accent-white cursor-pointer"
                        />
                        <span className="w-8 text-right font-mono text-xs">{lora_strength}</span>
                    </label>

                    <NaiveButton
                        className="h-11 px-5 rounded-xl bg-black text-white dark:bg-white dark:text-black text-sm font-semibold hover:opacity-90 disabled:opacity-40"
                        on_click={handle_load_lora}
                        disabled={state.is_loading_lora || !lora_url.trim() || !lora_name.trim()}
                    >
                        {state.is_loading_lora ? "Loading" : "Add"}
                    </NaiveButton>
                </div>
                <ErrorText message={state.lora_error} />

                <div className="flex flex-col gap-1.5">
                    <div className="flex items-center justify-between gap-2">
                        <span className="text-xs font-medium text-black/50 dark:text-white/50">
                            Active ({active_loras.length})
                        </span>
                        <div className="flex gap-1.5">
                            <NaiveButton
                                className="h-8 px-3 rounded-full text-xs font-medium bg-black/5 dark:bg-white/8 hover:bg-black/10 dark:hover:bg-white/12 disabled:opacity-40"
                                on_click={() => model_controller.set_loras_enabled(true)}
                                disabled={active_loras.length === 0 || loras_enabled}
                            >
                                Enable all
                            </NaiveButton>
                            <NaiveButton
                                className="h-8 px-3 rounded-full text-xs font-medium bg-black/5 dark:bg-white/8 hover:bg-black/10 dark:hover:bg-white/12 disabled:opacity-40"
                                on_click={() => model_controller.set_loras_enabled(false)}
                                disabled={active_loras.length === 0 || !loras_enabled}
                            >
                                Disable all
                            </NaiveButton>
                        </div>
                    </div>
                    {active_loras.length === 0 ? (
                        <div className="py-6 text-center text-sm text-black/35 dark:text-white/35 border border-dashed border-black/12 dark:border-white/12 rounded-2xl">
                            No LoRA adapters loaded
                        </div>
                    ) : (
                        active_loras.map(([name, weight]) => <LoraAdapterRow key={name} name={name} weight={weight} />)
                    )}
                </div>
            </Panel>

            {!is_flux && (
                <Panel
                    title="IP-Adapter"
                    description="Reference image encoder. Use it from the Create page once loaded"
                >
                    {ip_adapter ? (
                        <div className="flex flex-col gap-3">
                            <div className="p-3.5 rounded-2xl bg-black/4 dark:bg-white/6 text-sm">
                                <p className="font-mono truncate">{ip_adapter.weight_name}</p>
                                <p className="text-xs text-black/50 dark:text-white/50 truncate">{ip_adapter.uri}</p>
                            </div>
                            <label className={`h-11 flex items-center gap-2 px-3 ${FIELD_CLASS}`}>
                                <span className="text-xs text-black/50 dark:text-white/50">Strength</span>
                                <input
                                    type="range"
                                    min={0}
                                    max={1}
                                    step={0.05}
                                    defaultValue={ip_adapter.strength}
                                    key={ip_adapter.strength}
                                    onPointerUp={(e) =>
                                        model_controller.set_ip_adapter_strength(Number(e.currentTarget.value))
                                    }
                                    className="flex-1 accent-black dark:accent-white cursor-pointer"
                                />
                                <span className="w-8 text-right font-mono text-xs">{ip_adapter.strength}</span>
                            </label>
                            <NaiveButton
                                className="h-10 rounded-xl text-sm font-medium bg-black/5 dark:bg-white/8 hover:bg-black/10 dark:hover:bg-white/12"
                                on_click={() => model_controller.delete_ip_adapter()}
                            >
                                Remove IP-Adapter
                            </NaiveButton>
                        </div>
                    ) : (
                        <div className="flex flex-col gap-2">
                            <StringInput
                                value={ip_uri}
                                on_change={set_ip_uri}
                                placeholder="Repo, e.g. h94/IP-Adapter"
                                className={`h-11 ${FIELD_CLASS}`}
                            />
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                                <StringInput
                                    value={ip_subfolder}
                                    on_change={set_ip_subfolder}
                                    placeholder="Subfolder"
                                    className={`h-11 ${FIELD_CLASS}`}
                                />
                                <StringInput
                                    value={ip_name}
                                    on_change={set_ip_name}
                                    placeholder="Weight file name"
                                    className={`h-11 ${FIELD_CLASS}`}
                                />
                            </div>
                            <StringInput
                                value={ip_encoder}
                                on_change={set_ip_encoder}
                                placeholder="Image encoder folder"
                                className={`h-11 ${FIELD_CLASS}`}
                            />
                            <label className={`h-11 flex items-center gap-2 px-3 ${FIELD_CLASS}`}>
                                <span className="text-xs text-black/50 dark:text-white/50">Strength</span>
                                <input
                                    type="range"
                                    min={0}
                                    max={1}
                                    step={0.05}
                                    value={ip_strength}
                                    onChange={(e) => set_ip_strength(Number(e.target.value))}
                                    className="flex-1 accent-black dark:accent-white cursor-pointer"
                                />
                                <span className="w-8 text-right font-mono text-xs">{ip_strength}</span>
                            </label>
                            <NaiveButton
                                className="h-11 rounded-xl bg-black text-white dark:bg-white dark:text-black text-sm font-semibold hover:opacity-90 disabled:opacity-40"
                                on_click={() =>
                                    ip_uri.trim() &&
                                    ip_name.trim() &&
                                    model_controller.load_ip_adapter(
                                        ip_uri,
                                        ip_subfolder,
                                        ip_name,
                                        ip_encoder,
                                        ip_strength
                                    )
                                }
                                disabled={state.is_loading_ip_adapter || !ip_uri.trim() || !ip_name.trim()}
                            >
                                {state.is_loading_ip_adapter ? "Loading" : "Load IP-Adapter"}
                            </NaiveButton>
                        </div>
                    )}
                    <ErrorText message={state.ip_adapter_error} />
                </Panel>
            )}

            <Panel
                title="ControlNet"
                description="Leave the model empty to use the built-in default. Offload before switching models"
            >
                <div className="flex items-center justify-between gap-3 p-3.5 rounded-2xl bg-black/4 dark:bg-white/6">
                    <span className="text-sm">{state.system_info?.controlnet_loaded ? "Loaded" : "Not loaded"}</span>
                    <span className="text-xs font-mono text-black/50 dark:text-white/50 truncate">
                        {controlnet_model || "default"}
                    </span>
                </div>
                <StringInput
                    value={controlnet_model}
                    on_change={set_controlnet_model}
                    placeholder="ControlNet URL or repo (optional)"
                    className={`h-11 ${FIELD_CLASS}`}
                />
                <div className="flex gap-2">
                    <NaiveButton
                        className="flex-1 h-11 rounded-xl bg-black text-white dark:bg-white dark:text-black text-sm font-semibold hover:opacity-90 disabled:opacity-40"
                        on_click={() => model_controller.load_controlnet(controlnet_model)}
                        disabled={state.is_loading_controlnet}
                    >
                        {state.is_loading_controlnet ? "Working" : "Load"}
                    </NaiveButton>
                    <NaiveButton
                        className="flex-1 h-11 rounded-xl text-sm font-medium bg-black/5 dark:bg-white/8 hover:bg-black/10 dark:hover:bg-white/12 disabled:opacity-40"
                        on_click={() => model_controller.offload_controlnet()}
                        disabled={state.is_loading_controlnet || !state.system_info?.controlnet_loaded}
                    >
                        Offload
                    </NaiveButton>
                </div>
                <ErrorText message={state.controlnet_error} />
            </Panel>
        </div>
    )
}
