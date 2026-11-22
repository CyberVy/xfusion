"use client"

import {
    Box as CubeIcon,
    RefreshCw as RefreshIcon,
    Trash2 as TrashIcon,
    Copy as CopyIcon,
    WandSparkles as WandIcon,
    ChevronDown,
    Check,
    X
} from "lucide-react"
import { useState } from "react"
import type { ReactNode } from "react"
import { Badge, Button, ContextMenu, EmptyState, ListRow, SectionHeader, Select, TextField } from "@/components"
import { model_controller } from "@/core/controllers"
import { useController } from "@/components/hooks"

const MODEL_VERSIONS = [
    { label: "Version: Auto", value: "" },
    { label: "Version: SD 1.5", value: "1.5" },
    { label: "Version: SD 2", value: "2" },
    { label: "Version: SDXL", value: "xl" },
    { label: "Version: Pony", value: "pony" },
    { label: "Version: SD 3", value: "3" },
    { label: "Version: SD 3.5", value: "3.5" },
    { label: "Version: FLUX", value: "flux" },
    { label: "Version: MOSS", value: "moss" }
]

const BACKBONE_MODES = [
    { label: "Backbone: As published", value: "fp16" },
    { label: "Backbone: 4-bit NF4", value: "int4" },
    { label: "Backbone: 8-bit", value: "int8" },
    { label: "Backbone: FP8", value: "fp8" },
    { label: "Backbone: GGUF", value: "gguf" }
]

const TEXT_ENCODER_MODES = [
    { label: "Encoder: Default (Auto)", value: "default" },
    { label: "Encoder: 4-bit NF4", value: "int4" },
    { label: "Encoder: 8-bit", value: "int8" },
    { label: "Encoder: FP8", value: "fp8" },
    { label: "Encoder: Skip (Disabled)", value: "skip" }
]

const DTYPE_MODES = [
    { label: "Dtype: Auto", value: "auto" },
    { label: "Dtype: FP16", value: "fp16" },
    { label: "Dtype: BF16", value: "bf16" }
]

const MEMORY_MODES = [
    { label: "Memory: GPU", value: "gpu" },
    { label: "Memory: Offload", value: "offload" },
    { label: "Memory: Sequential", value: "sequential" }
]

const FIELD_CLASS =
    "rounded-xl bg-black/4 dark:bg-white/6 border border-black/8 dark:border-white/10 focus-within:border-black/30 dark:focus-within:border-white/30 transition"

const SELECT_INDICATOR = <ChevronDown width={14} height={14} className="opacity-55" />
const SELECT_SELECTED_INDICATOR = <Check width={14} height={14} />

function menu_label(text: string) {
    return <span className="min-w-0 truncate text-[13px] font-medium leading-tight">{text}</span>
}

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
        <ContextMenu
            className="w-full"
            sections={[
                {
                    items: [
                        {
                            key: "set_full",
                            label: menu_label("Set weight to 1.0"),
                            icon: <RefreshIcon width={16} height={16} />,
                            on_click: () => {
                                model_controller.set_lora_strength(name, 1.0)
                                set_strength(1.0)
                            }
                        },
                        {
                            key: "set_half",
                            label: menu_label("Set weight to 0.5"),
                            icon: <RefreshIcon width={16} height={16} />,
                            on_click: () => {
                                model_controller.set_lora_strength(name, 0.5)
                                set_strength(0.5)
                            }
                        },
                        {
                            key: "copy_name",
                            label: menu_label("Copy adapter name"),
                            icon: <CopyIcon width={16} height={16} />,
                            on_click: () => navigator.clipboard?.writeText(name)
                        },
                        {
                            key: "delete",
                            label: menu_label("Remove adapter"),
                            icon: <TrashIcon width={16} height={16} />,
                            tone: "destructive",
                            on_click: () => model_controller.delete_lora(name)
                        }
                    ]
                }
            ]}
            close_label="Close"
        >
            <ListRow
                size="sm"
                variant="filled"
                leading={<CubeIcon width={16} height={16} className="text-black/40 dark:text-white/40" />}
                title={name}
                trailing={
                    <div className="flex items-center gap-2">
                        <input
                            type="range"
                            min={0}
                            max={1}
                            step={0.05}
                            value={strength}
                            onChange={(e) => set_strength(Number(e.target.value))}
                            onPointerUp={() => model_controller.set_lora_strength(name, strength)}
                            className="w-24 accent-black dark:accent-white cursor-pointer shrink-0"
                            aria-label={`${name} strength`}
                        />
                        <span className="w-8 text-right font-mono text-xs shrink-0">{strength}</span>
                        <Button
                            variant="ghost"
                            size="icon"
                            shape="pill"
                            onClick={() => model_controller.delete_lora(name)}
                            title="Remove adapter"
                            aria-label={`Remove ${name}`}
                        >
                            <TrashIcon width={15} height={15} />
                        </Button>
                    </div>
                }
            />
        </ContextMenu>
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
    const [model_version, set_model_version] = useState("")
    const [dtype, set_dtype] = useState("auto")
    const [backbone, set_backbone] = useState("fp16")
    const [text_encoder, set_text_encoder] = useState("default")
    const [memory, set_memory] = useState("gpu")
    const [delete_file_after_load, set_delete_file_after_load] = useState(false)
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
    const model_loaded = Boolean(state.system_info?.model_version)
    const adapters = state.system_info?.adapters
    const supports_lora = !model_loaded || Boolean(adapters?.includes("lora"))
    const supports_controlnet = !model_loaded || Boolean(adapters?.includes("controlnet"))
    const supports_ip_adapter = !model_loaded || Boolean(adapters?.includes("ip_adapter"))
    const ip_adapter = state.system_info?.ip_adapter

    return (
        <div className="w-full max-w-2xl mx-auto px-4 md:px-6 pt-2 pb-32">
            <Panel
                title="Base model"
                description="Hugging Face repo, local path, or download URL. Version can stay on Auto. Dtype, backbone, and text encoder precisions can be configured independently, including 4-bit NF4, FP8, and GGUF. MOSS-Image uses OpenMOSS-Team/MOSS-Image-SFT."
            >
                <ContextMenu
                    className="w-full"
                    sections={
                        state.system_info?.current_model
                            ? [
                                  {
                                      items: [
                                          {
                                              key: "copy_model_name",
                                              label: menu_label("Copy model repo/path"),
                                              icon: <CopyIcon width={16} height={16} />,
                                              on_click: () =>
                                                  navigator.clipboard?.writeText(state.system_info?.current_model || "")
                                          },
                                          {
                                              key: "paste_to_input",
                                              label: menu_label("Fill into model input"),
                                              icon: <WandIcon width={16} height={16} />,
                                              on_click: () => set_model_url(state.system_info?.current_model || "")
                                          }
                                      ]
                                  }
                              ]
                            : []
                    }
                    close_label="Close"
                >
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-4 rounded-2xl bg-black/4 dark:bg-white/6">
                        <div className="min-w-0 flex-1">
                            <p className="text-xs text-black/50 dark:text-white/50">Currently loaded</p>
                            <p className="text-sm font-mono truncate">{state.system_info?.current_model || "None"}</p>
                        </div>
                        <div className="flex flex-wrap items-center gap-1.5 shrink-0">
                            {state.system_info?.dtype && state.system_info.dtype !== "auto" && (
                                <Badge tone="subtle" mono className="px-2 py-1 text-xs uppercase">
                                    {state.system_info.dtype}
                                </Badge>
                            )}
                            {state.system_info?.backbone && state.system_info.backbone !== "fp16" && (
                                <Badge tone="subtle" mono className="px-2 py-1 text-xs uppercase">
                                    {state.system_info.text_encoder && state.system_info.text_encoder !== "default"
                                        ? `B:${state.system_info.backbone}`
                                        : state.system_info.backbone}
                                </Badge>
                            )}
                            {state.system_info?.text_encoder && state.system_info.text_encoder !== "default" && (
                                <Badge tone="subtle" mono className="px-2 py-1 text-xs uppercase">
                                    {`TE:${state.system_info.text_encoder}`}
                                </Badge>
                            )}
                            {state.system_info?.memory && state.system_info.memory !== "gpu" && (
                                <Badge tone="subtle" mono className="px-2 py-1 text-xs uppercase">
                                    {state.system_info.memory}
                                </Badge>
                            )}
                            <Badge tone="subtle" mono className="px-2 py-1 text-xs uppercase">
                                {state.system_info?.model_version || "N/A"}
                            </Badge>
                        </div>
                    </div>
                </ContextMenu>

                <TextField
                    clear_label="Clear"
                    clear_icon={<X width={12} height={12} />}
                    value={model_url}
                    on_change={set_model_url}
                    placeholder={model_version === "moss" ? "eramth/MOSS-Image-SFT-4bit" : "e.g. eramth/realism-sdxl"}
                    className={`h-11 ${FIELD_CLASS}`}
                />

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                    <Select
                        aria_label="Model version hint"
                        align="left"
                        className="w-full"
                        trigger_class_name="w-full min-w-0 h-10 px-3"
                        value={model_version}
                        on_change={set_model_version}
                        indicator={SELECT_INDICATOR}
                        selected_indicator={SELECT_SELECTED_INDICATOR}
                        options={MODEL_VERSIONS}
                    />

                    <Select
                        aria_label="Compute and weight dtype"
                        align="left"
                        className="w-full"
                        trigger_class_name="w-full min-w-0 h-10 px-3"
                        value={dtype}
                        on_change={set_dtype}
                        indicator={SELECT_INDICATOR}
                        selected_indicator={SELECT_SELECTED_INDICATOR}
                        options={DTYPE_MODES}
                    />

                    <Select
                        aria_label="VRAM offload mode"
                        align="right"
                        className="w-full"
                        trigger_class_name="w-full min-w-0 h-10 px-3"
                        value={memory}
                        on_change={set_memory}
                        indicator={SELECT_INDICATOR}
                        selected_indicator={SELECT_SELECTED_INDICATOR}
                        options={MEMORY_MODES}
                    />
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    <Select
                        aria_label="Backbone precision"
                        align="left"
                        className="w-full"
                        trigger_class_name="w-full min-w-0 h-10 px-3"
                        value={backbone}
                        on_change={set_backbone}
                        indicator={SELECT_INDICATOR}
                        selected_indicator={SELECT_SELECTED_INDICATOR}
                        options={BACKBONE_MODES}
                    />

                    <Select
                        aria_label="Text encoder precision"
                        align="right"
                        className="w-full"
                        trigger_class_name="w-full min-w-0 h-10 px-3"
                        value={text_encoder}
                        on_change={set_text_encoder}
                        indicator={SELECT_INDICATOR}
                        selected_indicator={SELECT_SELECTED_INDICATOR}
                        options={TEXT_ENCODER_MODES}
                    />
                </div>

                <div className="flex flex-col gap-2">
                    <label className="flex items-center gap-2 text-xs text-black/60 dark:text-white/60 cursor-pointer select-none">
                        <input
                            type="checkbox"
                            checked={delete_file_after_load}
                            onChange={(e) => set_delete_file_after_load(e.target.checked)}
                            className="accent-black dark:accent-white cursor-pointer"
                        />
                        Purge downloaded archive after loading into memory
                    </label>
                </div>

                <Button
                    variant="primary"
                    size="auto"
                    className="h-11 self-start px-5 text-sm font-semibold"
                    onClick={() =>
                        model_url.trim() &&
                        model_controller.load_model(model_url, model_version, {
                            backbone,
                            text_encoder,
                            dtype,
                            memory,
                            delete_file_after_load
                        })
                    }
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
                </Button>
                <ErrorText message={state.model_error} />
            </Panel>

            {supports_lora && (
                <Panel title="LoRA adapters" description="Drag a loaded adapter's strength, then release to apply it">
                    <TextField
                        clear_label="Clear"
                        clear_icon={<X width={12} height={12} />}
                        value={lora_url}
                        on_change={set_lora_url}
                        placeholder="LoRA URL or local path"
                        className={`h-11 ${FIELD_CLASS}`}
                    />

                    <div className="grid grid-cols-1 sm:grid-cols-[1fr_1fr_auto] gap-2">
                        <TextField
                            clear_label="Clear"
                            clear_icon={<X width={12} height={12} />}
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

                        <Button
                            variant="primary"
                            size="auto"
                            className="h-11 px-5 text-sm font-semibold"
                            onClick={handle_load_lora}
                            disabled={state.is_loading_lora || !lora_url.trim() || !lora_name.trim()}
                        >
                            {state.is_loading_lora ? "Loading" : "Add"}
                        </Button>
                    </div>
                    <ErrorText message={state.lora_error} />

                    <div className="flex flex-col gap-1.5">
                        <div className="flex items-center justify-between gap-2">
                            <SectionHeader title="Active" count={active_loras.length} />
                            <div className="flex gap-1.5">
                                <Button
                                    variant="secondary"
                                    size="sm"
                                    shape="pill"
                                    className="h-8 font-medium"
                                    onClick={() => model_controller.set_loras_enabled(true)}
                                    disabled={active_loras.length === 0 || loras_enabled}
                                >
                                    Enable all
                                </Button>
                                <Button
                                    variant="secondary"
                                    size="sm"
                                    shape="pill"
                                    className="h-8 font-medium"
                                    onClick={() => model_controller.set_loras_enabled(false)}
                                    disabled={active_loras.length === 0 || !loras_enabled}
                                >
                                    Disable all
                                </Button>
                            </div>
                        </div>
                        {active_loras.length === 0 ? (
                            <EmptyState
                                title="No LoRA adapters loaded"
                                className="py-6! border border-dashed border-black/12 dark:border-white/12 rounded-2xl"
                            />
                        ) : (
                            active_loras.map(([name, weight]) => (
                                <LoraAdapterRow key={name} name={name} weight={weight} />
                            ))
                        )}
                    </div>
                </Panel>
            )}

            {supports_ip_adapter && (
                <Panel
                    title="IP-Adapter"
                    description="Reference image encoder. Use it from the Create page once loaded"
                >
                    {ip_adapter ? (
                        <div className="flex flex-col gap-3">
                            <ListRow
                                size="sm"
                                variant="filled"
                                title={<span className="font-mono">{ip_adapter.weight}</span>}
                                subtitle={ip_adapter.source}
                                className="rounded-2xl px-3.5 py-3"
                            />
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
                            <Button
                                variant="secondary"
                                size="auto"
                                className="h-10 px-4 text-sm font-medium"
                                onClick={() => model_controller.delete_ip_adapter()}
                            >
                                Remove IP-Adapter
                            </Button>
                        </div>
                    ) : (
                        <div className="flex flex-col gap-2">
                            <TextField
                                clear_label="Clear"
                                clear_icon={<X width={12} height={12} />}
                                value={ip_uri}
                                on_change={set_ip_uri}
                                placeholder="Repo, e.g. h94/IP-Adapter"
                                className={`h-11 ${FIELD_CLASS}`}
                            />
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                                <TextField
                                    clear_label="Clear"
                                    clear_icon={<X width={12} height={12} />}
                                    value={ip_subfolder}
                                    on_change={set_ip_subfolder}
                                    placeholder="Subfolder"
                                    className={`h-11 ${FIELD_CLASS}`}
                                />
                                <TextField
                                    clear_label="Clear"
                                    clear_icon={<X width={12} height={12} />}
                                    value={ip_name}
                                    on_change={set_ip_name}
                                    placeholder="Weight file name"
                                    className={`h-11 ${FIELD_CLASS}`}
                                />
                            </div>
                            <TextField
                                clear_label="Clear"
                                clear_icon={<X width={12} height={12} />}
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
                            <Button
                                variant="primary"
                                size="auto"
                                className="h-11 px-5 text-sm font-semibold"
                                onClick={() =>
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
                            </Button>
                        </div>
                    )}
                    <ErrorText message={state.ip_adapter_error} />
                </Panel>
            )}

            {supports_controlnet && (
                <Panel
                    title="ControlNet"
                    description="Leave the model empty to use the built-in default. Offload before switching models"
                >
                    <ListRow
                        size="sm"
                        variant="filled"
                        title={state.system_info?.controlnet_loaded ? "Loaded" : "Not loaded"}
                        trailing={
                            <span className="text-xs font-mono text-black/50 dark:text-white/50 truncate max-w-48">
                                {controlnet_model || "default"}
                            </span>
                        }
                        className="rounded-2xl px-3.5 py-3"
                    />
                    <TextField
                        clear_label="Clear"
                        clear_icon={<X width={12} height={12} />}
                        value={controlnet_model}
                        on_change={set_controlnet_model}
                        placeholder="ControlNet URL or repo (optional)"
                        className={`h-11 ${FIELD_CLASS}`}
                    />
                    <div className="flex gap-2">
                        <Button
                            variant="primary"
                            size="auto"
                            className="flex-1 h-11 px-4 text-sm font-semibold"
                            onClick={() => model_controller.load_controlnet(controlnet_model)}
                            disabled={state.is_loading_controlnet}
                        >
                            {state.is_loading_controlnet ? "Working" : "Load"}
                        </Button>
                        <Button
                            variant="secondary"
                            size="auto"
                            className="flex-1 h-11 px-4 text-sm font-medium"
                            onClick={() => model_controller.offload_controlnet()}
                            disabled={state.is_loading_controlnet || !state.system_info?.controlnet_loaded}
                        >
                            Offload
                        </Button>
                    </div>
                    <ErrorText message={state.controlnet_error} />
                </Panel>
            )}
        </div>
    )
}
