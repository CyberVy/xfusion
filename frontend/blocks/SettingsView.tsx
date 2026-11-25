"use client"

import { RefreshCw, Trash2, X } from "lucide-react"
import { useState, useCallback, useSyncExternalStore } from "react"
import { Alert, Badge, Button, SectionHeader, Text, TextField } from "@/components"
import { settings_controller, generation_controller, model_controller } from "@/core/controllers"
import { useKeyedController, useController } from "@/components/hooks"
import type { GenerationState, ModelState } from "@/core/controllers"

const select_history_count = (s: GenerationState) => s.history.length
const select_system_info = (s: ModelState) => s.system_info

function subscribe_connection(callback: () => void): () => void {
    settings_controller.addEventListener("set:connection", callback)
    return () => settings_controller.removeEventListener("set:connection", callback)
}

function get_is_connected(): boolean {
    return settings_controller.is_connected
}

function get_false(): boolean {
    return false
}

export function SettingsView() {
    const api_url = useKeyedController(settings_controller, "api_url", "http://localhost:8000")
    const is_connected = useSyncExternalStore(subscribe_connection, get_is_connected, get_false)
    const history_count = useController(generation_controller, {
        events: ["history_change"],
        selector: select_history_count
    })
    const system_info = useController(model_controller, { events: ["change"], selector: select_system_info })

    const [is_checking, set_is_checking] = useState(false)
    const [is_confirming_clear, set_is_confirming_clear] = useState(false)

    const connection_status: "connected" | "disconnected" | "checking" = is_checking
        ? "checking"
        : is_connected
          ? "connected"
          : "disconnected"

    const verify_connection = useCallback(async () => {
        set_is_checking(true)
        try {
            await settings_controller.check_connection()
        } finally {
            set_is_checking(false)
        }
    }, [])

    const handle_api_change = (new_url: string) => {
        settings_controller.set("api_url", new_url)
    }

    return (
        <div className="w-full max-w-xl mx-auto px-4 md:px-6 py-6 pb-32">
            <div className="pb-4 border-b border-black/10 dark:border-white/10">
                <Text as="h2" variant="title" weight="semibold">
                    Settings
                </Text>
                <Text as="p" variant="caption" tone="secondary">
                    Configure backend API & inference endpoint
                </Text>
            </div>

            {/* API Endpoint Configuration */}
            <div className="my-4.5 p-4.5 sm:p-5 rounded-2xl bg-black/4 dark:bg-white/5 border border-black/8 dark:border-white/10">
                <div className="flex items-center justify-between mb-2.5">
                    <Text as="label" variant="body" weight="medium">
                        Inference API Endpoint
                    </Text>
                    <Badge
                        tone="custom"
                        shape="pill"
                        className={`gap-1.5 px-2.5 py-0.5 text-xs border ${
                            connection_status === "connected"
                                ? "border-black/20 dark:border-white/25"
                                : "border-black/10 dark:border-white/10 text-black/50 dark:text-white/50"
                        }`}
                    >
                        <span
                            className={`size-1.5 rounded-full ${
                                connection_status === "connected"
                                    ? "bg-black dark:bg-white"
                                    : connection_status === "checking"
                                      ? "bg-black/40 dark:bg-white/40 animate-pulse"
                                      : "border border-black/40 dark:border-white/40"
                            }`}
                        />
                        {connection_status === "connected"
                            ? "Online"
                            : connection_status === "checking"
                              ? "Checking..."
                              : "Offline"}
                    </Badge>
                </div>

                <div className="flex gap-2.5">
                    <TextField
                        clear_label="Clear"
                        clear_icon={<X width={12} height={12} />}
                        value={api_url}
                        on_change={handle_api_change}
                        placeholder="http://localhost:8000"
                        className="flex-1 bg-white dark:bg-[#1a1a1a] text-sm"
                    />
                    <Button variant="primary" size="md" onClick={verify_connection} disabled={is_checking}>
                        <span className="inline-flex items-center gap-1.5">
                            <RefreshCw width={14} height={14} className={is_checking ? "animate-spin" : ""} />
                            Test
                        </span>
                    </Button>
                </div>
                <Text as="p" variant="footnote" tone="tertiary" className="mt-2.5">
                    Provide the HTTP/HTTPS address where your Python xfusion backend is running (e.g. local GPU or cloud
                    server).
                </Text>
            </div>

            {/* GPU & Model Info */}
            <div className="my-4.5 p-4.5 sm:p-5 rounded-2xl bg-black/4 dark:bg-white/5 border border-black/8 dark:border-white/10">
                <SectionHeader title="Hardware & Model State" className="mb-3" />
                <div className="space-y-1.5">
                    <div className="flex justify-between py-1.5 border-b border-black/5 dark:border-white/5">
                        <Text variant="caption" tone="secondary">
                            GPU Count
                        </Text>
                        <Text variant="caption" mono>
                            {system_info?.gpu_count ?? 0}
                        </Text>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-black/5 dark:border-white/5">
                        <Text variant="caption" tone="secondary">
                            Active Model
                        </Text>
                        <Text variant="caption" mono align="right" truncate className="max-w-xs">
                            {system_info?.current_model || "None Loaded"}
                        </Text>
                    </div>
                    {system_info?.python_version && (
                        <div className="flex justify-between py-1.5 border-b border-black/5 dark:border-white/5">
                            <Text variant="caption" tone="secondary">
                                Python
                            </Text>
                            <Text variant="caption" mono>
                                {system_info.python_version}
                            </Text>
                        </div>
                    )}
                    {system_info?.os_platform && (
                        <div className="flex justify-between gap-4 py-1.5 border-b border-black/5 dark:border-white/5">
                            <Text variant="caption" tone="secondary" className="shrink-0">
                                OS
                            </Text>
                            <Text variant="caption" mono align="right" truncate>
                                {system_info.os_platform}
                            </Text>
                        </div>
                    )}
                    <div className="flex justify-between py-1.5 border-b border-black/5 dark:border-white/5">
                        <Text variant="caption" tone="secondary">
                            Architecture Version
                        </Text>
                        <Text variant="caption" mono className="uppercase">
                            {system_info?.model_version || "N/A"}
                        </Text>
                    </div>
                    {system_info?.gpu_names && system_info.gpu_names.length > 0 && (
                        <div className="flex flex-col py-1.5">
                            <Text variant="caption" tone="secondary" className="mb-1">
                                Devices:
                            </Text>
                            {system_info.gpu_names.map((name, i) => (
                                <Text key={i} variant="footnote" mono className="text-black/80 dark:text-white/80">
                                    [{i}] {name}
                                </Text>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            {/* Local Storage & History */}
            <div className="my-4.5 p-4.5 sm:p-5 rounded-2xl bg-black/4 dark:bg-white/5 border border-black/8 dark:border-white/10">
                <SectionHeader title="Client Storage" className="mb-3" />
                <div className="flex items-center justify-between">
                    <div>
                        <Text as="div" variant="caption" weight="medium">
                            Generation History
                        </Text>
                        <Text as="div" variant="footnote" tone="secondary">
                            {history_count} images saved in client database
                        </Text>
                    </div>
                    <Button
                        variant="outline"
                        size="sm"
                        className="font-medium"
                        onClick={() => set_is_confirming_clear(true)}
                        disabled={history_count === 0}
                    >
                        <span className="inline-flex items-center gap-1">
                            <Trash2 width={13} height={13} />
                            Clear
                        </span>
                    </Button>
                </div>
            </div>

            <Alert
                open={is_confirming_clear}
                on_close={() => set_is_confirming_clear(false)}
                title="Clear generation history"
                message={`Delete all ${history_count} saved images from client storage? This cannot be undone.`}
                close_label="Cancel"
                actions={[
                    { key: "cancel", label: "Cancel" },
                    {
                        key: "clear",
                        label: "Clear",
                        primary: true,
                        on_click: () => {
                            generation_controller.clear_history()
                        }
                    }
                ]}
            />
        </div>
    )
}
