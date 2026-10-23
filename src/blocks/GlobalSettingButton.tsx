"use client"

import { useState, useCallback, useEffect } from "react"
import { NaiveButton, BodyPortal, AnimationContainer, FullscreenModalContainer, StringInput } from "@/components"
import { SettingIcon, CloseIcon } from "@/components/icons"
import { RefreshIcon, TrashIcon } from "./icons"
import { settings_controller, generation_controller } from "@/core/controllers"
import { useKeyedController, useController } from "@/components/hooks"

export type GlobalSettingButtonProps = {
    className?: string
}

export function GlobalSettingButton({ className = "" }: GlobalSettingButtonProps) {
    const api_url = useKeyedController(settings_controller, "api_url", "http://localhost:8000")
    const gen_state = useController(generation_controller, {
        events: ["system_info_change", "history_change"]
    })

    const [is_open, set_is_open] = useState(false)
    const [is_checking, set_is_checking] = useState(false)
    const [connection_status, set_connection_status] = useState<"connected" | "disconnected" | "checking">("checking")

    const handle_open = useCallback(() => set_is_open(true), [])
    const handle_close = useCallback(() => set_is_open(false), [])
    const handle_toggle = useCallback(() => set_is_open((prev) => !prev), [])

    const verify_connection = useCallback(async () => {
        set_is_checking(true)
        set_connection_status("checking")
        const ok = await settings_controller.check_connection()
        set_connection_status(ok ? "connected" : "disconnected")
        if (ok) {
            await generation_controller.refresh_system_info()
        }
        set_is_checking(false)
    }, [])

    useEffect(() => {
        if (is_open) {
            verify_connection()
        }
    }, [is_open, verify_connection])

    const handle_api_change = (new_url: string) => {
        settings_controller.set("api_url", new_url)
    }

    const handle_clear_history = async () => {
        if (window.confirm("Are you sure you want to clear all generation history?")) {
            await generation_controller.clear_history()
        }
    }

    return (
        <>
            <NaiveButton className={className} on_click={handle_toggle} title="Settings">
                <SettingIcon width={22} height={22} />
            </NaiveButton>

            <BodyPortal>
                <AnimationContainer
                    enter_from={{ transform: "translateY(100vh)" }}
                    enter_to={{ transform: "translateY(0)" }}
                    show={is_open}
                    className="fixed inset-0 z-50"
                >
                    <FullscreenModalContainer
                        enable_escape={is_open}
                        on_escape={handle_close}
                        className="bg-white/95 dark:bg-[#111111]/95 backdrop-blur-2xl text-foreground pt-[max(env(safe-area-inset-top),16px)] pb-[max(env(safe-area-inset-bottom),20px)] px-5 overflow-y-auto"
                    >
                        <div className="max-w-xl mx-auto w-full">
                            <div className="flex items-center justify-between pb-4 border-b border-black/10 dark:border-white/10">
                                <div>
                                    <h2 className="text-lg font-semibold">Settings</h2>
                                    <p className="text-xs text-black/50 dark:text-white/50">
                                        Configure backend API & inference endpoint
                                    </p>
                                </div>
                                <NaiveButton
                                    className="size-8 rounded-full border border-black/10 dark:border-white/10 bg-transparent hover:bg-black/5 dark:hover:bg-white/5"
                                    on_click={handle_close}
                                >
                                    <CloseIcon width={16} height={16} />
                                </NaiveButton>
                            </div>

                            {/* API Endpoint Configuration */}
                            <div className="my-5 p-4 rounded-2xl bg-black/5 dark:bg-white/5 border border-black/10 dark:border-white/10">
                                <div className="flex items-center justify-between mb-2">
                                    <label className="text-sm font-medium">Inference API Endpoint</label>
                                    <span
                                        className={`inline-flex items-center gap-1.5 text-xs px-2.5 py-0.5 rounded-full font-medium ${
                                            connection_status === "connected"
                                                ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30"
                                                : connection_status === "checking"
                                                  ? "bg-amber-500/15 text-amber-600 dark:text-amber-400 border border-amber-500/30"
                                                  : "bg-rose-500/15 text-rose-600 dark:text-rose-400 border border-rose-500/30"
                                        }`}
                                    >
                                        <span
                                            className={`size-1.5 rounded-full ${
                                                connection_status === "connected"
                                                    ? "bg-emerald-500"
                                                    : connection_status === "checking"
                                                      ? "bg-amber-500 animate-pulse"
                                                      : "bg-rose-500"
                                            }`}
                                        />
                                        {connection_status === "connected"
                                            ? "Online"
                                            : connection_status === "checking"
                                              ? "Checking..."
                                              : "Offline"}
                                    </span>
                                </div>

                                <div className="flex gap-2">
                                    <StringInput
                                        value={api_url}
                                        on_change={handle_api_change}
                                        placeholder="http://localhost:8000"
                                        className="flex-1 py-2 px-3 bg-white dark:bg-[#1a1a1a] border border-black/10 dark:border-white/10 rounded-xl focus-within:ring-1 focus-within:ring-black/30 dark:focus-within:ring-white/30"
                                    />
                                    <NaiveButton
                                        className="px-3.5 py-2 rounded-xl bg-black dark:bg-white text-white dark:text-black font-medium text-xs hover:opacity-90"
                                        on_click={verify_connection}
                                        disabled={is_checking}
                                    >
                                        <RefreshIcon
                                            width={14}
                                            height={14}
                                            className={`mr-1.5 ${is_checking ? "animate-spin" : ""}`}
                                        />
                                        Test
                                    </NaiveButton>
                                </div>
                                <p className="text-[11px] text-black/40 dark:text-white/40 mt-2">
                                    Provide the HTTP/HTTPS address where your Python XFusion backend is running (e.g.
                                    local GPU or cloud server).
                                </p>
                            </div>

                            {/* GPU & Model Info */}
                            <div className="my-5 p-4 rounded-2xl bg-black/5 dark:bg-white/5 border border-black/10 dark:border-white/10">
                                <h3 className="text-xs font-semibold text-black/50 dark:text-white/50 uppercase tracking-wider mb-3">
                                    Hardware & Model State
                                </h3>
                                <div className="space-y-2 text-xs">
                                    <div className="flex justify-between py-1 border-b border-black/5 dark:border-white/5">
                                        <span className="text-black/60 dark:text-white/60">GPU Count</span>
                                        <span className="font-mono">{gen_state.system_info?.gpu_count ?? 0}</span>
                                    </div>
                                    <div className="flex justify-between py-1 border-b border-black/5 dark:border-white/5">
                                        <span className="text-black/60 dark:text-white/60">Active Model</span>
                                        <span className="font-mono text-right max-w-xs truncate">
                                            {gen_state.system_info?.current_model || "None Loaded"}
                                        </span>
                                    </div>
                                    <div className="flex justify-between py-1 border-b border-black/5 dark:border-white/5">
                                        <span className="text-black/60 dark:text-white/60">Architecture Version</span>
                                        <span className="font-mono uppercase">
                                            {gen_state.system_info?.model_version || "N/A"}
                                        </span>
                                    </div>
                                    {gen_state.system_info?.gpu_names && gen_state.system_info.gpu_names.length > 0 && (
                                        <div className="flex flex-col py-1">
                                            <span className="text-black/60 dark:text-white/60 mb-1">Devices:</span>
                                            {gen_state.system_info.gpu_names.map((name, i) => (
                                                <span key={i} className="font-mono text-[11px] text-black/80 dark:text-white/80">
                                                    [{i}] {name}
                                                </span>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Local Storage & History */}
                            <div className="my-5 p-4 rounded-2xl bg-black/5 dark:bg-white/5 border border-black/10 dark:border-white/10">
                                <h3 className="text-xs font-semibold text-black/50 dark:text-white/50 uppercase tracking-wider mb-3">
                                    Client Storage
                                </h3>
                                <div className="flex items-center justify-between">
                                    <div>
                                        <div className="text-xs font-medium">Generation History</div>
                                        <div className="text-[11px] text-black/50 dark:text-white/50">
                                            {gen_state.history.length} images saved in client database
                                        </div>
                                    </div>
                                    <NaiveButton
                                        className="px-3 py-1.5 rounded-lg border border-rose-500/30 text-rose-600 dark:text-rose-400 hover:bg-rose-500/10 text-xs font-medium"
                                        on_click={handle_clear_history}
                                        disabled={gen_state.history.length === 0}
                                    >
                                        <TrashIcon width={13} height={13} className="mr-1" />
                                        Clear
                                    </NaiveButton>
                                </div>
                            </div>
                        </div>
                    </FullscreenModalContainer>
                </AnimationContainer>
            </BodyPortal>
        </>
    )
}
