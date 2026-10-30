"use client"

import { useState } from "react"
import {
    NaiveButton,
    FullscreenModalContainer,
    AnimationContainer,
    BodyPortal,
    view_switcher_controller
} from "@/components"
import { CloseIcon } from "@/components/icons"
import { GalleryIcon, DownloadIcon, TrashIcon, WandIcon } from "./icons"
import { generation_controller } from "@/core/controllers"
import { useController } from "@/components/hooks"
import type { GeneratedImageItem } from "@/core/api/types"

export function GalleryView() {
    const state = useController(generation_controller, {
        events: ["history_change", "change"]
    })

    const [inspected_item, set_inspected_item] = useState<GeneratedImageItem | null>(null)

    const handle_delete = async (e: React.MouseEvent, id: string) => {
        e.stopPropagation()
        await generation_controller.delete_image(id)
        if (inspected_item?.id === id) {
            set_inspected_item(null)
        }
    }

    const handle_reuse_prompt = (item: GeneratedImageItem) => {
        generation_controller.reuse_settings(item)
        generation_controller.select_image(item)
        set_inspected_item(null)
        view_switcher_controller.switch_view("create", "xfusion_switcher")
    }

    return (
        <div className="w-full max-w-6xl mx-auto px-4 py-6 pb-28">
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h2 className="text-lg font-semibold">Artworks Gallery</h2>
                    <p className="text-xs text-black/50 dark:text-white/50">
                        {state.history.length} creations saved in local device storage
                    </p>
                </div>
            </div>

            {state.history.length === 0 ? (
                <div className="flex flex-col items-center justify-center min-h-[50vh] text-center text-black/30 dark:text-white/30 text-xs">
                    <GalleryIcon width={48} height={48} className="mb-2 opacity-40" />
                    <p className="text-sm font-medium">No artworks generated yet</p>
                    <p className="mt-1 max-w-xs">Images you generate are saved on this device and listed here.</p>
                </div>
            ) : (
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                    {state.history.map((item) => (
                        <div
                            key={item.id}
                            onClick={() => set_inspected_item(item)}
                            className="group relative aspect-square rounded-2xl overflow-hidden bg-black/5 dark:bg-white/5 border border-black/10 dark:border-white/10 cursor-pointer hover:shadow-lg transition"
                        >
                            <img
                                src={item.image_base64}
                                alt={item.prompt}
                                className="w-full h-full object-cover group-hover:scale-105 transition duration-300"
                                loading="lazy"
                            />

                            <div className="absolute inset-0 bg-linear-to-t from-black/70 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition p-2.5 flex flex-col justify-between text-white">
                                <div className="flex justify-end">
                                    <button
                                        type="button"
                                        onClick={(e) => handle_delete(e, item.id)}
                                        className="p-1 rounded-full bg-black/40 hover:bg-black/70 transition"
                                        title="Delete"
                                    >
                                        <TrashIcon width={13} height={13} />
                                    </button>
                                </div>
                                <div>
                                    <p className="text-[11px] font-medium line-clamp-2 leading-tight">{item.prompt}</p>
                                    <p className="text-[9px] opacity-70 mt-1 uppercase">
                                        {item.type} • {item.width}×{item.height}
                                    </p>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* Artwork Details Modal */}
            <BodyPortal>
                <AnimationContainer
                    enter_from={{ opacity: 0, transform: "scale(0.96)" }}
                    enter_to={{ opacity: 1, transform: "scale(1)" }}
                    show={!!inspected_item}
                    className="fixed inset-0 z-50 flex items-center justify-center p-4"
                >
                    <FullscreenModalContainer
                        enable_escape={!!inspected_item}
                        on_escape={() => set_inspected_item(null)}
                        className="bg-white/95 dark:bg-[#141414]/95 backdrop-blur-3xl text-foreground p-6 rounded-3xl max-w-3xl w-full max-h-[90vh] overflow-y-auto border border-black/10 dark:border-white/10 shadow-2xl"
                    >
                        {inspected_item && (
                            <div className="flex flex-col md:flex-row gap-6">
                                <div className="md:w-1/2 flex items-center justify-center bg-black/5 dark:bg-white/5 rounded-2xl overflow-hidden aspect-square">
                                    <img
                                        src={inspected_item.image_base64}
                                        alt={inspected_item.prompt}
                                        className="max-w-full max-h-full object-contain"
                                    />
                                </div>

                                <div className="md:w-1/2 flex flex-col justify-between">
                                    <div className="flex flex-col gap-3">
                                        <div className="flex justify-between items-start">
                                            <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-black/10 dark:bg-white/10">
                                                {inspected_item.type}
                                            </span>
                                            <NaiveButton
                                                className="size-7 rounded-full border border-black/10 dark:border-white/10 hover:bg-black/5 dark:hover:bg-white/5"
                                                on_click={() => set_inspected_item(null)}
                                            >
                                                <CloseIcon width={14} height={14} />
                                            </NaiveButton>
                                        </div>

                                        <div>
                                            <h4 className="text-[11px] font-semibold text-black/50 dark:text-white/50 uppercase">
                                                Prompt
                                            </h4>
                                            <p className="text-xs mt-1 max-h-24 overflow-y-auto leading-relaxed">
                                                {inspected_item.prompt}
                                            </p>
                                        </div>

                                        {inspected_item.negative_prompt && (
                                            <div>
                                                <h4 className="text-[11px] font-semibold text-black/50 dark:text-white/50 uppercase">
                                                    Negative Prompt
                                                </h4>
                                                <p className="text-[11px] text-black/60 dark:text-white/60 mt-1 max-h-16 overflow-y-auto">
                                                    {inspected_item.negative_prompt}
                                                </p>
                                            </div>
                                        )}

                                        <div className="grid grid-cols-2 gap-2 text-[11px] p-3 rounded-xl bg-black/5 dark:bg-white/5 font-mono">
                                            <div>Seed: {inspected_item.seed}</div>
                                            <div>
                                                Size: {inspected_item.width}×{inspected_item.height}
                                            </div>
                                            {inspected_item.elapsed_seconds && (
                                                <div>Time: {inspected_item.elapsed_seconds}s</div>
                                            )}
                                        </div>
                                    </div>

                                    <div className="flex gap-2 mt-5">
                                        <NaiveButton
                                            className="flex-1 py-2.5 rounded-xl border border-black/15 dark:border-white/15 text-xs font-semibold hover:bg-black/5 dark:hover:bg-white/5"
                                            on_click={() => handle_reuse_prompt(inspected_item)}
                                        >
                                            <WandIcon width={14} height={14} className="mr-1.5" />
                                            Reuse Prompt
                                        </NaiveButton>
                                        <a
                                            href={inspected_item.image_base64}
                                            download={`xfusion_${inspected_item.seed}.png`}
                                            className="flex-1 py-2.5 rounded-xl bg-black dark:bg-white text-white dark:text-black text-xs font-semibold flex items-center justify-center hover:opacity-90"
                                        >
                                            <DownloadIcon width={14} height={14} className="mr-1.5" />
                                            Download
                                        </a>
                                    </div>
                                </div>
                            </div>
                        )}
                    </FullscreenModalContainer>
                </AnimationContainer>
            </BodyPortal>
        </div>
    )
}
