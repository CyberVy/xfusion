"use client"

import { NaiveButton } from "@/components"
import { useController } from "@/components/hooks"
import { generation_controller } from "@/core/controllers"
import { DownloadIcon, RefreshIcon, WandIcon } from "./icons"

const RECENT_RESULT_COUNT = 24

export function ResultPanel() {
    const state = useController(generation_controller, { events: ["change"] })
    const selected = state.selected_image
    const recent_items = state.history.slice(0, RECENT_RESULT_COUNT)

    return (
        <div className="flex flex-col gap-3">
            <div className="relative flex items-center justify-center min-h-[40vh] md:min-h-[60vh] rounded-3xl bg-black/4 dark:bg-white/5 overflow-hidden">
                {selected ? (
                    <img
                        src={selected.image_base64}
                        alt={selected.prompt}
                        className={`max-w-full max-h-[70vh] object-contain select-none transition-opacity ${
                            state.is_generating ? "opacity-30" : ""
                        }`}
                    />
                ) : (
                    !state.is_generating && (
                        <div className="flex flex-col items-center gap-2 text-black/35 dark:text-white/35">
                            <WandIcon width={36} height={36} />
                            <span className="text-sm">Results will appear here</span>
                        </div>
                    )
                )}

                {state.is_generating && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-black/60 dark:text-white/60">
                        <RefreshIcon width={28} height={28} className="animate-spin" />
                        <span className="text-sm text-center px-6">
                            {state.progress?.active
                                ? `Step ${state.progress.step} / ${state.progress.total_steps} · image ${state.progress.image_index + 1} / ${state.progress.image_count}`
                                : "Generating..."}
                        </span>
                    </div>
                )}
            </div>

            {selected && (
                <div className="flex items-start justify-between gap-4 px-1">
                    <div className="min-w-0 flex flex-col gap-1">
                        <p className="text-sm line-clamp-2">{selected.prompt || "(no prompt)"}</p>
                        <p className="text-xs font-mono text-black/50 dark:text-white/50">
                            {selected.type} · {selected.width}×{selected.height} · seed {selected.seed}
                            {selected.elapsed_seconds !== undefined && ` · ${selected.elapsed_seconds}s`}
                        </p>
                    </div>
                    <div className="flex shrink-0 gap-2">
                        <NaiveButton
                            className="h-9 px-3.5 rounded-full text-xs font-medium bg-black/5 dark:bg-white/8 hover:bg-black/10 dark:hover:bg-white/12"
                            on_click={() => generation_controller.reuse_settings(selected)}
                            title="Copy prompt, seed and size into the editor"
                        >
                            Reuse
                        </NaiveButton>
                        <a
                            href={selected.image_base64}
                            download={`xfusion_${selected.type}_${selected.seed}.png`}
                            className="size-9 rounded-full flex items-center justify-center bg-black text-white dark:bg-white dark:text-black hover:opacity-85 transition"
                            title="Download"
                        >
                            <DownloadIcon width={16} height={16} />
                        </a>
                    </div>
                </div>
            )}

            {recent_items.length > 1 && (
                <div className="flex gap-2 overflow-x-auto no-scrollbar py-1 px-1">
                    {recent_items.map((item) => (
                        <button
                            key={item.id}
                            type="button"
                            onClick={() => generation_controller.select_image(item)}
                            className={`shrink-0 size-16 rounded-xl overflow-hidden cursor-pointer transition ${
                                item.id === selected?.id
                                    ? "ring-2 ring-black dark:ring-white"
                                    : "opacity-60 hover:opacity-100"
                            }`}
                        >
                            <img
                                src={item.image_base64}
                                alt={item.prompt}
                                className="w-full h-full object-cover"
                                loading="lazy"
                            />
                        </button>
                    ))}
                </div>
            )}
        </div>
    )
}
