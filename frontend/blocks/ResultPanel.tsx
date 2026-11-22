"use client"

import {
    Download as DownloadIcon,
    RefreshCw as RefreshIcon,
    WandSparkles as WandIcon,
    Copy as CopyIcon,
    Trash2 as TrashIcon,
    Image as ImageIcon
} from "lucide-react"
import { Button, ContextMenu, EmptyState, HorizontalScroller, Image } from "@/components"
import { useController } from "@/components/hooks"
import { generation_controller } from "@/core/controllers"
import type { GeneratedImageItem } from "@/core/api/types"

const RECENT_RESULT_COUNT = 24

function menu_label(text: string) {
    return <span className="min-w-0 truncate text-[13px] font-medium leading-tight">{text}</span>
}

async function copy_image_to_clipboard(base64: string) {
    try {
        const res = await fetch(base64)
        const blob = await res.blob()
        await navigator.clipboard?.write?.([
            new ClipboardItem({
                [blob.type || "image/png"]: blob
            })
        ])
    } catch {
        // Ignore clipboard write failures
    }
}

function download_image_file(item: GeneratedImageItem) {
    const link = document.createElement("a")
    link.href = item.image_base64
    link.download = `xfusion_${item.type}_${item.seed}.png`
    link.click()
}

export function ResultPanel() {
    const state = useController(generation_controller, { events: ["change"] })
    const selected = state.selected_image
    const recent_items = state.history.slice(0, RECENT_RESULT_COUNT)

    return (
        <div className="flex flex-col gap-3.5">
            <div className="relative flex items-center justify-center min-h-[40vh] md:min-h-[60vh] rounded-3xl bg-black/4 dark:bg-white/5 p-3 sm:p-4 overflow-hidden">
                {selected ? (
                    <ContextMenu
                        className="w-full h-full flex items-center justify-center"
                        sections={[
                            {
                                items: [
                                    {
                                        key: "reuse",
                                        label: menu_label("Reuse parameters"),
                                        icon: <WandIcon width={16} height={16} />,
                                        on_click: () => generation_controller.reuse_settings(selected)
                                    },
                                    {
                                        key: "copy_prompt",
                                        label: menu_label("Copy prompt"),
                                        icon: <CopyIcon width={16} height={16} />,
                                        disabled: !selected.prompt,
                                        on_click: () => {
                                            if (selected.prompt) {
                                                navigator.clipboard?.writeText(selected.prompt)
                                            }
                                        }
                                    },
                                    {
                                        key: "copy_image",
                                        label: menu_label("Copy image"),
                                        icon: <ImageIcon width={16} height={16} />,
                                        on_click: () => copy_image_to_clipboard(selected.image_base64)
                                    },
                                    {
                                        key: "download",
                                        label: menu_label("Download"),
                                        icon: <DownloadIcon width={16} height={16} />,
                                        on_click: () => download_image_file(selected)
                                    },
                                    {
                                        key: "delete",
                                        label: menu_label("Delete from history"),
                                        icon: <TrashIcon width={16} height={16} />,
                                        tone: "destructive",
                                        on_click: () => generation_controller.delete_image(selected.id)
                                    }
                                ]
                            }
                        ]}
                        close_label="Close"
                    >
                        <Image
                            src={selected.image_base64}
                            alt={selected.prompt}
                            className={`max-w-full max-h-[65vh] object-contain rounded-2xl select-none transition-opacity ${
                                state.is_generating ? "opacity-30" : ""
                            }`}
                        />
                    </ContextMenu>
                ) : (
                    !state.is_generating && (
                        <EmptyState
                            icon={<WandIcon width={36} height={36} />}
                            title="Results will appear here"
                            className="py-8!"
                        />
                    )
                )}

                {state.is_generating && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-black/60 dark:text-white/60 bg-white/30 dark:bg-black/30 backdrop-blur-xs">
                        <RefreshIcon width={28} height={28} className="animate-spin" />
                        <span className="text-sm text-center px-6">
                            {state.progress?.active
                                ? `Step ${state.progress.step} / ${state.progress.total_steps} · image ${state.progress.image_index + 1} / ${state.progress.image_count}`
                                : "Generating..."}
                        </span>
                        <Button
                            variant="ghost"
                            size="xs"
                            shape="pill"
                            onClick={() => generation_controller.cancel_generation()}
                            className="underline mt-0.5"
                        >
                            Cancel
                        </Button>
                    </div>
                )}
            </div>

            {selected && (
                <div className="flex items-start justify-between gap-4 px-2 py-0.5">
                    <div className="min-w-0 flex flex-col gap-1">
                        <p className="text-sm line-clamp-2 leading-relaxed">{selected.prompt || "(no prompt)"}</p>
                        <p className="text-xs font-mono text-black/50 dark:text-white/50">
                            {selected.type} · {selected.width}×{selected.height} · seed {selected.seed}
                            {selected.elapsed_seconds !== undefined && ` · ${selected.elapsed_seconds}s`}
                        </p>
                    </div>
                    <div className="flex shrink-0 gap-2 items-center pt-0.5">
                        <Button
                            variant="secondary"
                            size="sm"
                            shape="pill"
                            className="h-9 px-3.5 font-medium"
                            onClick={() => generation_controller.reuse_settings(selected)}
                            title="Copy prompt, seed and size into the editor"
                        >
                            Reuse
                        </Button>
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
                <HorizontalScroller className="gap-2.5 py-2 px-1.5">
                    {recent_items.map((item) => (
                        <ContextMenu
                            key={item.id}
                            sections={[
                                {
                                    items: [
                                        {
                                            key: "select",
                                            label: menu_label("Select image"),
                                            icon: <ImageIcon width={16} height={16} />,
                                            on_click: () => generation_controller.select_image(item)
                                        },
                                        {
                                            key: "reuse",
                                            label: menu_label("Reuse parameters"),
                                            icon: <WandIcon width={16} height={16} />,
                                            on_click: () => generation_controller.reuse_settings(item)
                                        },
                                        {
                                            key: "copy_prompt",
                                            label: menu_label("Copy prompt"),
                                            icon: <CopyIcon width={16} height={16} />,
                                            disabled: !item.prompt,
                                            on_click: () => {
                                                if (item.prompt) {
                                                    navigator.clipboard?.writeText(item.prompt)
                                                }
                                            }
                                        },
                                        {
                                            key: "download",
                                            label: menu_label("Download"),
                                            icon: <DownloadIcon width={16} height={16} />,
                                            on_click: () => download_image_file(item)
                                        },
                                        {
                                            key: "delete",
                                            label: menu_label("Delete"),
                                            icon: <TrashIcon width={16} height={16} />,
                                            tone: "destructive",
                                            on_click: () => generation_controller.delete_image(item.id)
                                        }
                                    ]
                                }
                            ]}
                            on_click_trigger={() => generation_controller.select_image(item)}
                            className={`shrink-0 size-16 rounded-xl overflow-hidden cursor-pointer transition ${
                                item.id === selected?.id
                                    ? "ring-2 ring-black dark:ring-white"
                                    : "opacity-60 hover:opacity-100"
                            }`}
                            close_label="Close"
                        >
                            <Image
                                src={item.image_base64}
                                alt={item.prompt}
                                className="w-full h-full object-cover"
                                loading="lazy"
                            />
                        </ContextMenu>
                    ))}
                </HorizontalScroller>
            )}
        </div>
    )
}
