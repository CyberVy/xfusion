"use client"

import { Download, RefreshCw, WandSparkles, Copy, Trash2, Image } from "lucide-react"
import { Button, ContextMenu, EmptyState, HorizontalScroller, Image as ImageView, Text } from "@/components"
import { useController } from "@/components/hooks"
import { generation_controller } from "@/core/controllers"
import type { GeneratedImageItem } from "@/core/api"

const RECENT_RESULT_COUNT = 24

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
            <div className="relative flex items-center justify-center min-h-[40vh] md:min-h-[60vh] rounded-3xl bg-ui-fill-subtle p-3 sm:p-4 overflow-hidden">
                {selected ? (
                    <ContextMenu
                        className="w-full h-full flex items-center justify-center"
                        sections={[
                            {
                                items: [
                                    {
                                        key: "reuse",
                                        label: "Reuse parameters",
                                        icon: <WandSparkles width={16} height={16} />,
                                        on_click: () => generation_controller.reuse_settings(selected)
                                    },
                                    {
                                        key: "copy_prompt",
                                        label: "Copy prompt",
                                        icon: <Copy width={16} height={16} />,
                                        disabled: !selected.prompt,
                                        on_click: () => {
                                            if (selected.prompt) {
                                                navigator.clipboard?.writeText(selected.prompt)
                                            }
                                        }
                                    },
                                    {
                                        key: "copy_image",
                                        label: "Copy image",
                                        icon: <Image width={16} height={16} />,
                                        on_click: () => copy_image_to_clipboard(selected.image_base64)
                                    },
                                    {
                                        key: "download",
                                        label: "Download",
                                        icon: <Download width={16} height={16} />,
                                        on_click: () => download_image_file(selected)
                                    },
                                    {
                                        key: "delete",
                                        label: "Delete from history",
                                        icon: <Trash2 width={16} height={16} />,
                                        tone: "destructive",
                                        on_click: () => generation_controller.delete_image(selected.id)
                                    }
                                ]
                            }
                        ]}
                        close_label="Close"
                    >
                        <ImageView
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
                            icon={<WandSparkles width={36} height={36} />}
                            title="Results will appear here"
                            className="py-8!"
                        />
                    )
                )}

                {state.is_generating && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-ui-label-secondary bg-ui-surface-glass backdrop-blur-xs">
                        <RefreshCw width={28} height={28} className="animate-spin" />
                        <Text variant="body" tone="inherit" align="center" className="px-6">
                            {state.progress?.active
                                ? `Step ${state.progress.step} / ${state.progress.total_steps} · image ${state.progress.image_index + 1} / ${state.progress.image_count}`
                                : "Generating..."}
                        </Text>
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
                        <Text as="p" variant="body" lines={2} className="leading-relaxed">
                            {selected.prompt || "(no prompt)"}
                        </Text>
                        <Text as="p" variant="caption" tone="secondary" mono>
                            {selected.type} · {selected.width}×{selected.height} · seed {selected.seed}
                            {selected.elapsed_seconds !== undefined && ` · ${selected.elapsed_seconds}s`}
                        </Text>
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
                            className="size-9 rounded-full flex items-center justify-center bg-ui-fill-solid text-ui-label-inverse hover:opacity-85 transition"
                            title="Download"
                        >
                            <Download width={16} height={16} />
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
                                            label: "Select image",
                                            icon: <Image width={16} height={16} />,
                                            on_click: () => generation_controller.select_image(item)
                                        },
                                        {
                                            key: "reuse",
                                            label: "Reuse parameters",
                                            icon: <WandSparkles width={16} height={16} />,
                                            on_click: () => generation_controller.reuse_settings(item)
                                        },
                                        {
                                            key: "copy_prompt",
                                            label: "Copy prompt",
                                            icon: <Copy width={16} height={16} />,
                                            disabled: !item.prompt,
                                            on_click: () => {
                                                if (item.prompt) {
                                                    navigator.clipboard?.writeText(item.prompt)
                                                }
                                            }
                                        },
                                        {
                                            key: "download",
                                            label: "Download",
                                            icon: <Download width={16} height={16} />,
                                            on_click: () => download_image_file(item)
                                        },
                                        {
                                            key: "delete",
                                            label: "Delete",
                                            icon: <Trash2 width={16} height={16} />,
                                            tone: "destructive",
                                            on_click: () => generation_controller.delete_image(item.id)
                                        }
                                    ]
                                }
                            ]}
                            on_click_trigger={() => generation_controller.select_image(item)}
                            className={`shrink-0 size-16 rounded-xl overflow-hidden cursor-pointer transition ${
                                item.id === selected?.id ? "ring-2 ring-ui-fill-solid" : "opacity-60 hover:opacity-100"
                            }`}
                            close_label="Close"
                        >
                            <ImageView
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
