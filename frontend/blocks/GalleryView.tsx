"use client"

import {
    X as CloseIcon,
    Images as GalleryIcon,
    Download as DownloadIcon,
    Trash2 as TrashIcon,
    WandSparkles as WandIcon,
    Copy as CopyIcon,
    Eye as EyeIcon
} from "lucide-react"
import { useState, useCallback } from "react"
import {
    Badge,
    Button,
    Card,
    CardSkeleton,
    ContextMenu,
    EmptyState,
    Image,
    LazyContainer,
    Modal,
    view_switcher_controller
} from "@/components"
import { generation_controller } from "@/core/controllers"
import { useController } from "@/components/hooks"
import type { GeneratedImageItem } from "@/core/api/types"

function menu_label(text: string) {
    return <span className="min-w-0 truncate text-[13px] font-medium leading-tight">{text}</span>
}

export function GalleryView() {
    const state = useController(generation_controller, {
        events: ["history_change", "change"]
    })

    const [inspected_item, set_inspected_item] = useState<GeneratedImageItem | null>(null)
    const [cached_item, set_cached_item] = useState<GeneratedImageItem | null>(null)

    const open_inspection = useCallback((item: GeneratedImageItem) => {
        view_switcher_controller.hide_toolbar()
        set_inspected_item(item)
        set_cached_item(item)
    }, [])

    const close_inspection = useCallback(() => {
        view_switcher_controller.show_toolbar()
        set_inspected_item(null)
    }, [])

    const current_modal_item = inspected_item ?? cached_item

    const handle_delete = async (id: string, e?: React.MouseEvent) => {
        e?.stopPropagation()
        await generation_controller.delete_image(id)
        if (inspected_item?.id === id) {
            close_inspection()
        }
    }

    const handle_reuse_prompt = useCallback(
        (item: GeneratedImageItem) => {
            generation_controller.reuse_settings(item)
            generation_controller.select_image(item)
            close_inspection()
            view_switcher_controller.switch_view("create", "xfusion_switcher")
        },
        [close_inspection]
    )

    const handle_copy_prompt = (prompt: string) => {
        if (!prompt) return
        navigator.clipboard?.writeText(prompt)
    }

    const handle_download = (item: GeneratedImageItem) => {
        const link = document.createElement("a")
        link.href = item.image_base64
        link.download = `xfusion_${item.seed}.png`
        link.click()
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
                <EmptyState
                    size="page"
                    icon={<GalleryIcon width={48} height={48} />}
                    title="No artworks generated yet"
                    description="Images you generate are saved on this device and listed here."
                />
            ) : (
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                    {state.history.map((item) => (
                        <LazyContainer
                            key={item.id}
                            className="w-full"
                            root_margin="100% 200%"
                            placeholder={<CardSkeleton aspect="square" />}
                        >
                            <ContextMenu
                                sections={[
                                    {
                                        items: [
                                            {
                                                key: "inspect",
                                                label: menu_label("View details"),
                                                icon: <EyeIcon width={16} height={16} />,
                                                on_click: () => open_inspection(item)
                                            },
                                            {
                                                key: "reuse",
                                                label: menu_label("Reuse parameters"),
                                                icon: <WandIcon width={16} height={16} />,
                                                on_click: () => handle_reuse_prompt(item)
                                            },
                                            {
                                                key: "copy_prompt",
                                                label: menu_label("Copy prompt"),
                                                icon: <CopyIcon width={16} height={16} />,
                                                disabled: !item.prompt,
                                                on_click: () => handle_copy_prompt(item.prompt)
                                            },
                                            {
                                                key: "download",
                                                label: menu_label("Download"),
                                                icon: <DownloadIcon width={16} height={16} />,
                                                on_click: () => handle_download(item)
                                            },
                                            {
                                                key: "delete",
                                                label: menu_label("Delete artwork"),
                                                icon: <TrashIcon width={16} height={16} />,
                                                tone: "destructive",
                                                on_click: () => handle_delete(item.id)
                                            }
                                        ]
                                    }
                                ]}
                                on_click_trigger={() => open_inspection(item)}
                                className="h-full"
                                close_label="Close"
                            >
                                <Card
                                    aspect="square"
                                    className="cursor-pointer hover:-translate-y-1"
                                    media={
                                        <Image
                                            src={item.image_base64}
                                            alt={item.prompt}
                                            className="w-full h-full object-cover group-hover:scale-105 transition duration-300"
                                            loading="lazy"
                                        />
                                    }
                                    hover_icon={<EyeIcon width={20} height={20} />}
                                    top_left={
                                        <Badge mono className="uppercase">
                                            {item.type}
                                        </Badge>
                                    }
                                    top_right={
                                        <Button
                                            variant="unstyled"
                                            size="icon"
                                            shape="pill"
                                            onPointerDown={(e) => e.stopPropagation()}
                                            onClick={(e) => handle_delete(item.id, e)}
                                            className="size-7 bg-black/55 text-white opacity-0 group-hover:opacity-100 focus-visible:opacity-100 hover:bg-black/80"
                                            title="Delete"
                                            aria-label="Delete artwork"
                                        >
                                            <TrashIcon width={13} height={13} />
                                        </Button>
                                    }
                                    title={item.prompt || "(no prompt)"}
                                    title_lines={2}
                                    meta={`${item.width}×${item.height}`}
                                    meta_trailing={
                                        item.elapsed_seconds !== undefined ? `${item.elapsed_seconds}s` : undefined
                                    }
                                />
                            </ContextMenu>
                        </LazyContainer>
                    ))}
                </div>
            )}

            {/* Artwork Details Modal */}
            <Modal
                open={!!inspected_item}
                on_close={close_inspection}
                duration={200}
                backdrop_class_name="bg-black/40"
                container_class_name="p-4 sm:p-6"
                className="p-5 sm:p-6 max-w-3xl w-full max-h-[88vh] overflow-y-auto"
            >
                {current_modal_item && (
                    <div className="flex flex-col md:flex-row gap-6">
                        <div className="md:w-1/2 flex items-center justify-center bg-black/5 dark:bg-white/5 rounded-2xl overflow-hidden aspect-square p-2.5 sm:p-3">
                            <Image
                                src={current_modal_item.image_base64}
                                alt={current_modal_item.prompt}
                                className="max-w-full max-h-full object-contain rounded-xl select-none"
                            />
                        </div>

                        <div className="md:w-1/2 flex flex-col justify-between">
                            <div className="flex flex-col gap-3.5">
                                <div className="flex justify-between items-start">
                                    <Badge tone="subtle" mono className="px-2.5 py-1 uppercase">
                                        {current_modal_item.type}
                                    </Badge>
                                    <Button
                                        variant="outline"
                                        size="icon"
                                        shape="pill"
                                        className="size-7"
                                        onClick={close_inspection}
                                        aria-label="Close"
                                    >
                                        <CloseIcon width={14} height={14} />
                                    </Button>
                                </div>

                                <div>
                                    <h4 className="text-[11px] font-semibold text-black/50 dark:text-white/50 uppercase tracking-wider">
                                        Prompt
                                    </h4>
                                    <p className="text-xs mt-1.5 p-3 rounded-xl bg-black/4 dark:bg-white/6 max-h-24 overflow-y-auto leading-relaxed">
                                        {current_modal_item.prompt || "(no prompt)"}
                                    </p>
                                </div>

                                {current_modal_item.negative_prompt && (
                                    <div>
                                        <h4 className="text-[11px] font-semibold text-black/50 dark:text-white/50 uppercase tracking-wider">
                                            Negative Prompt
                                        </h4>
                                        <p className="text-[11px] text-black/60 dark:text-white/60 mt-1.5 p-2.5 rounded-xl bg-black/4 dark:bg-white/6 max-h-16 overflow-y-auto leading-relaxed">
                                            {current_modal_item.negative_prompt}
                                        </p>
                                    </div>
                                )}

                                <div className="grid grid-cols-2 gap-2 text-[11px] p-3.5 rounded-xl bg-black/4 dark:bg-white/6 font-mono text-black/75 dark:text-white/75">
                                    <div>Seed: {current_modal_item.seed}</div>
                                    <div>
                                        Size: {current_modal_item.width}×{current_modal_item.height}
                                    </div>
                                    {current_modal_item.elapsed_seconds !== undefined && (
                                        <div>Time: {current_modal_item.elapsed_seconds}s</div>
                                    )}
                                </div>
                            </div>

                            <div className="flex gap-2.5 mt-6">
                                <Button
                                    variant="outline"
                                    size="auto"
                                    className="flex-1 py-2.5 px-4 text-xs font-semibold"
                                    onClick={() => handle_reuse_prompt(current_modal_item)}
                                >
                                    <span className="inline-flex items-center gap-1.5">
                                        <WandIcon width={14} height={14} />
                                        Reuse Prompt
                                    </span>
                                </Button>
                                <a
                                    href={current_modal_item.image_base64}
                                    download={`xfusion_${current_modal_item.seed}.png`}
                                    className="flex-1 py-2.5 rounded-xl bg-black dark:bg-white text-white dark:text-black text-xs font-semibold flex items-center justify-center hover:opacity-90 transition"
                                >
                                    <DownloadIcon width={14} height={14} className="mr-1.5" />
                                    Download
                                </a>
                            </div>
                        </div>
                    </div>
                )}
            </Modal>
        </div>
    )
}
