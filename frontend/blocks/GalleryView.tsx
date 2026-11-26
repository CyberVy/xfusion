"use client"

import { X, Images, Download, Trash2, WandSparkles, Copy, Eye } from "lucide-react"
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
    Text,
    view_switcher_controller
} from "@/components"
import { generation_controller } from "@/core/controllers"
import { useController } from "@/components/hooks"
import type { GeneratedImageItem } from "@/core/api"

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
                    <Text as="h2" variant="title" weight="semibold">
                        Artworks Gallery
                    </Text>
                    <Text as="p" variant="caption" tone="secondary">
                        {state.history.length} creations saved in local device storage
                    </Text>
                </div>
            </div>

            {state.history.length === 0 ? (
                <EmptyState
                    size="page"
                    icon={<Images width={48} height={48} />}
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
                                                label: "View details",
                                                icon: <Eye width={16} height={16} />,
                                                on_click: () => open_inspection(item)
                                            },
                                            {
                                                key: "reuse",
                                                label: "Reuse parameters",
                                                icon: <WandSparkles width={16} height={16} />,
                                                on_click: () => handle_reuse_prompt(item)
                                            },
                                            {
                                                key: "copy_prompt",
                                                label: "Copy prompt",
                                                icon: <Copy width={16} height={16} />,
                                                disabled: !item.prompt,
                                                on_click: () => handle_copy_prompt(item.prompt)
                                            },
                                            {
                                                key: "download",
                                                label: "Download",
                                                icon: <Download width={16} height={16} />,
                                                on_click: () => handle_download(item)
                                            },
                                            {
                                                key: "delete",
                                                label: "Delete artwork",
                                                icon: <Trash2 width={16} height={16} />,
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
                                    hover_icon={<Eye width={20} height={20} />}
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
                                            className="size-7 bg-ui-scrim text-ui-label-overlay opacity-0 group-hover:opacity-100 focus-visible:opacity-100 hover:bg-ui-scrim-strong"
                                            title="Delete"
                                            aria-label="Delete artwork"
                                        >
                                            <Trash2 width={13} height={13} />
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
                container_class_name="p-4 sm:p-6"
                className="p-5 sm:p-6 max-w-3xl w-full max-h-[88vh] overflow-y-auto"
            >
                {current_modal_item && (
                    <div className="flex flex-col md:flex-row gap-6">
                        <div className="md:w-1/2 flex items-center justify-center bg-ui-fill-subtle rounded-2xl overflow-hidden aspect-square p-2.5 sm:p-3">
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
                                        <X width={14} height={14} />
                                    </Button>
                                </div>

                                <div>
                                    <Text
                                        as="h4"
                                        variant="footnote"
                                        weight="semibold"
                                        tone="secondary"
                                        className="uppercase tracking-wider"
                                    >
                                        Prompt
                                    </Text>
                                    <Text
                                        as="p"
                                        variant="caption"
                                        className="mt-1.5 p-3 rounded-xl bg-ui-fill-subtle max-h-24 overflow-y-auto leading-relaxed"
                                    >
                                        {current_modal_item.prompt || "(no prompt)"}
                                    </Text>
                                </div>

                                {current_modal_item.negative_prompt && (
                                    <div>
                                        <Text
                                            as="h4"
                                            variant="footnote"
                                            weight="semibold"
                                            tone="secondary"
                                            className="uppercase tracking-wider"
                                        >
                                            Negative Prompt
                                        </Text>
                                        <Text
                                            as="p"
                                            variant="footnote"
                                            tone="secondary"
                                            className="mt-1.5 p-2.5 rounded-xl bg-ui-fill-subtle max-h-16 overflow-y-auto leading-relaxed"
                                        >
                                            {current_modal_item.negative_prompt}
                                        </Text>
                                    </div>
                                )}

                                <div className="grid grid-cols-2 gap-2 p-3.5 rounded-xl bg-ui-fill-subtle">
                                    <Text as="div" variant="footnote" mono>
                                        Seed: {current_modal_item.seed}
                                    </Text>
                                    <Text as="div" variant="footnote" mono>
                                        Size: {current_modal_item.width}×{current_modal_item.height}
                                    </Text>
                                    {current_modal_item.elapsed_seconds !== undefined && (
                                        <Text as="div" variant="footnote" mono>
                                            Time: {current_modal_item.elapsed_seconds}s
                                        </Text>
                                    )}
                                </div>
                            </div>

                            <div className="flex gap-2.5 mt-6">
                                <Button
                                    variant="outline"
                                    size="auto"
                                    className="flex-1 py-2.5 px-4"
                                    onClick={() => handle_reuse_prompt(current_modal_item)}
                                >
                                    <span className="inline-flex items-center gap-1.5">
                                        <WandSparkles width={14} height={14} />
                                        <Text variant="caption" weight="semibold" tone="inherit">
                                            Reuse Prompt
                                        </Text>
                                    </span>
                                </Button>
                                <a
                                    href={current_modal_item.image_base64}
                                    download={`xfusion_${current_modal_item.seed}.png`}
                                    className="flex-1 py-2.5 rounded-xl bg-ui-fill-solid text-ui-label-inverse flex items-center justify-center hover:opacity-90 transition"
                                >
                                    <Download width={14} height={14} className="mr-1.5" />
                                    <Text variant="caption" weight="semibold" tone="inherit">
                                        Download
                                    </Text>
                                </a>
                            </div>
                        </div>
                    </div>
                )}
            </Modal>
        </div>
    )
}
