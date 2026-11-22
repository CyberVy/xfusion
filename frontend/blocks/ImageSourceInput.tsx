"use client"

import { Image, Trash2, Copy } from "lucide-react"
import { useRef, useState } from "react"
import type { ReactNode } from "react"
import { Button, ContextMenu } from "@/components"

export type ImageSourceInputProps = {
    image: string | null
    on_change: (image: string | null) => void
    children?: ReactNode
}

function menu_label(text: string) {
    return <span className="min-w-0 truncate text-[13px] font-medium leading-tight">{text}</span>
}

function read_file_as_data_url(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => resolve(reader.result as string)
        reader.onerror = () => reject(reader.error)
        reader.readAsDataURL(file)
    })
}

export function ImageSourceInput({ image, on_change, children }: ImageSourceInputProps) {
    const file_input_ref = useRef<HTMLInputElement | null>(null)
    const [is_drag_over, set_is_drag_over] = useState(false)

    const accept_file = async (file: File | undefined) => {
        if (!file || !file.type.startsWith("image/")) return
        on_change(await read_file_as_data_url(file))
    }

    const open_file_picker = () => file_input_ref.current?.click()

    const paste_from_clipboard = async () => {
        try {
            const items = await navigator.clipboard?.read?.()
            if (!items) return
            for (const item of items) {
                const image_type = item.types.find((t) => t.startsWith("image/"))
                if (image_type) {
                    const blob = await item.getType(image_type)
                    on_change(await read_file_as_data_url(new File([blob], "pasted.png", { type: image_type })))
                    return
                }
            }
        } catch {
            // Ignore clipboard errors
        }
    }

    return (
        <div
            onDragOver={(e) => {
                e.preventDefault()
                set_is_drag_over(true)
            }}
            onDragLeave={() => set_is_drag_over(false)}
            onDrop={(e) => {
                e.preventDefault()
                set_is_drag_over(false)
                accept_file(e.dataTransfer.files[0])
            }}
        >
            <input
                type="file"
                accept="image/*"
                ref={file_input_ref}
                onChange={(e) => {
                    accept_file(e.target.files?.[0])
                    e.target.value = ""
                }}
                className="hidden"
            />

            {image ? (
                <div className="flex flex-col gap-2.5">
                    <ContextMenu
                        className="w-full"
                        sections={[
                            {
                                items: [
                                    {
                                        key: "replace",
                                        label: menu_label("Replace image"),
                                        icon: <Image width={16} height={16} />,
                                        on_click: open_file_picker
                                    },
                                    {
                                        key: "paste",
                                        label: menu_label("Paste from clipboard"),
                                        icon: <Copy width={16} height={16} />,
                                        on_click: paste_from_clipboard
                                    },
                                    {
                                        key: "remove",
                                        label: menu_label("Remove image"),
                                        icon: <Trash2 width={16} height={16} />,
                                        tone: "destructive",
                                        on_click: () => on_change(null)
                                    }
                                ]
                            }
                        ]}
                        close_label="Close"
                    >
                        <div
                            className={`flex justify-center rounded-2xl p-3 sm:p-3.5 bg-black/4 dark:bg-white/6 transition ${
                                is_drag_over ? "ring-2 ring-black/40 dark:ring-white/40" : ""
                            }`}
                        >
                            {children}
                        </div>
                    </ContextMenu>
                    <div className="flex gap-2 px-1">
                        <Button
                            variant="secondary"
                            size="sm"
                            shape="pill"
                            className="h-8 font-medium"
                            onClick={open_file_picker}
                        >
                            Replace
                        </Button>
                        <Button
                            variant="ghost"
                            size="sm"
                            shape="pill"
                            className="h-8 font-medium"
                            onClick={() => on_change(null)}
                        >
                            Remove
                        </Button>
                    </div>
                </div>
            ) : (
                <Button
                    variant="unstyled"
                    onClick={open_file_picker}
                    className={`w-full h-40 rounded-2xl border border-dashed p-4 text-black/45 dark:text-white/45 ${
                        is_drag_over
                            ? "border-black/50 dark:border-white/50 bg-black/6 dark:bg-white/8"
                            : "border-black/15 dark:border-white/15 hover:border-black/30 dark:hover:border-white/30"
                    }`}
                >
                    <span className="flex flex-col items-center justify-center gap-2.5">
                        <Image width={28} height={28} />
                        <span className="text-sm font-medium">Click or drop an image</span>
                    </span>
                </Button>
            )}
        </div>
    )
}
