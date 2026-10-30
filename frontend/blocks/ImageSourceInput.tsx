"use client"

import { useRef, useState } from "react"
import type { ReactNode } from "react"
import { NaiveButton } from "@/components"
import { ImageIcon } from "./icons"

export type ImageSourceInputProps = {
    image: string | null
    on_change: (image: string | null) => void
    children?: ReactNode
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
                <div className="flex flex-col gap-2">
                    <div
                        className={`flex justify-center rounded-2xl p-2 bg-black/4 dark:bg-white/6 transition ${
                            is_drag_over ? "ring-2 ring-black/40 dark:ring-white/40" : ""
                        }`}
                    >
                        {children}
                    </div>
                    <div className="flex gap-2">
                        <NaiveButton
                            className="h-8 px-3 rounded-full text-xs font-medium bg-black/5 dark:bg-white/8 hover:bg-black/10 dark:hover:bg-white/12"
                            on_click={open_file_picker}
                        >
                            Replace
                        </NaiveButton>
                        <NaiveButton
                            className="h-8 px-3 rounded-full text-xs font-medium text-black/60 dark:text-white/60 hover:bg-black/5 dark:hover:bg-white/8"
                            on_click={() => on_change(null)}
                        >
                            Remove
                        </NaiveButton>
                    </div>
                </div>
            ) : (
                <button
                    type="button"
                    onClick={open_file_picker}
                    className={`w-full h-40 rounded-2xl border border-dashed flex flex-col items-center justify-center gap-2 text-black/45 dark:text-white/45 cursor-pointer transition ${
                        is_drag_over
                            ? "border-black/50 dark:border-white/50 bg-black/6 dark:bg-white/8"
                            : "border-black/15 dark:border-white/15 hover:border-black/30 dark:hover:border-white/30"
                    }`}
                >
                    <ImageIcon width={28} height={28} />
                    <span className="text-sm font-medium">Click or drop an image</span>
                </button>
            )}
        </div>
    )
}
