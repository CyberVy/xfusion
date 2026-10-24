"use client"

import { useRef, useState } from "react"
import type { PointerEvent } from "react"
import { NaiveButton } from "@/components"

export type MaskPainterProps = {
    image_src: string
    on_mask_change: (mask_data_url: string | null) => void
}

type CanvasPoint = { x: number; y: number }

export function MaskPainter({ image_src, on_mask_change }: MaskPainterProps) {
    const canvas_ref = useRef<HTMLCanvasElement | null>(null)
    const last_point_ref = useRef<CanvasPoint | null>(null)
    const [natural_size, set_natural_size] = useState<{ width: number; height: number } | null>(null)
    const [brush_size, set_brush_size] = useState(36)

    const to_canvas_point = (e: PointerEvent<HTMLCanvasElement>): CanvasPoint => {
        const rect = e.currentTarget.getBoundingClientRect()
        const scale = e.currentTarget.width / rect.width
        return { x: (e.clientX - rect.left) * scale, y: (e.clientY - rect.top) * scale }
    }

    const draw_segment = (canvas: HTMLCanvasElement, from: CanvasPoint, to: CanvasPoint) => {
        const ctx = canvas.getContext("2d")
        if (!ctx) return
        ctx.lineWidth = brush_size * (canvas.width / canvas.getBoundingClientRect().width)
        ctx.lineCap = "round"
        ctx.lineJoin = "round"
        ctx.strokeStyle = "#ffffff"
        ctx.beginPath()
        ctx.moveTo(from.x, from.y)
        ctx.lineTo(to.x, to.y)
        ctx.stroke()
    }

    const handle_pointer_down = (e: PointerEvent<HTMLCanvasElement>) => {
        e.currentTarget.setPointerCapture(e.pointerId)
        const point = to_canvas_point(e)
        draw_segment(e.currentTarget, point, point)
        last_point_ref.current = point
    }

    const handle_pointer_move = (e: PointerEvent<HTMLCanvasElement>) => {
        if (!last_point_ref.current) return
        const point = to_canvas_point(e)
        draw_segment(e.currentTarget, last_point_ref.current, point)
        last_point_ref.current = point
    }

    const handle_pointer_end = (e: PointerEvent<HTMLCanvasElement>) => {
        if (!last_point_ref.current) return
        last_point_ref.current = null
        on_mask_change(e.currentTarget.toDataURL("image/png"))
    }

    const clear_mask = () => {
        const canvas = canvas_ref.current
        canvas?.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height)
        on_mask_change(null)
    }

    return (
        <div className="flex flex-col items-center gap-3 w-full">
            <div className="relative max-w-full">
                <img
                    src={image_src}
                    alt="Inpainting source"
                    onLoad={(e) =>
                        set_natural_size({
                            width: e.currentTarget.naturalWidth,
                            height: e.currentTarget.naturalHeight
                        })
                    }
                    className="block max-w-full max-h-[50vh] w-auto h-auto rounded-xl pointer-events-none select-none"
                    draggable={false}
                />
                {natural_size && (
                    <canvas
                        ref={canvas_ref}
                        width={natural_size.width}
                        height={natural_size.height}
                        onPointerDown={handle_pointer_down}
                        onPointerMove={handle_pointer_move}
                        onPointerUp={handle_pointer_end}
                        onPointerCancel={handle_pointer_end}
                        className="absolute inset-0 w-full h-full rounded-xl cursor-crosshair touch-none mix-blend-difference"
                    />
                )}
            </div>

            <div className="flex items-center gap-3 w-full text-xs text-black/60 dark:text-white/60">
                <span className="shrink-0">Brush</span>
                <input
                    type="range"
                    min={8}
                    max={120}
                    value={brush_size}
                    onChange={(e) => set_brush_size(Number(e.target.value))}
                    className="flex-1 accent-black dark:accent-white cursor-pointer"
                />
                <span className="w-8 font-mono text-right">{brush_size}</span>
                <NaiveButton
                    className="h-7 px-3 rounded-full font-medium bg-black/5 dark:bg-white/8 hover:bg-black/10 dark:hover:bg-white/12"
                    on_click={clear_mask}
                >
                    Clear mask
                </NaiveButton>
            </div>
        </div>
    )
}
