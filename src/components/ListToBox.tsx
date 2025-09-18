"use client"

import {useState} from "react"
import {highlight, string_icons} from "@/infra/custom_ui_constants"

type ListToButtonsInputs = {
    list: string[]
    callback?: (item: string | null) => void
}

function ListToButtons({ list, callback }: ListToButtonsInputs) {

    const [selected_item,set_selected_item] = useState<string | number | null>(null)
    const [is_collapsed,set_is_collapsed] = useState(true)
    const default_item_style = "px-4 py-2  rounded-xl border hover:bg-zinc-700"
    const selected_item_style = `px-4 py-2 ${highlight} border rounded-2xl hover:bg-zinc-700`

    return (
        <div className="select-none">
            <button className={`text-2xl mx-4 px-2 text-zinc-300 hover:cursor-pointer border rounded rounded-xl`}
                    onClick={() => set_is_collapsed(!is_collapsed)}>
                <span className={`${selected_item ? highlight : "text-black dark:text-white"}`}>
                    {string_icons.menu} {is_collapsed ? string_icons.down_triangle : string_icons.up_triangle} {list.length}
                </span>
            </button>

            {<div className={`flex flex-wrap gap-1 py-2 px-11  max-h-[100px] overflow-y-auto ${is_collapsed ? 'hidden' : 'block'}`}>
                {list.map((item, index) => (
                    <button
                        key={index}
                        className={item === selected_item ? selected_item_style : default_item_style}
                        onClick={event => {
                            set_selected_item(item === selected_item ? null : item)
                            callback?.(item === selected_item ? null : item)
                        }}>
                        {item}
                    </button>
                ))}
            </div>}
        </div>
    )}


export {ListToButtons}
