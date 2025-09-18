"use client"

import React, {useRef, useState} from "react"
import {string_icons} from "@/infra/custom_ui_constants"

type URLInputInputs = {
    default_url?: string
    callback: (url: string) => void
    description: string
}

type SearchWordInputInputs = {
    callback: (word: string) => void
}

function URLInput({default_url,callback,description}: URLInputInputs){
    const [is_collapsed,set_is_collapsed] = useState(false)
    const input_ref = useRef<HTMLInputElement | null>(null)
    return (
        <div className="p-4">
            <button className="text-lg font-bold mb-2 hover:cursor-pointer" onClick={
                () => {
                    set_is_collapsed(!is_collapsed)
                }
            }>
                {description} {is_collapsed ? string_icons.up_triangle : string_icons.down_triangle}
            </button>
            <div className={`flex gap-2 items-center ${is_collapsed ? "hidden" : "block"}`}>
                <input
                    type="text"
                    placeholder={`Please input a URL of ${description}`}
                    defaultValue={default_url || ""}
                    onChange={event => {
                        callback(event.target.value)
                    }}
                    className="border border-gray-300 rounded px-3 py-2 flex-1"
                    ref={input_ref}
                />
            </div>
        </div>
    )
}


function SearchWordInput({callback}: SearchWordInputInputs){
    const input_ref = useRef<HTMLInputElement | null>(null)
    return (
        <div className="p-4">
            <div className={`flex gap-2`}>
                <input
                    type="text"
                    placeholder="Search something? "
                    className="border border-gray-300 rounded rounded-lg px-3 py-2 flex-1"
                    ref={input_ref}
                    onKeyDown={event => {
                        if (event.key === "Enter"){
                            input_ref?.current?.blur()
                            callback(input_ref?.current?.value || "")
                        }
                    }}
                    onChange={event => {
                        if (event.target.value === ""){
                            callback(event.target.value)
                        }
                    }}
                />
                <button className="border border-gray-300 rounded rounded-lg px-3 py-2 hover:cursor-pointer"
                    onClick={event => callback(input_ref?.current?.value || "")}>
                    Search
                </button>
            </div>
        </div>
    )
}


export {URLInput, SearchWordInput}
