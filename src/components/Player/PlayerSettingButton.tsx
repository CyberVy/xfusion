"use client"


import {useState} from "react"
import {highlight, string_icons} from "@/infra/custom_ui_constants"
import {URLInput} from "@/components/FormInputs"


type PlayerSettingButtonInputs = {
    callback?: (url: string) => void
}

function PlayerSettingButton({callback}: PlayerSettingButtonInputs){
    const [is_open,set_is_open] = useState(false)

    return (
        <div className="relative">
            <button
                className={`${is_open ? highlight : ""} select-none text-center hover:cursor-pointer`}
                onClick={() => set_is_open(!is_open)}
            >
                {string_icons.setting}
            </button>
            <div className={`${is_open ? "block" : "hidden"} absolute left-1/2 -translate-x-1/2 top-5 z-1 bg-black rounded rounded-xl`}>
                <button
                    className="absolute top-0 left-0 text-center text-xl ml-3 hover:cursor-pointer"
                    onClick={() => set_is_open(false)}
                >
                    {string_icons.close}
                </button>
                <URLInput description="M3U8 Proxy API" callback={url => callback?.(url)} />
            </div>
        </div>
    )
}

export {PlayerSettingButton}