import {useState} from "react"
import {string_icons} from "@/infra/custom_ui_constants"


type SettingButtonInputs = {
    callback?: () => void
}

function SettingButton({callback}: SettingButtonInputs){
    return(
        <>
            <div
                className="select-none fixed text-2xl top-0 right-0 m-4 p-2 dark:bg-black bg-white dark:text-white text-black rounded rounded-xl">
                <button className="hover:cursor-pointer"
                        onClick={() => {
                            callback?.()
                        }}>
                    {string_icons.setting}
                </button>
            </div>
        </>
    )
}

export {SettingButton}