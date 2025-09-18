"use client"

import {EpisodeInputBoxInputs} from "@/infra/types"
import {highlight} from "@/infra/custom_ui_constants"


function PlayerEpisodeInputBox({src,selected_episode,onClick}: EpisodeInputBoxInputs){
    return (
        <div
            className={`text-center rounded rounded-xl grid grid-cols-4 px-4 py-1 gap-x-3 max-h-[200px] overflow-y-auto overflow-x-hidden`}>
            {src.map((item, index) => (
                <div key={index}
                     className={`${index + 1 === selected_episode ? highlight : "hover:cursor-pointer text-white"}`}
                     onClick={() => onClick?.(index)}
                >
                    {index + 1}
                </div>
            ))}
        </div>
    )
}

export {PlayerEpisodeInputBox}
