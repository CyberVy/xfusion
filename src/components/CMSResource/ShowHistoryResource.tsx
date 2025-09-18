"use client"

import {ResourceListItem} from "@/infra/types"
import {useEffect, useState} from "react"
import {LabeledImage} from "@/components/LabeledImage"
import {highlight, string_icons} from "@/infra/custom_ui_constants"
import {HistoryButtonInputs,ShowHistoryResourceInputs} from "@/infra/types"


function ShowHistoryResource({history_resource,callback,delete_callback}: ShowHistoryResourceInputs){

    return (
        <>
            {history_resource &&
                <div className="select-none grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-0">
                    {history_resource.map((v: ResourceListItem, index) => {
                        const score = v.vod_douban_score != "0.0" ? v.vod_douban_score : v.vod_score
                        let label_background_color = Number(score) >= 8.0 ? "bg-red-700" : Number(score) >= 6.0 ? "bg-yellow-700" : "bg-teal-700"
                        label_background_color = Number(score) == 0.0 ? "" : label_background_color
                        return (
                            <LabeledImage
                                key={index} alt={v.vod_name}
                                label={score != "0.0" ? score : ""}
                                label_background_color={label_background_color}
                                src={v.vod_pic !== "" ? v.vod_pic : undefined}
                                description={v._vod_description}
                                onClickImage={() => {
                                    callback?.(v)
                                }}
                                onClickDelete={() => {
                                    delete_callback?.(v)
                                }}
                            />
                        )
                    })}
                </div>
            }
        </>
    )
}

function HistoryButton({callback}: HistoryButtonInputs) {
    const [is_active,set_is_active] = useState(false)
    return (
        <div
            className={`${is_active ? highlight : ""} select-none fixed text-2xl bottom-13 right-0 m-4 p-2 dark:bg-black bg-white rounded rounded-xl`}>
            <button className="hover:cursor-pointer"
                    onClick={() => {
                        callback?.()
                        set_is_active(!is_active)
                    }}>
                {string_icons.history}
            </button>
        </div>
    )
}

export {
    ShowHistoryResource,
    HistoryButton
}