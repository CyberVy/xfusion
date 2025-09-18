"use client"


import React, {useRef, useState} from "react"
import {string_icons} from "@/infra/custom_ui_constants";
import {generate_cover_image} from "@/infra/data_generation_lib";


type LabeledImageInputs = {
    src: string | undefined
    label: string
    label_background_color?: string
    alt?: string
    description? : string
    onClickImage?: () => void
    onClickDelete?: () => void
}

function LabeledImage({src,label,label_background_color,alt,onClickImage,onClickDelete,description}: LabeledImageInputs) {
    const [is_loaded,set_is_loaded] = useState(false)
    const [show_description,set_show_description] = useState(false)

    return (
        <div className="relative">
            <img
                src={src} alt={alt}
                className="w-full h-full object-cover"
                onClick={onClickImage}
                onLoad={() => set_is_loaded(true)}
                onError={async event => {
                    event.currentTarget.src = await generate_cover_image(alt || "",{})
                    set_is_loaded(true)
                }}
            />
            <div
                className={`absolute top-2 left-2 px-2 text-white text-xs font-bold rounded-md ${label_background_color} ${is_loaded ? "block" : "hidden"}`}
            >
                {label}
            </div>

            <div>
                {description && <button
                    className={`opacity-50 border hover:cursor-pointer absolute bottom-1 right-1 px-2 text-white text-xs font-bold rounded-md ${is_loaded ? "block" : "hidden"}`}
                    onClick={() => {
                        set_show_description(!show_description)
                    }}
                    onMouseEnter={() => {
                        set_show_description(true)
                    }}
                    onMouseLeave={() => {
                        set_show_description(false)
                    }}
                >
                    {string_icons.info}
                </button>}
                <div
                    className={`mx-2 max-h-[300px] overflow-y-auto bg-black/50 absolute bottom-10 right-2 px-2 py-1 text-white text-sm italic rounded-lg ${show_description ? "block" : "hidden"}`}
                >
                    <p className="text-center whitespace-pre-line">
                        {description}
                    </p>
                </div>
            </div>

            {onClickDelete && <div>
                <button
                    className="absolute bottom-1 left-2 px-2 text-red-400 border text-xs rounded-md hover:cursor-pointer"
                    onClick={onClickDelete}
                >
                    {string_icons.del}
                </button>
            </div>}

        </div>
    )
}

export {LabeledImage}
