"use client"

import {string_icons} from "@/infra/custom_ui_constants"

function ScrollToTopButton() {
    return (
        <div className="select-none fixed text-2xl bottom-0 right-0 m-4 p-2 dark:bg-black bg-white dark:text-white text-black rounded rounded-xl">
            <button className="hover:cursor-pointer"
                onClick={() => {
                window.scrollTo(0,0)
            }}>
                {string_icons.up_triangle}
            </button>
        </div>
    )
}

function ScrollToBottomButton() {
    return (
        <div className="select-none fixed text-2xl bottom-0 right-10 m-4 p-2 dark:bg-black bg-white dark:text-white text-black rounded rounded-xl">
            <button className="hover:cursor-pointer"
                    onClick={() => {
                        window.scrollTo(0,document.documentElement.scrollHeight - document.documentElement.clientHeight - 1)
                    }}>
                {string_icons.down_triangle}
            </button>
        </div>
    )
}

export {ScrollToTopButton,ScrollToBottomButton}
