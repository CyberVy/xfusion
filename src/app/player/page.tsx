"use client"

import {useState, useRef, useEffect} from "react"
import {VideoPlayer} from "@/components/Player/Player"
import {URLInput} from "@/components/FormInputs"

export default function Page() {

    const [src,set_src] = useState("")
    const [title,set_title] = useState("")

    useEffect(() => {
        const current_url = new URL(location.href)
        const url = current_url.searchParams.get("url")
        const title = current_url.searchParams.get("title")
        if (url){
            set_src(url)
        }
        if (title){
            set_title(title)
        }
        history.replaceState(null,"","/player")
    }, [])

    useEffect(() => {
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js').then(() => {
                console.log('Service worker is registered successfully.')
            }).catch(err => {console.error('Failed to register Service worker.', err)})}
    }, [])

    return (

        <div className="w-full overflow-x-hidden">
            <URLInput
                default_url={src}
                description="M3U8 Resource"
                callback={url => {
                    if (/^https?:\/\//.test(url)) {
                        set_src(url)
                    }
                }}
            />
            {src && <VideoPlayer src={[src]} description="" title={title}/>}
        </div>
    )
}
