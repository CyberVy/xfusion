"use client"

import {useState, useRef} from "react"
import {generate_silent_wav_base64} from "@/infra/data_generation_lib";

export default function Page() {
    const blank_audio_ref = useRef<HTMLAudioElement | null>(null)
    return (
        <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
            <audio
                ref={blank_audio_ref}
                src={generate_silent_wav_base64(3)}
                controls
            >
            </audio>
            <button onClick={() => {
                blank_audio_ref.current?.play()
            }}>
                click
            </button>
        </div>
    )
}
