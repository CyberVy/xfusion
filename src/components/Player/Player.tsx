"use client"

import {useEffect, useRef, useState} from "react"


import {XPlayer} from "@/infra/player.client"
import {PlayerEpisodeInputBox} from "@/components/Player/PlayerEpisodeInputBox"
import {PlayerSettingButton} from "@/components/Player/PlayerSettingButton"

import Player from "video.js/dist/types/player"
import ControlBar from 'video.js/dist/types/control-bar/control-bar'

import {VideoPlayerInputs} from "@/infra/types"
import {generate_silent_wav_base64} from "@/infra/data_generation_lib"
import {highlight, string_icons} from "@/infra/custom_ui_constants"


declare global{
    interface Window {
        xplayer?: XPlayer
    }
}

function VideoPlayer({src,title,description,poster_url,start_episode,start_time,counter,callback}: VideoPlayerInputs) {

    if (!start_episode || start_episode > src.length || start_episode < 1){
        start_episode = 1
    }
    start_time = start_time || 0

    const xplayer_ref = useRef<XPlayer | null>(null)
    const [is_open,set_is_open] = useState(true)
    const [is_open_episode_selector,set_is_open_episode_selector] = useState(false)
    const [selected_episode,set_selected_episode] = useState(start_episode)
    const [resolution,set_resolution] = useState([0,0])

    const video_ref = useRef<HTMLVideoElement | null>(null)
    const blank_audio_ref = useRef<HTMLAudioElement | null>(null)
    const episode_input_ref = useRef<HTMLInputElement | null>(null)

    const [share_icon,set_share_icon] = useState(string_icons.share)

    useEffect(() => {
        return () => {
            xplayer_ref.current?.dispose()
            delete window.xplayer
        }
    }, [])

    useEffect(() => {
        set_is_open(true)
        set_is_open_episode_selector(false)
        set_selected_episode(start_episode)
        set_resolution([0,0])
        if (episode_input_ref.current?.value) {
            episode_input_ref.current.value = ""
        }

        if (src.length){
            // create a long life circle player
            if (!xplayer_ref.current) {
                console.log("Creating the XPlayer...")
                xplayer_ref.current = new XPlayer(video_ref.current!,blank_audio_ref.current,{
                    controlBar:{
                        skipButtons:{
                            forward: 10,
                            backward: 10
                        }},
                    enableSmoothSeeking: true,
                    playsinline: true,
                    playbackRates: [0.5, 1, 1.5, 2],
                    autoplay: true,
                    controls: true,
                    client_height_ratio: 0.5,
                    client_width_ratio: 1.0
                })
                const caption_button = (xplayer_ref.current._player as Player & {controlBar: ControlBar})
                    .controlBar.getChild('SubsCapsButton')
                if (caption_button){
                    (xplayer_ref.current._player as Player & {controlBar: ControlBar}).controlBar.removeChild(caption_button)
                }
                xplayer_ref.current.set_auto_play_next_episode_callback(() => set_selected_episode(selected_episode => selected_episode + 1))
                xplayer_ref.current.set_media_session_next_track_callback(() => set_selected_episode(selected_episode => selected_episode + 1))
                xplayer_ref.current.set_media_session_previous_track_callback(() => set_selected_episode(selected_episode => selected_episode - 1))
                xplayer_ref.current.on("loadedmetadata",() => {
                    // Safari can not get the resolution instantly after loadedmetadata is triggered.
                    if (navigator.userAgent.toLowerCase().includes("safari")){
                        xplayer_ref.current?.one("playing",() => update_resolution())
                        return
                    }
                    update_resolution()
                })
                callback?.(xplayer_ref.current)
                window.xplayer = xplayer_ref.current
            }

            xplayer_ref.current.set_source(src)
            xplayer_ref.current.set_media_session_metadata(title || "",poster_url)
            xplayer_ref.current.play_episode(start_episode,start_time)
            xplayer_ref.current.on("leavepictureinpicture",() => {
                set_is_open(true)
                xplayer_ref.current?.play()
            })
        }
    }, [src,counter])

    function update_resolution(){
        const resolution = [0,0]
        resolution[0] = xplayer_ref.current?.get_video_width() || 0
        resolution[1] = xplayer_ref.current?.get_video_height() || 0
        set_resolution(resolution)
    }

    function close_player_callback(player: XPlayer){
        set_is_open(false)
        set_is_open_episode_selector(false)
        if (player._player.isInPictureInPicture()){
            return
        }

        if (!player.paused()){
            player.pause()
        }
        else{
            // if the video is not loaded completely, the video will not be played simply after load is finished
            // if the video is loaded but paused by the user, the autoplay will be disabled, activate it manually.
            player.set_autoplay(false)
            player.one("play",() => player.set_autoplay(true))
        }
    }

    return (
        <div className={`select-none rounded rounded-2xl text-white dark:text-zinc-200 bg-black/50 dark:bg-black/50 ${is_open ? 'block' : 'hidden'}`}
        >

            {/**
              * This hidden audio element plays a short silent loop to keep the Media Session active.
             * Some browsers (especially on iOS) require an immediate media response after a user gesture
             * without an active media element, Media Session actions may fail.
             */
            }
            <audio
                ref={blank_audio_ref}
                loop
                src={generate_silent_wav_base64(3)}
            >
            </audio>

            {/** main player UI */}
            <main>
                <button
                    className="text-center text-xl ml-3 hover:cursor-pointer"
                    onClick={
                        () => {
                            close_player_callback(xplayer_ref.current!)
                        }
                    }
                >
                    {string_icons.close}
                </button>

                <div className="text-center">
                    <PlayerSettingButton callback={url => xplayer_ref.current?.set_m3u8_chunk_proxy(url)}/>
                </div>

                <div className="text-center">
                    <h1 className="font-bold text-xl max-h-[30px] overflow-y-auto">
                        {title || ""}
                    </h1>
                    <span className="text-xs italic">
                    {resolution[0] != 0 && resolution[1]!= 0 ? `${resolution[0]} x ${resolution[1]}` : "..."}
                    </span>
                </div>

                {/** This div allows users to navigate to the previous, next episode or an input episode. */}
                <div className="text-center font-bold">
                    <span className="mr-10 hover:cursor-pointer"
                          onClick={event => {
                              if (xplayer_ref.current?.previous()){
                                  set_selected_episode(selected_episode => selected_episode - 1)
                              }
                          }}>
                        {string_icons.left_arrow}
                    </span>
                    {selected_episode}/{src.length}
                    <span className="ml-10 hover:cursor-pointer"
                          onClick={event => {
                              if(xplayer_ref.current?.next()){
                                  set_selected_episode(selected_episode => selected_episode + 1)
                              }
                          }}>
                        {string_icons.right_arrow}
                    </span>
                    <div className="mb-1">
                        <input
                            ref={episode_input_ref}
                            placeholder=">>>"
                            className="w-15 text-center px-2 border border-gray-300 rounded-md focus:outline-none focus:ring-1 focus:ring-black-500"
                            onKeyDown={event => {
                                if (event.key === "Enter") {
                                    episode_input_ref.current?.blur()
                                    const input_episode = Number(episode_input_ref.current?.value || selected_episode)
                                    if (input_episode == selected_episode){
                                        const current_time = xplayer_ref.current?.get_current_time() || 0
                                        xplayer_ref.current?.one("loadedmetadata",() => xplayer_ref.current?.set_current_time(current_time))
                                    }
                                    if (xplayer_ref.current?.check_episode(input_episode)) {
                                        xplayer_ref.current?.play_episode(input_episode)
                                        set_selected_episode(input_episode)
                                    }
                                }
                            }}
                        />
                    </div>
                </div>

                <div className="select-none text-center">
                    <video ref={video_ref} className="video-js"/>
                </div>

                <div className="select-none mx-8 mb-0">
                    <button
                        className="mx-4 text-center font-bold border rounded rounded-lg px-5 hover:cursor-pointer"
                        onClick={
                            () => {
                                close_player_callback(xplayer_ref.current!)
                            }
                        }
                    >
                        {string_icons.back}
                    </button>

                    <button className="mx-4 text-center font-bold border rounded rounded-lg px-5 hover:cursor-pointer"
                            onClick={() => {
                                const share_url = xplayer_ref.current?.source[selected_episode - 1]
                                if (share_url) {
                                    navigator.clipboard.writeText(encodeURI(share_url)).then(() => {
                                        set_share_icon(string_icons.success)
                                        setTimeout(() => set_share_icon(string_icons.share),2000)
                                    })
                                }
                            }}
                    >
                        {share_icon}
                    </button>

                    {src.length > 1 &&
                        <button
                            className={`${is_open_episode_selector ? highlight : ""} mx-4 text-center font-bold border rounded rounded-lg px-3 hover:cursor-pointer`}
                            onClick={() => {
                                set_is_open_episode_selector(is_open_episode_selector => !is_open_episode_selector)
                            }}
                        >
                            {selected_episode}/{src.length}
                        </button>}
                </div>

                <div className="max-h-[75px] overflow-y-auto p-2 pt-1">
                    <p className="text-center italic whitespace-pre-line">
                        {description}
                    </p>
                </div>
            </main>

            {/** episode selector UI */}
            <div
                className={`select-none bg-black/50 fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded rounded-xl ${is_open_episode_selector ? 'block' : 'hidden'}`}>
                <button className="text-center text-white text-xl ml-3 hover:cursor-pointer" onClick={
                    () => {
                        set_is_open_episode_selector(false)
                    }
                }>
                    {string_icons.close}
                </button>
                <PlayerEpisodeInputBox src={src} selected_episode={selected_episode} onClick={index => {
                    xplayer_ref.current?.play_episode(index + 1)
                    set_selected_episode(index + 1)
                }}/>
            </div>
        </div>
    )
}

export {VideoPlayer}
