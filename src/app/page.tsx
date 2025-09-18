"use client"

import React, {useState, useRef, useEffect} from "react"
import {SearchWordInput,URLInput} from "@/components/FormInputs"
import {SelectCMSType,ShowCMSResource} from "@/components/CMSResource/GetCMSResource"
import {ScrollToTopButton,ScrollToBottomButton} from "@/components/ScrollButton"
import {ShowHistoryResource,HistoryButton} from "@/components/CMSResource/ShowHistoryResource"
import {ResourceListItem} from "@/infra/types"
import {VideoPlayer} from "@/components/Player/Player"
import {string_icons} from "@/infra/custom_ui_constants"
import {throttle} from "@/infra/performance"


export default function Home() {

    const [cms_api,set_cms_api] = useState("")
    const [current_type_id,set_current_type_id] = useState<string | null>(null)
    const [search_word,set_search_word] = useState("")
    const [selected_resource,set_selected_resource] = useState<ResourceListItem | null>(null)
    const [select_count,set_select_count] = useState(0)

    const [show_history,set_show_history] = useState(false)
    const scroll_y_when_clicking_history_ref= useRef(0)
    const [history_resource,set_history_resource] = useState<ResourceListItem[]>([])

    const selected_resource_ref = useRef(selected_resource)
    useEffect(() => {
        selected_resource_ref.current = selected_resource
    }, [selected_resource])

    useEffect(() => {
        const url = new URL(location.href)
        set_cms_api(url.searchParams.get("api") || localStorage.getItem("cms_api") || "")
        set_history_resource(JSON.parse(localStorage.getItem("history_resource") || "[]"))
        history.replaceState(null,"","/")
    }, [])

    useEffect(() => {
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js').then(() => {
                console.log('Service worker is registered successfully.')
            }).catch(err => {console.error('Failed to register Service worker.', err)})}
    }, [])

    useEffect(() => {
        set_current_type_id(null)
        set_search_word("")
        cms_api !== ""  && localStorage.setItem("cms_api",cms_api)
    }, [cms_api])

    useEffect(() => {
        localStorage.setItem("history_resource",JSON.stringify(history_resource))
    }, [history_resource])


    function update_history_resource(selected_resource: ResourceListItem, updates?: Partial<ResourceListItem>) {
        if (updates){
            Object.assign(selected_resource, updates)
        }
        set_history_resource(history_resource => {

            history_resource = [selected_resource,...history_resource.filter(item => {
                return item.vod_name !== selected_resource.vod_name
            })]
            return history_resource
        })
    }

    return (
        <>
            {!show_history &&
                <URLInput default_url={cms_api} callback={set_cms_api} description="CMS API"/>
            }
            {cms_api && /https?:\/\/.*/.test(cms_api) &&
                <div className={!show_history ? "block" :"hidden"}>
                    <SelectCMSType url={cms_api} callback={set_current_type_id}/>
                    <SearchWordInput callback={set_search_word}/>
                    <ShowCMSResource
                        url={cms_api} current_type_id={current_type_id} word={search_word}
                        callback={selected_resource => {
                            for (const item of history_resource){
                                if (item.vod_name == selected_resource.vod_name) {
                                    selected_resource._start_episode = item._start_episode
                                    selected_resource._start_time = item._start_time
                                }
                            }
                            set_selected_resource(selected_resource)
                            set_select_count(select_count => select_count + 1)
                            update_history_resource(selected_resource)
                        }}
                    />
                </div>
            }

            {show_history &&
                <>
                    <div className="select-none text-3xl text-center font-bold">
                        <button className="hover:cursor-pointer"
                                onClick={() => {
                                    set_history_resource([])
                                }}
                        >
                            {string_icons.del}
                        </button>
                    </div>
                    <ShowHistoryResource
                        callback={selected_resource=> {
                            set_selected_resource(selected_resource)
                            set_select_count(select_count => select_count + 1)
                            update_history_resource(selected_resource)
                        }}
                        delete_callback={
                            selected_resource => {
                                set_history_resource(history_resource => {
                                    return history_resource.filter(item => {
                                        return item.vod_name !== selected_resource.vod_name
                                    })
                                })
                            }}
                        history_resource={history_resource}
                    />
                </>
            }

            {selected_resource &&
                <div className="fixed w-full top-10">
                    <VideoPlayer
                        src={selected_resource._vod_play_url_list}
                        title={selected_resource.vod_name}
                        description={`${selected_resource._vod_description}`}
                        poster_url={selected_resource.vod_pic}
                        start_episode={selected_resource._start_episode}
                        start_time={selected_resource._start_time}
                        counter={select_count}
                        callback={
                            xplayer => {
                                const throttled_update_history_resource = throttle(update_history_resource,5000)
                                xplayer.on("timeupdate", () => {
                                    if (selected_resource_ref.current){
                                        selected_resource_ref.current._start_episode = xplayer.selected_episode
                                        selected_resource_ref.current._start_time = xplayer.get_current_time()
                                        throttled_update_history_resource(selected_resource_ref.current)
                                    }
                                })
                            }
                        }
                    />
                </div>
            }
            <HistoryButton callback={() => {
                set_show_history(show_history => {
                    if (!show_history){
                        scroll_y_when_clicking_history_ref.current = scrollY
                        setTimeout(() => window.scrollTo(0,0))
                    }
                    else {
                        setTimeout(() => window.scrollTo(0,scroll_y_when_clicking_history_ref.current))
                    }
                    return !show_history
                })
            }}
            />
            <ScrollToTopButton />
            <ScrollToBottomButton />
        </>
    )
}
