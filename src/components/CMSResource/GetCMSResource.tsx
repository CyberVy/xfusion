'use client'

import React, { useState, useEffect, useRef} from "react"
import {ListToButtons} from "@/components/ListToBox"
import {LabeledImage} from "@/components/LabeledImage"
import {fetchCMSCategory,fetchCMSResource} from "@/infra/cms_lib"
import {throttle} from "@/infra/performance"
import {ResourceType,ResourceListItem,SelectCMSTypeInputs,ShowCMSResourceInputs} from "@/infra/types"
import {highlight} from "@/infra/custom_ui_constants"


function SelectCMSType({url,callback}: SelectCMSTypeInputs){
    const [type_name_list,set_type_name_list] = useState<string[] | null>(null)
    const [type_id_list,set_type_id_list] = useState([""])
    const [type_dict,set_type_dict] = useState({} as Record<string, string | number>)

    const [loading,set_loading] = useState(false)
    const [error, set_error] = useState<string | null>(null)

    useEffect(() => {
        set_type_name_list(null)
        set_type_id_list([""])
        set_type_dict({})
        set_loading(true)
        set_error(null)
        fetchCMSCategory(url).then((result) => {
            set_type_name_list(result.type_name_list)
            set_type_id_list(result.type_id_list)
            set_type_dict(result.type_dict)
            set_error(null)
        })
            .catch((error) => set_error(error.message))
            .finally(() => {
                set_loading(false)
            })
    }, [url])

    return (
        <>
            <div className="text-center font-bold">
                {loading && <p>Loading Category...</p>}
                {error && <p>Error: {error}</p>}
            </div>
            {type_name_list && <ListToButtons list={type_name_list} callback={current_type_name => {
                callback?.(current_type_name ? type_dict[current_type_name].toString() : null)
            }}/>}
        </>
    )
}

function ShowCMSResource({url,current_type_id,word,callback}: ShowCMSResourceInputs){
    const [current_resource, set_current_resource] = useState<ResourceType | null>(null)
    const [current_page,set_current_page] = useState(1)
    const [loading,set_loading] = useState(false)
    const [error, set_error] = useState<string | null>(null)
    const [vector_mode,set_vector_mode] = useState(false)
    const [random_mode,set_random_mode] = useState(false)
    const vector_mode_grid_tailwind_cls = "select-none grid grid-cols-1 gap-0 p-0"
    const default_grid_tailwind_cls = "select-none grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-0 p-0"

    const current_resource_ref = useRef(current_resource)
    useEffect(() => {
        current_resource_ref.current = current_resource
    }, [current_resource])

    const current_type_id_ref = useRef(current_type_id)
    useEffect(() => {
        current_type_id_ref.current = current_type_id
    }, [current_type_id])

    const word_ref = useRef(word)
    useEffect(() => {
        word_ref.current = word
    }, [word])

    const url_ref = useRef(url)
    useEffect(() => {
        url_ref.current = url
    }, [url])

    const random_mode_ref = useRef(random_mode)
    useEffect(() => {
        random_mode_ref.current = random_mode
    }, [random_mode])

    useEffect(() => {
        set_vector_mode(false)
    }, [url])

    useEffect(() => {
        set_loading(true)
        set_error(null)
        set_current_resource(null)
        set_current_page(1)
        if (url){
            fetchCMSResource(url,current_type_id,1,word)
                .then(result => {
                    set_current_resource(current_resource => {
                        if (result?.list?.length){
                            console.log(result.list[0]["type_name"],1,result.pagecount,current_type_id,word)
                            set_error(null)
                            return result
                        }
                        else {
                            set_error(null)
                            return current_resource
                        }
                    })
                })
                .catch(error => set_error(error.message))
                .finally(() => {
                    set_loading(false)
                })
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    },[url,current_type_id,word])

    useEffect(() => {

        // a closure for listener, use reference to get the latest states
        const update_resource = throttle(() => {
            console.log(">>>")
            set_current_page(current_page => {
                if (!current_resource_ref.current){
                    return current_page
                }
                if (current_page >= current_resource_ref.current.pagecount){
                    console.log(`${current_page} is the last page.`)
                    return current_page
                }
                set_loading(true)
                set_error(null)

                // get the latest states via useRef before fetch
                url = url_ref.current
                current_type_id = current_type_id_ref.current
                if (!random_mode_ref.current){
                    current_page += 1
                }
                else {
                    current_page = Math.floor(Math.random() * current_resource_ref.current.pagecount) + 1
                }
                word = word_ref.current

                fetchCMSResource(url,current_type_id,current_page,word)
                    .then(result => {
                        set_current_resource(current_resource => {
                            if (result?.list?.length){
                                // the states can change when waiting the promise of fetch
                                // here we don't want the latest states
                                // we just want to know the arguments for fetch.
                                console.log(result.list[0]["type_name"],current_page,result.pagecount,current_type_id,word)
                                if (current_resource?.list?.length){
                                    result.list = [...current_resource.list,...result.list]
                                }
                                set_error(null)
                                return result
                            }
                            else {
                                set_error(null)
                                return current_resource
                            }
                        })
                    })
                    .catch(error => set_error(error.message))
                    .finally(() => {
                        set_loading(false)
                    })
                return current_page
            })
        },500)

        let scrollTop = 0
        function handle_scroll_to_bottom(){

            if (scrollY + document.documentElement.clientHeight >= document.documentElement.scrollHeight && scrollTop < scrollY){
                update_resource()
            }
            scrollTop = scrollY
        }
        window.addEventListener("scroll",handle_scroll_to_bottom)
        return () => {
            window.removeEventListener("scroll",handle_scroll_to_bottom)
        }
    }, [])

    if (current_resource && current_resource.list.length > 0){
        return (
            <>
                <div className="select-none mx-4">
                    <button
                        className={`mr-1 text-center text-zinc-300 font-bold border hover:cursor-pointer px-3 rounded rounded-xl`}
                        onClick={() => set_vector_mode(!vector_mode)}
                    >
                        <span
                            className={`${vector_mode ? highlight : "text-black dark:text-white"}`}>
                            Vector Mode
                        </span>
                    </button>

                    <button
                        className={`mr-1 text-center text-zinc-300 font-bold border hover:cursor-pointer px-3 rounded rounded-xl`}
                        onClick={() => set_random_mode(!random_mode)}
                    >
                        <span
                            className={`${random_mode ? highlight : "text-black dark:text-white"}`}>
                            Random Mode
                        </span>
                    </button>

                </div>
                <div className="select-none text-center font-bold">
                    <p>{!current_type_id ? "" : current_resource.list[0]["type_name"]}</p>
                    <p>{`${current_page}/${current_resource.pagecount}(${current_resource.total})`}</p>
                </div>
                <div className={vector_mode ? vector_mode_grid_tailwind_cls : default_grid_tailwind_cls}>
                    {current_resource.list.map((v: ResourceListItem,index) => {
                        const score = v.vod_douban_score != "0.0" ? v.vod_douban_score : v.vod_score
                        let label_background_color = Number(score) >= 8.0 ? "bg-red-700/90" : Number(score) >= 6.0 ? "bg-yellow-700/90" : "bg-teal-700/90"
                        label_background_color = Number(score) == 0.0 ? "" : label_background_color
                        return (
                            <LabeledImage
                                key={index} alt={v.vod_name}
                                label={score != "0.0" ? score : ""}
                                label_background_color={label_background_color}
                                src={v.vod_pic ? v.vod_pic : undefined}
                                description={v._vod_description}
                                onClickImage={() => {
                                    callback?.(v)
                                }}
                            />
                        )})}
                </div>

                <div className="select-none text-center font-bold text-3xl">
                    <p>{`${current_page}/${current_resource.pagecount}(${current_resource.total})`}</p>
                    <br/>
                    <p>Scroll down to the next page.</p>
                    <p>▼</p>
                    <br/>
                    {loading && <p>Loading...</p>}
                    {error && <p>Error: {error}</p>}
                    <br/>
                </div>
            </>
        )
    } else {
        return (
            <div className="select-none text-center font-bold">
                {!loading ? <p>Empty</p> : <p>Loading...</p>}
            </div>
        )
    }
}


export {
    SelectCMSType,
    ShowCMSResource
}
