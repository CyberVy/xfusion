import {
    ResourceType,
    ResourceListItem, _ResourceCategoryClassItem
} from "./types"

import {sort_resource_list_by_score} from "@/infra/filter"


async function normalize_cms_fetch(url: string, init?: RequestInit) {
    const response = await fetch(url,init)
    if (response.status != 200){
        return response
    }
    const json = await response.json()
    if (json.data){

        const _page = json.page
        json.page = _page.pageIndex
        json.pagecount = _page.pagecount
        json.limit = _page.pagesize
        json.total = Number(_page.recordcount)

        const json_class:_ResourceCategoryClassItem[] = []
        const json_list = json.data
        for (const item of json.list){
            const type_id: number = item.list_id || ""
            const type_name: string = item.list_name || ""
            json_class.push({type_id:type_id,type_name:type_name,type_pid: 0})
        }

        json.class = json_class

        for (const item of json_list){
            item.vod_play_url = item.vod_url
            item.type_name = item.list_name
            item.vod_blurb = item.vod_content
        }
        json.list = json_list

        return new Response(JSON.stringify(json),{status:200,headers:response.headers})
    }

    return new Response(JSON.stringify(json),{status:200,headers:response.headers})
}

async function fetchCMSCategory (url: string) {
    const response = await normalize_cms_fetch(url)
    const json = await response.json()
    const type_name_list: string[] = []
    const type_id_list: string[] = []
    const type_dict: Record<string, string> = {}

    if (json["class"]){
        for (const item of json["class"]) {
            const type_name = item["type_name"]
            const type_id = item["type_id"]
            type_name_list.push(type_name)
            type_id_list.push(type_id.toString())
            type_dict[type_name] = type_id.toString()
        }
        return {type_name_list,type_id_list,type_dict}
    }
    return {type_name_list,type_id_list,type_dict}
}

async function fetchCMSResource(url: string,type_id: string | undefined | null,page: number,word: string | undefined | null){
    const _url = new URL(url)
    _url.searchParams.set("ac", "videolist")
    type_id && _url.searchParams.set("t", type_id)
    page && _url.searchParams.set("pg", page.toString())
    word && _url.searchParams.set("wd",word)
    const response = await normalize_cms_fetch(_url.href)
    if (response.status === 200){
        const json: ResourceType = await response.json()
        if (json.list){
            for (const item of json.list){
                item._vod_play_url_list = get_resource_list_item_episodes(item.vod_play_url)
                item._vod_description = get_resource_list_item_description(item)
                item._start_episode = 1
                item._start_time = 0
            }
            sort_resource_list_by_score(json.list)
            return json
        }
    }
    else {
        throw Error(`${response.status}, when fetching resource.`)
    }
}

function get_resource_list_item_description(resource_list_item: ResourceListItem){
    let r = ""
    if (resource_list_item.vod_name){
        r += `${resource_list_item.vod_name}\n`
    }
    if (resource_list_item.type_name){
        r += `${resource_list_item.vod_year || "unknown"}, ${resource_list_item.type_name}`
        if (resource_list_item._vod_play_url_list.length > 1){
            r += `, EP: ${resource_list_item._vod_play_url_list.length}\n`
        } else {
            r += "\n"
        }
    }
    if (resource_list_item.vod_blurb && resource_list_item.vod_blurb !== resource_list_item.vod_name){
        r += `${resource_list_item.vod_blurb} ...\n`
    }
    if (resource_list_item.vod_director){
        r += `Director: ${resource_list_item.vod_director}\n`
    }
    if (resource_list_item.vod_actor){
        r += `Actor: ${resource_list_item.vod_actor}\n`
    }
    if (resource_list_item._vod_play_url_list[0]){
        try {
            r += `Source: ${new URL(resource_list_item._vod_play_url_list[0]).hostname}`
        }
        catch {}
    }
    return r
}

function get_resource_list_item_episodes(s: string){
    const r = []
    for (const item of s.split("$")){
        const match_result = item.match(/https?:\/\/\S+\.m3u8/g)
        if (match_result){
            r.push(match_result[0])
        }
    }
    return r
}

export {
    fetchCMSCategory,
    fetchCMSResource,
}
