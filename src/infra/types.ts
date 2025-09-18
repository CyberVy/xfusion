import type {XPlayer} from "@/infra/player.client"

export type ResourceListItem = {
    vod_id: number
    type_id: number
    type_id_1: number
    group_id: number
    vod_name: string
    vod_sub: string
    vod_en: string
    vod_status: number
    vod_letter: string
    vod_color: string
    vod_tag: string
    vod_class: string
    vod_pic: string // video post url
    vod_pic_thumb: string
    vod_pic_slide: string,
    vod_pic_screenshot: string
    vod_actor: string
    vod_director: string
    vod_writer: string
    vod_behind: string
    vod_blurb: string
    vod_remarks: string
    vod_pubdate: string
    vod_total: number
    vod_serial: string
    vod_tv: string
    vod_weekday: string
    vod_area: string
    vod_lang: string
    vod_year: string
    vod_version: string
    vod_state: string
    vod_author: string
    vod_jumpurl: string
    vod_tpl: string
    vod_tpl_play: string
    vod_tpl_down: string
    vod_isend: number
    vod_lock: number
    vod_level: number
    vod_copyright: number
    vod_points: number
    vod_points_play: number
    vod_points_down: number
    vod_hits: number
    vod_hits_day: number
    vod_hits_week: number
    vod_hits_month: number
    vod_duration: string
    vod_up: number
    vod_down: number
    vod_score: string
    vod_score_all: number
    vod_score_num: number
    vod_time: string
    vod_time_add: number
    vod_time_hits: number
    vod_time_make: number
    vod_trysee: number
    vod_douban_id: number
    vod_douban_score: string
    vod_reurl: string
    vod_rel_vod: string
    vod_rel_art: string
    vod_pwd: string
    vod_pwd_url: string
    vod_pwd_play: string
    vod_pwd_play_url: string
    vod_pwd_down: string
    vod_pwd_down_url: string
    vod_content: string
    vod_play_from: string
    vod_play_server: string
    vod_play_note: string
    vod_play_url: string // video urls
    vod_down_from: string
    vod_down_server: string
    vod_down_note: string
    vod_down_url: string
    vod_plot: number
    vod_plot_name: string
    vod_plot_detail: string
    type_name: string
    _vod_play_url_list: string[]
    _vod_description: string
    _start_episode: number
    _start_time: number
}
export type ResourceType = {
    code: number
    msg: string
    page: number
    pagecount: number
    limit: string
    total: number
    list: ResourceListItem[]
}
export type _ResourceCategoryClassItem = {
    type_id: number
    type_pid: number
    type_name: string
}
export type _ResourceCategoryType = {
    code: number
    msg: string
    page: number
    pagecount: number
    limit: string
    total: string
    list : ResourceListItem
    class: _ResourceCategoryClassItem[]
}
export type SelectCMSTypeInputs = {
    url: string
    callback?: (item: string | null) => void
}
export type ShowCMSResourceInputs = {
    url: string
    current_type_id?: string | null
    word?: string | null
    callback?: (
        selected_resource: ResourceListItem
    ) => void
}
export type HistoryButtonInputs = {
    callback?: () => void
}
export type ShowHistoryResourceInputs = {
    history_resource: ResourceListItem[]
    callback?: (selected_resource: ResourceListItem) => void
    delete_callback?: (selected_resource: ResourceListItem) => void
}
export type VideoPlayerInputs = {
    src: string[]
    title?: string
    description?: string
    poster_url?: string
    start_episode?: number
    start_time?: number
    counter?: number
    callback?: (xplayer:XPlayer) => void
}
export type EpisodeInputBoxInputs = {
    src: string[]
    selected_episode: number
    onClick?: (index:number) => void
}
export type XPlayerOptionsType = {
    fluid?: boolean
    controlBar?:{
        skipButtons?:{
            forward?: number
            backward?: number
        }
    }
    enableSmoothSeeking?: boolean
    playsinline?: boolean
    playbackRates?: number[]
    autoplay?: boolean
    controls?: boolean
    sources?: Record<string, string>[]
    width?: number
    height?: number
    client_width_ratio?: number
    client_height_ratio?: number
}
export type CoverImageOptions = {
    width?: number
    height?: number
    background?: string
    color?: string
    fontSize?: number
    fontFamily?: string
}
export type XPlayerEventType =
    "timeupdate" | "loadedmetadata" | "loadeddata" |
    "pause" | "play" | "playing" | "canplay" | "ready" | "canplaythrough" | "ended" |
    "seeking" | "seeked" | "waiting" | "dispose" |
    "enterpictureinpicture" | "leavepictureinpicture" |
    "enterFullWindow" | "exitFullWindow" |
    "tap" | "error" | "stalled"
