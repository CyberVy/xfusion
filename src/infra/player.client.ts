import Player from "video.js/dist/types/player"
import videojs from "video.js"

import {
    fetch_m3u8_playlist,
    get_m3u8_ad_breakpoint, normalize_m3u8_playlist,
    remove_m3u8_ad_chunk
} from "@/infra/m3u8_lib"

import {XPlayerOptionsType,XPlayerEventType} from "./types"


/**
 * video element:
 * ```html
 * <video ref={video_ref} className="video-js"/>
 * ```
 *
 * The audio element plays a short silent loop to keep the Media Session active.
 * Some browsers (especially on iOS) require an immediate media response after a user gesture
 * without an active media element, Media Session actions like play/pause may fail.
 * ```html
 * <audio
 *     ref={blank_audio_ref}
 *     loop
 *     src={generate_silent_wav_base64(5)}
 * >
 *   </audio>
 * ```
 */
export class XPlayer{
    public _player: Player
    public audio_element: HTMLAudioElement | null
    public options: XPlayerOptionsType | null

    public ad_threshold: number
    public m3u8_chunk_proxy: string
    public enable_remove_ad: boolean
    private waiting_for_remove_ad: boolean
    private waiting_for_m3u8_proxy: boolean
    public ignore_error: boolean
    public retry_interval: number
    private play_counter: number
    private retry_counter: number
    public ios_network_checker: number
    public ios_network_check_interval: number

    // works with "ended" listener
    public auto_play_next_episode_callback: (() => void) | null
    // works with media session "nexttrack" listener
    public media_session_next_track_callback: (() => void) | null
    // works with media session "previoustrack" listener
    public media_session_previous_track_callback: (() => void) | null
    // works with media session "seekto" listener
    public media_session_seek_to_callback: (() => void) | null
    // executed as the final part of play_episode
    public play_episode_callback: (() => void) | null

    // caution: the properties below will be re-init calling init()
    public source: string[]
    public selected_episode: number
    // works with a "timeupdate" listener
    private _skip_ad_callback: (() => void) | null
    // workers with a "loadedmetadata" listener
    private _time_tracker_callback: (() => void) | null
    public media_title: string
    public poster_url: string

    public init(){
        this.source = []
        this.selected_episode = 1
        this.media_title = "XPlayer"
        this.poster_url = ""

        this.clear_skip_ad_legacy()
        if (this.options?.autoplay){
            this.set_autoplay(true)
        }
        this.remove_time_tracker()
    }

    public normalize_options(options?: XPlayerOptionsType | null){
        if (!options){
            return
        }

        if (options.client_width_ratio && options.client_height_ratio){

            const client_width_ratio = options.client_width_ratio
            const client_height_ratio = options.client_height_ratio
            options.width = window.document.documentElement.clientWidth * client_width_ratio
            options.height = window.document.documentElement.clientHeight * client_height_ratio

            const resize_callback = () => {
                try {
                    if (this.get_width() || this.get_height()){
                        this.set_width(window.document.documentElement.clientWidth * client_width_ratio)
                        this.set_height(window.document.documentElement.clientHeight * client_height_ratio)
                    }
                    else {
                        console.log(`Removed XPlayer resize listener(${client_width_ratio}/${client_height_ratio}).`)
                        window.removeEventListener("resize",resize_callback)
                    }
                }
                catch (error){
                    window.removeEventListener("resize",resize_callback)
                    console.log(`${(error as Error).message}\nRemoved XPlayer resize listener(${client_width_ratio}/${client_height_ratio}).`)
                }
            }
            window.addEventListener("resize", resize_callback)
            console.log(`Set XPlayer resize listener(${client_width_ratio}/${client_height_ratio}).`)

            delete options.client_width_ratio
            delete options.client_height_ratio
        }
        return options
    }

    constructor(video_element: HTMLVideoElement | string, audio_element?: HTMLAudioElement | null, options?: XPlayerOptionsType | null) {

        this._player = videojs(video_element,this.normalize_options(options))
        this.audio_element = audio_element || null
        this.options = options || null
        this.ad_threshold = 30
        this.m3u8_chunk_proxy = ""
        this.enable_remove_ad = true
        this.waiting_for_remove_ad = false
        this.waiting_for_m3u8_proxy = false
        this.ignore_error = false
        this.retry_interval = 3000
        this.play_counter = 0
        this.retry_counter = 0
        this.ios_network_checker = 0
        this.ios_network_check_interval = 6000

        this.source = []
        this.selected_episode = 1
        this._skip_ad_callback = null
        this.auto_play_next_episode_callback = null
        this.media_session_next_track_callback = null
        this.media_session_previous_track_callback = null
        this.media_session_seek_to_callback = null
        this.play_episode_callback = null
        this._time_tracker_callback = null
        this.media_title = "XPlayer"
        this.poster_url = ""

        this.set_handlers_for_media_session()
        this.set_auto_play_next_episode()
        this.set_error_listener()
        this.set_ios_network_checker()
    }

    public paused(){
        return this._player.paused()
    }

    public seeking(){
        return this._player.seeking()
    }

    public play(){
       return this._player.play()
    }

    public pause(){
        return this._player.pause()
    }

    public get_autoplay(){
        return this._player.autoplay() as boolean
    }

    public set_autoplay(autoplay: boolean){
        this._player.autoplay(autoplay)
    }

    /**
     * Use with caution in React: calling this will dispose the player
     * and remove the underlying DOM element, which may break React rendering.
     *
     * Recommended usage:
     * ```ts
     * const [showPlayer, setShowPlayer] = useState(true)
     * useEffect(() => { xplayer.dispose(); setShowPlayer(false) }, [])
     * ```
     */
    public dispose(){
        this._player.dispose()
    }

    public one(event_name: XPlayerEventType,callback: () => void){
        return this._player.one(event_name,callback)
    }

    public on(event_name: XPlayerEventType,callback: () => void){
        return this._player.on(event_name,callback)
    }

    public off(event_name: XPlayerEventType,callback: () => void){
        return this._player.off(event_name,callback)
    }

    public get_default_playback_rate(){
        return this._player.defaultPlaybackRate() as number
    }

    public set_default_playback_rate(rate: number){
        return this._player.defaultPlaybackRate(rate) as void
    }

    public get_playback_rate(){
        return this._player.playbackRate() as number
    }

    public set_current_time(time: number){
        return this._player.currentTime(time) as void
    }

    public set_playback_rate(rate: number){
        return this._player.playbackRate(rate) as void
    }

    public get_duration(){
        return this._player.duration() as number
    }

    public set_error(message_or_code: string | number){
        this._player.error(message_or_code)
    }

    public get_error(){
        return this._player.error()
    }

    public get_current_time(){
        return this._player.currentTime() as number
    }

    public get_width(){
        return this._player.width() as number
    }

    public get_height(){
        return this._player.height() as number
    }

    public set_width(width: number){
        this._player.width(width)
    }

    public set_height(height: number){
        this._player.height(height)
    }

    public get_video_width(){
        return this._player.videoWidth()
    }

    public get_video_height(){
        return this._player.videoHeight()
    }

    public set_source(source: string[]){
        this.init()
        this.source = source
    }

    public set_episode(episode_number: number){
        if (this.check_episode(episode_number)){
            this.selected_episode = episode_number
        }
    }

    public check_episode(episode_number: number){
        if (episode_number >=1 && episode_number <= this.source.length){
            return true
        }
        else {
            console.log(`Invalid episode: ${episode_number}, which is not in [1 - ${this.source.length}]`)
            return false
        }
    }

    public set_poster(poster?: string){
        poster = poster ? poster : this.poster_url
        if (poster){
            this.poster_url = poster
            this._player.poster(poster)
        }
    }

    public get_poster(){
        return this.poster_url
    }

    public play_selected_episode(){
        this.play_episode(this.selected_episode)
    }

    public async play_episode(episode_number: number, start_time = 0){

        if (this.check_episode(episode_number)){

            console.log(`Playing: ${this.media_title}, episode: ${episode_number}`)

            this.play_counter += 1
            const counter = this.play_counter
            this.set_episode(episode_number)

            if (this.enable_remove_ad){
                this.pause()
                this._player.addClass('vjs-waiting')

                // if removing ad failed, trigger the error listener of the player,
                // and throw an error to stop play_episode function,
                // the error listener will retry playing
                await this.remove_ad()
                    .catch(error => {
                        if (counter == this.play_counter){
                            this.set_error(`XPlayer remove ad error: ${error.message}`)
                        }
                        throw error
                    })
            }
            else {
                this.clear_skip_ad_legacy()
            }

            if (this.m3u8_chunk_proxy){
                this.pause()
                this._player.addClass('vjs-waiting')

                // if proxy failed, trigger the error listener of the player,
                // and throw an error to stop play_episode function,
                // the error listener will retry playing
                await this.start_proxy()
                    .catch(error => {
                        if (counter == this.play_counter){
                            this.set_error(`XPlayer proxy error: ${error.message}`)
                        }
                        throw error
                    })
            }
            else {
                this.clear_proxy_legacy()
            }

            // only play the latest source
            if (counter == this.play_counter){
                this.set_time_tracker(start_time)
                this.set_default_playback_rate(this.get_playback_rate())
                this.set_media_session_metadata(this.media_title,this.poster_url)
                this._player.src(this.source[episode_number - 1])
                this.play_episode_callback?.()
            }
        }
    }

    public clear_skip_ad_legacy(){
        if (this._skip_ad_callback){
            this.off("timeupdate",this._skip_ad_callback)
            this._skip_ad_callback = null
            console.log("Cleared the legacy of skip ad callback.")
        }
    }

    /**
     * this is a fallback function for remove_ad, also is used to test ad detection,
     * it works in most of the environments,
     * however, if the ad chunk is dynamically generated, it will not work properly.
     */
    public async skip_ad(){

        this.clear_skip_ad_legacy()

        const m3u8_text_list = await fetch_m3u8_playlist(this.source[this.selected_episode - 1])
        if (!m3u8_text_list.length){
            console.error("Not found any available play list.")
            return
        }
        const m3u8_text = m3u8_text_list[0]
        const ad_breakpoint = get_m3u8_ad_breakpoint(m3u8_text,this.ad_threshold)

        if (!ad_breakpoint.length){
            return
        }
        const callback = () => {
            const current_time = this.get_current_time()
            for (const [ad_starting_time,ad_ending_time] of ad_breakpoint){
                if (ad_ending_time > current_time! && current_time! > ad_starting_time){
                    console.log(`Skipped AD: ${current_time}, [${ad_starting_time} - ${ad_ending_time}]`)
                    // The +0.017 second offset prevents frame-precision issues that could cause the player
                    // to get stuck inside the ad segment.
                    this.set_current_time(ad_ending_time + 0.017)
                }
            }
        }
        this.on("timeupdate",callback)
        this._skip_ad_callback = callback
    }

    /**
     * remove ad in m3u8 playlist via service worker,
     * if service worker is not available, automatically use skip_ad instead
     */
    public async remove_ad(){

        this.clear_skip_ad_legacy()

        if (!("serviceWorker" in navigator)){
            console.warn("Service worker is not available, so remove ad can't work, have automatically used skip ad instead.")
            this.skip_ad()
            return
        }
        this.clear_proxy_legacy()
        this.waiting_for_remove_ad = true
        const m3u8_text_list = await fetch_m3u8_playlist(this.source[this.selected_episode - 1]).finally(() => this.waiting_for_remove_ad = false)
        if (!m3u8_text_list.length){
            console.error("Not found any available play list.")
            return
        }
        // Todo: Prefer using a resolution selector when multiple resolutions are available
        const m3u8_text = m3u8_text_list[0]

        navigator.serviceWorker.controller?.postMessage({
            type: "update_removed_ad_playlist_storage",
            target_url: decodeURI(new URL(this.source[this.selected_episode - 1]).href),
            override_content: {
                content:remove_m3u8_ad_chunk(m3u8_text,this.ad_threshold),
                headers: {
                    "Content-Type": "application/vnd.apple.mpegurl",
                    "Generated-By":"remove_ad"
                }
            }
        })
    }

    public set_ad_threshold(threshold: number){
        this.ad_threshold = threshold
    }

    public set_media_session_metadata(title: string, poster_url?: string){
        this.media_title = title
        this.poster_url = poster_url || ""
        navigator.mediaSession.metadata = new MediaMetadata({
            title: `${title}`,
            artist: `${this.source.length > 1 ? `EP: ${this.selected_episode}/${this.source.length}` : ""}`,
            artwork: [{src: poster_url || ""}]
        })
    }

    public next(){
        if (this.check_episode(this.selected_episode + 1)){
            this.set_episode(this.selected_episode + 1)
            this.play_episode(this.selected_episode)
            return true
        }
        return false
    }

    public previous(){
        if (this.check_episode(this.selected_episode - 1)){
            this.set_episode(this.selected_episode - 1)
            this.play_episode(this.selected_episode)
            return true
        }
        return false
    }

    public async send_immediate_media_response(){
        if (this.audio_element){
            return await this.audio_element.play().then(() => this.audio_element?.pause())
        }
        else {
            console.warn("No available audio element, please set an audio element to send an immediate media response.")
        }
    }

    /**
     * Register Media Session action handlers (play, pause, next, previous).
     * This enables media control integration with lock screen and notification UI on mobile devices (e.g., iOS).
     */
    public set_handlers_for_media_session(){
        if ('mediaSession'in navigator) {
            navigator.mediaSession.setActionHandler("nexttrack", () => {
                if (this.check_episode(this.selected_episode + 1)){
                    this.send_immediate_media_response().then(() => {
                        this.next()
                        this.media_session_next_track_callback?.()
                    })
                }
            })
            navigator.mediaSession.setActionHandler("previoustrack", () => {
                if (this.check_episode(this.selected_episode - 1)){
                    this.send_immediate_media_response().then(() => {
                        this.previous()
                        this.media_session_previous_track_callback?.()
                    })
                }
            })
            navigator.mediaSession.setActionHandler("seekto", (details) => {
                if (details.seekTime){
                    this.set_current_time(details.seekTime)
                    this.media_session_seek_to_callback?.()
                }
            })
            navigator.mediaSession.setActionHandler("play", () => {
                this.play()
            })
            navigator.mediaSession.setActionHandler("pause", () => {
                this.pause()
            })
        }
    }

    public set_auto_play_next_episode_callback(callback: () => void){
        this.auto_play_next_episode_callback = callback
    }

    public set_auto_play_next_episode(){

        this.on("ended", () => {
            if (this.check_episode(this.selected_episode + 1)){
                this.send_immediate_media_response().then(() => {
                    this.next()
                    this.auto_play_next_episode_callback?.()
                })
            }
            else {
                console.log(`The current episode is ${this.selected_episode}, which is the last episode, can not navigate to the next episode.`)
            }
        })
    }

    public set_media_session_next_track_callback(callback: () => void){
        this.media_session_next_track_callback = callback
    }

    public set_media_session_previous_track_callback(callback: () => void){
        this.media_session_previous_track_callback = callback
    }

    public set_media_session_seek_to_callback(callback: () => void){
        this.media_session_seek_to_callback = callback
    }

    public set_error_listener(){
        const error_callback = async () => {
            const err = this.get_error()
            // Check these variable to prevent old/obsolete error events from interfering with new playback sessions,
            // which is important for concurrent/async scenarios
            this.retry_counter += 1
            const retry_counter = this.retry_counter
            const play_counter = this.play_counter
            if (err){
                const time = this.get_current_time()
                if (retry_counter > 1) {
                    await new Promise(resolve => setTimeout(resolve,this.retry_interval))
                }
                // if ignore_error is falsy and the error and the source are the latest, retry
                if (!this.ignore_error && retry_counter == this.retry_counter && play_counter == this.play_counter){
                    console.error(`Reconnecting for ${err.message}.`)
                    this.send_immediate_media_response().then(() => this.play_episode(this.selected_episode,time))
                }
            }
        }
        this.on("error",error_callback)
    }

    public set_ios_network_checker(){
        const user_agent = navigator.userAgent.toLowerCase()
        if (!(user_agent.includes("iphone") || user_agent.includes("ipad"))){
            return
        }
        // make sure that only 1 ios network checker is working
        if (this.ios_network_checker){
            return
        }
        const callback = () => {
            if (this.paused() || this.seeking()){
                return
            }
            const player_time = this.get_current_time()
            setTimeout(() => {
                const current_player_time = this.get_current_time()
                const interval = current_player_time - player_time
                // this.get_current_time() can be 0 when loading video, which causes that the interval is always 0.
                if (interval == 0 && current_player_time != 0){
                    console.error("Reconnecting for bad network.")
                    this.send_immediate_media_response().then(() => this.play_episode(this.selected_episode,current_player_time))
                }
            },this.ios_network_check_interval)
        }
        this.ios_network_checker = window.setInterval(callback,this.ios_network_check_interval)
    }

    public remove_ios_network_checker(){
        if (this.ios_network_checker){
            window.clearInterval(this.ios_network_checker)
        }
    }

    public set_time_tracker(time: number){
        this.remove_time_tracker()
        if (time > 0){
            this._time_tracker_callback = () => {
                this.set_current_time(time)
                this._time_tracker_callback = null
                console.log(`Time tracker is consumed.`)
            }
            this.one("loadedmetadata", this._time_tracker_callback)
            console.log(`Time tracker(${time}) is set.`)
        }
    }

    public remove_time_tracker(){
        if (this._time_tracker_callback){
            this.off("loadedmetadata",this._time_tracker_callback)
            console.log(`Time tracker is removed.`)
        }
    }

    public set_m3u8_chunk_proxy(proxy:string){
        this.m3u8_chunk_proxy = proxy
    }

    public async start_proxy(){

        if (!("serviceWorker" in navigator)){
            console.warn("Service worker is not available, so the proxy can't work.")
            return
        }

        this.clear_proxy_legacy()
        this.waiting_for_m3u8_proxy = true
        const m3u8_text_list = await fetch_m3u8_playlist(this.source[this.selected_episode - 1],this.m3u8_chunk_proxy).finally(() => this.waiting_for_m3u8_proxy = false)
        if (!m3u8_text_list.length){
            console.error("Not found any available play list.")
            return
        }
        // Todo: Prefer using a resolution selector when multiple resolutions are available
        const m3u8_text = m3u8_text_list[0]
        navigator.serviceWorker.controller?.postMessage({
            type: "update_proxied_playlist_storage",
            target_url: decodeURI(new URL(this.source[this.selected_episode - 1]).href),
            override_content: {
                content:m3u8_text,
                headers: {
                    "Content-Type": "application/vnd.apple.mpegurl",
                    "Generated-By":"start_proxy"
                }
            }
        })
    }

    // clear cache to prevent multi-level proxy
    public clear_proxy_legacy(){
        navigator?.serviceWorker?.controller?.postMessage({
            type: "clear_proxied_playlist_storage"
        })
    }
}
