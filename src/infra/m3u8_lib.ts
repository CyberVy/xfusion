/**
 * normalize the m3u8 playlist, according to
 * [RFC8216](https://datatracker.ietf.org/doc/html/rfc8216)
 */

export function normalize_m3u8_playlist(m3u8_string: string, m3u8_origin: URL, proxy = "") {

    // avoid recursive proxy
    if (proxy && m3u8_string.includes(proxy)) proxy = ""

    const base_url = `${m3u8_origin.protocol}//${m3u8_origin.host}`
    const base_path = `${base_url}${m3u8_origin.pathname.split("/").slice(0, -1).join("/")}`

    const lines = m3u8_string.split("\n")
    const result: string[] = []

    for (let line of lines) {
        if (!line) {
            result.push(line)
            continue
        }

        if (line.startsWith("#")) {
            if (line.startsWith("#EXT-X-KEY") || line.startsWith("#EXT-X-MAP")) {
                line = line.replace(/URI="(.*?)"/, (_, uri) => {
                    if (/^https?:\/\//.test(uri)) return `URI="${proxy}${uri}"`
                    return uri.startsWith("/") ? `URI="${proxy}${base_url}${uri}"` : `URI="${proxy}${base_path}/${uri}"`
                })
            }
            result.push(line)
        }
        else {
            if (/^https?:\/\//.test(line)) {
                result.push(`${proxy}${line}`)
            }
            else {
                const full_url = line.startsWith("/") ? `${base_url}${line}` : `${base_path}/${line}`
                result.push(`${proxy}${full_url}`)
            }
        }
    }

    // fix possible #EXT-X-PLAYLIST-TYPE error
    if (!m3u8_string.includes("#EXT-X-PLAYLIST-TYPE")) {
        result.splice(1,0, "#EXT-X-PLAYLIST-TYPE:VOD")
    }

    return result.join("\n")
}

export async function fetch_m3u8_playlist(url: string,proxy = ""){
    const r: string[] = []
    const text = await fetch(url).then(r => r.text())
    const _url = new URL(url)
    if (!text.includes("#EXTM3U")){
        console.log("Not a valid m3u8 playlist url.")
        return r
    }
    if (text.includes("#EXT-X-STREAM-INF")){
        for (const item of text.split("\n")){
            if (!item.startsWith("https://") && !item.startsWith("http://")){
                // chunk line with relative path
                if (item && !item.startsWith("#")){
                    if (item[0] === "/"){
                        _url.pathname = item
                    }
                    else {
                        _url.pathname = `${_url.pathname.split("/").slice(0,-1).join("/")}/${item}`
                    }
                    const m3u8_string = await fetch(_url).then(r => r.text())
                    r.push(normalize_m3u8_playlist(m3u8_string,new URL(_url),proxy))
                }
            }
            // chunk line with http path
            else {
                const m3u8_string = await fetch(item).then(r => r.text())
                r.push(normalize_m3u8_playlist(m3u8_string,new URL(item),proxy))
            }
        }
        console.log(`${r.length} resolution detected.`)
    }
    else {
        r.push(normalize_m3u8_playlist(text,new URL(_url),proxy))
    }
    return r
}

export function get_m3u8_playlist_metadata(m3u8_string: string){

    const target_line_list: string[] = []
    for (const line of m3u8_string.trim().split("\n")){
        if (!line.startsWith("#EXTINF:")){
            target_line_list.push(line)
        }
        else {
            break
        }
    }
    return target_line_list.join("\n")
}

export function get_m3u8_discontinuity(m3u8_string: string){
    const duration_list = []
    const chunk_length_list = []
    for (const item of m3u8_string.split("#EXT-X-DISCONTINUITY\n")){
        let duration = 0
        let chunk_length = 0
        for (const _item of item.split("#EXTINF:")){
            if (/^\d/.test(_item)){
                const match = _item.match(/^\d(?:\.\d+)?/)
                if (match){
                    duration += Number(match[0])
                    chunk_length += 1
                }
            }
        }
        duration_list.push(duration)
        chunk_length_list.push(chunk_length)
    }
    function cum_sum(arr: number[]) {
        let sum = 0
        return arr.map(v => sum += v)
    }
    return [duration_list,[0,...cum_sum(duration_list)],chunk_length_list]
}

export function get_m3u8_ad_breakpoint(m3u8_string: string, threshold = 30){
    const [duration_list,breakpoint_list] = get_m3u8_discontinuity(m3u8_string)
    const r: number[][] = []
    duration_list.map((duration, index) => {
        if (duration < threshold && duration != 0){
            r.push([breakpoint_list[index],breakpoint_list[index + 1]])
        }
    })
    if (r.length){
        console.log("Detected AD:",r)
    }
    else {
        console.log("No AD.")
    }
    return r
}

export function remove_m3u8_ad_chunk(m3u8_string: string, threshold = 30){
    const [duration_list,breakpoint_list,chunk_length_list] = get_m3u8_discontinuity(m3u8_string)
    const ad_breakpoint: number[][] = []
    const _ = m3u8_string.split("#EXT-X-DISCONTINUITY\n")
    duration_list.map((duration,index) => {
        if (duration < threshold && duration != 0){
            _[index] = ""
            ad_breakpoint.push([breakpoint_list[index],breakpoint_list[index + 1]])
        }
    })

    let r = _.join("").trim()
    if (!r.endsWith("#EXT-X-ENDLIST")){
        r += "\n#EXT-X-ENDLIST\n"
    }
    if (!r.startsWith("#EXTM3U")){
        r = `${get_m3u8_playlist_metadata(m3u8_string)}\n${r}`
    }

    if (ad_breakpoint.length){
        console.log(`Removed ${ad_breakpoint.length} AD.`)
    }
    else {
        console.log("No AD.")
    }
    return r
}
