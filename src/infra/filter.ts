import {ResourceListItem} from "./types"

function remove_elements<T>(l:T[], to_remove:T[]) {
    const to_remove_set = new Set(to_remove)
    for (let i = l.length - 1; i >= 0; i--) {
        if (to_remove_set.has(l[i])) {
            l.splice(i, 1)
        }
    }
    return l
}

function sort_resource_list_by_score(l:ResourceListItem[]){
    l.sort((a,b) => {
        if ((a.vod_douban_score || a.vod_score) && (b.vod_douban_score || b.vod_score)){
            const score_a = Number(a.vod_douban_score != "0.0" ? a.vod_douban_score : a.vod_score)
            const score_b = Number(b.vod_douban_score != "0.0" ? b.vod_douban_score : b.vod_score)
            return score_b - score_a
        }
        return 0
    })
    return l
}

function remove_duplicates_from_resource_list(l: ResourceListItem[]): ResourceListItem[] {
    const seen = new Set<string>()
    const result: ResourceListItem[] = []
    for (const item of l) {
        if (!seen.has(item.vod_name)) {
            seen.add(item.vod_name)
            result.push(item)
        }
    }
    return result
}


export {
    sort_resource_list_by_score,
    remove_duplicates_from_resource_list
}
