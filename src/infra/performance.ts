export function throttle<Args extends unknown[], R>(f: (...args: Args) => R, delay: number){
    let waiting = false
    function g(...args: Args){
        if (waiting){
            return
        }
        waiting = true
        setTimeout(() => waiting = false, delay)
        return f(...args)
    }
    return g
}
