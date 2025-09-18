import {CoverImageOptions} from "@/infra/types"

export function generate_silent_wav_base64(durationSec = 2, sampleRate = 44100) {

    const numChannels = 1
    const bitsPerSample = 16
    const byteRate = (sampleRate * numChannels * bitsPerSample) / 8
    const blockAlign = (numChannels * bitsPerSample) / 8
    const numSamples = durationSec * sampleRate
    const dataSize = numSamples * blockAlign
    const buffer = Buffer.alloc(44 + dataSize)

    // RIFF header
    buffer.write("RIFF", 0)
    buffer.writeUInt32LE(36 + dataSize, 4)
    buffer.write("WAVE", 8)

    // fmt subchunk
    buffer.write("fmt ", 12)
    buffer.writeUInt32LE(16, 16)
    buffer.writeUInt16LE(1, 20) // PCM
    buffer.writeUInt16LE(numChannels, 22)
    buffer.writeUInt32LE(sampleRate, 24)
    buffer.writeUInt32LE(byteRate, 28)
    buffer.writeUInt16LE(blockAlign, 32)
    buffer.writeUInt16LE(bitsPerSample, 34)

    // data subchunk
    buffer.write("data", 36)
    buffer.writeUInt32LE(dataSize, 40)

    // samples are all zero (silence)
    // Buffer is already zero-filled

    return "data:audio/wav;base64," + buffer.toString("base64")
}

const cover_image_cache:Record<string, string> = {}

export function generate_cover_image(title: string, options:CoverImageOptions) {
    if (title in cover_image_cache){
        return Promise.resolve(cover_image_cache[title])
    }

    const {
        width = 720,
        height = 1024,
        background = "#333",
        color = "#fff",
        fontSize = 48,
        fontFamily = "sans-serif",
    } = options

    const canvas = document.createElement("canvas")
    canvas.width = width
    canvas.height = height
    const ctx = canvas.getContext("2d")
    if (ctx){
        ctx.fillStyle = background
        ctx.fillRect(0, 0, width, height)

        ctx.fillStyle = color
        ctx.font = `${fontSize}px ${fontFamily}`
        ctx.textAlign = "center"
        ctx.textBaseline = "middle"
        ctx.fillText(title, width / 2, height / 2)
    }

    return new Promise<string>(resolve => {
        canvas.toBlob(blob => {
            if (blob) {
                const url = URL.createObjectURL(blob)
                cover_image_cache[title] = url
                resolve(url)
            } else {
                resolve("")
            }
        }, "image/png")
    })
}
