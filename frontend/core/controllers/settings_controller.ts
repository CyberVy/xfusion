import { LocalForageMap } from "@/infra/storage/db.client"
import { BaseKeyedController } from "@/infra/base_controller"
import { create_logger } from "@/infra/logger"
import { ApiClient } from "../api/client"

const logger = create_logger("SettingsController")

export type SettingsEventType = "set" | "delete" | "clear" | `set:${string}` | `delete:${string}`

export class SettingsController extends BaseKeyedController<string, any, SettingsEventType> {
    private settings_store = new LocalForageMap<any>("settings", "xfusion")
    private memory_settings = new Map<string, any>()
    private _is_connected: boolean = false

    constructor() {
        super()
        // Default API url
        this.memory_settings.set("api_url", "http://localhost:8000")
        this.init().catch((err) => {
            logger.error("Failed to initialize settings", err)
        })
    }

    private async init(): Promise<void> {
        const keys = await this.settings_store.keys()
        for (const key of keys) {
            const value = await this.settings_store.get(key)
            if (value !== undefined) {
                this.memory_settings.set(key, value)
                const detail = { key, value }
                this.dispatchEvent(new CustomEvent("set", { detail }))
                this.dispatchEvent(new CustomEvent(`set:${key}`, { detail }))
            }
        }
        await this.check_connection()
    }

    public async check_connection(): Promise<boolean> {
        const url = this.get_value("api_url", "http://localhost:8000")
        const is_healthy = await ApiClient.check_health(url)
        this._is_connected = is_healthy
        this.dispatchEvent(new CustomEvent("set:connection", { detail: { is_connected: is_healthy } }))
        return is_healthy
    }

    public get is_connected(): boolean {
        return this._is_connected
    }

    public override get_value<T>(key: string, default_value: T): T {
        if (this.memory_settings.has(key)) {
            return this.memory_settings.get(key) as T
        }
        return default_value
    }

    public override get_key_events(key: string): SettingsEventType[] {
        return [`set:${key}`, `delete:${key}`, "clear"]
    }

    public async set(key: string, value: any): Promise<void> {
        this.memory_settings.set(key, value)
        await this.settings_store.set(key, value)
        const detail = { key, value }
        this.dispatchEvent(new CustomEvent("set", { detail }))
        this.dispatchEvent(new CustomEvent(`set:${key}`, { detail }))

        if (key === "api_url") {
            await this.check_connection()
        }
    }

    public async delete(key: string): Promise<void> {
        this.memory_settings.delete(key)
        await this.settings_store.delete(key)
        const detail = { key }
        this.dispatchEvent(new CustomEvent("delete", { detail }))
        this.dispatchEvent(new CustomEvent(`delete:${key}`, { detail }))
    }

    public entries(): IterableIterator<[string, any]> {
        return this.memory_settings.entries()
    }
}

export const settings_controller = new SettingsController()
