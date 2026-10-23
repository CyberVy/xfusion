import { settings_controller } from "./settings_controller"
import { generation_controller } from "./generation_controller"

export const debug_controllers = {
    settings: settings_controller,
    generation: generation_controller,

    settings_controller,
    generation_controller,

    dump_state: () => ({
        settings: Object.fromEntries(settings_controller.entries()),
        generation: generation_controller.state
    })
}

export type DebugControllers = typeof debug_controllers

export function mount_debug_controllers(): void {
    if (typeof window === "undefined") return

    ;(window as any).__controllers__ = debug_controllers
    ;(window as any).controllers = debug_controllers
}
