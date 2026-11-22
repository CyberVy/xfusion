"use client"

import { WandSparkles as WandIcon, Box as CubeIcon, Images as GalleryIcon, Settings as SettingIcon } from "lucide-react"
import { useState, useEffect } from "react"
import { WorkspaceView, ModelManageView, GalleryView, SettingsView, LaunchAnimation } from "@/blocks"
import { ViewSwitcher } from "@/components"
import { settings_controller } from "@/core/controllers/settings_controller"
import { mount_debug_controllers } from "@/core/controllers/mount"

type ActiveView = "create" | "models" | "gallery" | "settings"

export default function App() {
    const [show_launch_animation, set_show_launch_animation] = useState(true)
    const [active_view, set_active_view] = useState<ActiveView>("create")

    useEffect(() => {
        mount_debug_controllers()

        // Handle URL parameter for API configuration (e.g., https://app.domain/?api=http://gpu.server:8000)
        try {
            const url = new URL(location.href)
            const query_api = url.searchParams.get("api")
            if (query_api) {
                settings_controller.set("api_url", query_api)
                history.replaceState(null, "", "/")
            }
        } catch {
            // Ignore SSR/URL parsing issues
        }
    }, [])

    return (
        <>
            <div className="select-none min-h-screen" {...(show_launch_animation ? { inert: true } : {})}>
                <ViewSwitcher<ActiveView>
                    id="xfusion_switcher"
                    toolbar_layout="bottom-floating"
                    toolbar_hidden={show_launch_animation}
                    active_view_id={active_view}
                    on_view_change={set_active_view}
                    keep_alive_default={true}
                    views={[
                        {
                            id: "create",
                            icon: <WandIcon width={22} height={22} />,
                            native_icon: "wand.and.stars",
                            label: "Create",
                            swipe_enabled: false,
                            content: <WorkspaceView />
                        },
                        {
                            id: "models",
                            icon: <CubeIcon width={22} height={22} />,
                            native_icon: "cube",
                            label: "Models",
                            swipe_enabled: false,
                            content: <ModelManageView />
                        },
                        {
                            id: "gallery",
                            icon: <GalleryIcon width={22} height={22} />,
                            native_icon: "photo.on.rectangle.angled",
                            label: "Gallery",
                            swipe_enabled: false,
                            content: <GalleryView />
                        },
                        {
                            id: "settings",
                            icon: <SettingIcon width={22} height={22} />,
                            native_icon: "gearshape",
                            label: "Settings",
                            swipe_enabled: false,
                            content: <SettingsView />
                        }
                    ]}
                />
            </div>

            {show_launch_animation && <LaunchAnimation on_finish={() => set_show_launch_animation(false)} />}
        </>
    )
}
