"use client"

import { useState, useEffect } from "react"
import {
    WorkspaceView,
    ModelManageView,
    GalleryView,
    GlobalSettingButton,
    LaunchAnimation,
    WandIcon,
    CubeIcon,
    GalleryIcon
} from "@/blocks"
import { ViewSwitcher } from "@/components"
import { settings_controller } from "@/core/controllers/settings_controller"
import { mount_debug_controllers } from "@/core/controllers/mount"

type ActiveView = "create" | "models" | "gallery"

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
                    active_view_id={active_view}
                    on_view_change={set_active_view}
                    keep_alive_default={true}
                    toolbar_extra_actions={
                        <GlobalSettingButton className="mr-1 bg-background/0! backdrop-blur-none! border-none! size-10! rounded-[18px]! text-black/50! dark:text-white/50!" />
                    }
                    views={[
                        {
                            id: "create",
                            icon: <WandIcon width={22} height={22} />,
                            label: "Create",
                            swipe_enabled: false,
                            content: <WorkspaceView />
                        },
                        {
                            id: "models",
                            icon: <CubeIcon width={22} height={22} />,
                            label: "Models",
                            swipe_enabled: false,
                            content: <ModelManageView />
                        },
                        {
                            id: "gallery",
                            icon: <GalleryIcon width={22} height={22} />,
                            label: "Gallery",
                            swipe_enabled: false,
                            content: <GalleryView />
                        }
                    ]}
                />
            </div>

            {show_launch_animation && <LaunchAnimation on_finish={() => set_show_launch_animation(false)} />}
        </>
    )
}
