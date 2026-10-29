# xfusion

A decoupled, high-performance generative AI platform specializing in assembling open-source diffusion models (Stable Diffusion, FLUX, etc.).

Designed with a modern **Client-Server Decoupled Architecture**:

- **Frontend / Client**: A cross-platform Web / Desktop client built with **Nextkit** (React 19, Vite, Tailwind CSS 4, Tauri 2.0). Can be deployed as a static Single Page Application (SPA) directly to any CDN (Cloudflare Pages, Vercel, AWS S3) or run as a native desktop/mobile app.
- **Backend / Inference Server**: A stateless **FastAPI** inference server executing in-memory image generation without consuming server-side storage. Supports multi-GPU distribution and dynamic adapter loading.

---

## Architecture & Project Structure

```text
xfusion/
├── src/                  # Frontend Client (React 19 + Nextkit blocks & components)
│   ├── app/              # Entry points (SPA & native Tauri shell), global styles
│   ├── blocks/           # Domain UI blocks (Workspace, Model Manager, Gallery, Result Panel)
│   ├── components/       # Atomic primitives, animation containers & composite modals
│   ├── core/             # Domain controllers (Settings, Model, Generation) & API client
│   ├── infra/            # Browser platform adapters, storage (IndexedDB), Web IPC client
│   └── sw/               # Service worker for offline static asset caching
├── server/               # Python API Backend & Diffusion Engine
│   ├── api/              # FastAPI app, route handlers, Pydantic schemas & lifecycle manager
│   ├── engine/           # Unified pipeline engines (SD, FLUX), adapters, schedulers, prompt utils
│   ├── models/           # Model architecture detection, loaders (SD, FLUX) & pipeline configs
│   └── infra/            # Hardware/CUDA config, GPU memory management, image utils, model downloader
├── scripts/              # Helper scripts & development toolchain
│   ├── dev.sh            # Unified fullstack developer launcher (local / remote)
│   ├── remote/           # Remote GPU synchronization, daemon manager & port forwarding
│   └── platform/         # Native platform build, icon generation & app signing
├── src-tauri/            # Tauri 2.0 cross-platform desktop/mobile integration
├── public/               # Static assets & PWA manifest
├── cli/                  # Frontend debug bridge & testing utilities
├── package.json          # Frontend scripts, dependencies & build tasks
├── pyproject.toml        # Backend Python dependencies & entrypoints (managed via uv)
└── vite.config.ts        # Vite configuration & development server (Port: 8888)
```

---

## Supported Models & Features

- **Base Diffusion Models**:
    - **Stable Diffusion**: SD 1.5, SD 2.1, SDXL, SD 3 / SD 3.5, Pony
    - **FLUX**: FLUX.1-dev, FLUX.1-schnell (with FP8 quantization support)
    - Automatic architecture detection from safetensors checkpoint state dicts
- **Generation Pipelines**:
    - **Text-to-Image (`txt2img`)**: High-resolution generation, prompt weighting, custom dimensions & aspect ratios
    - **Image-to-Image (`img2img`)**: Image-guided synthesis with adjustable transformation strength
    - **Inpainting (`inpaint`)**: Mask-based editing with interactive browser canvas painter (`MaskPainter`)
- **Adapters & Enhancers**:
    - **LoRA**: Dynamic multi-LoRA loading, real-time strength adjustment without reloading, and global toggle
    - **IP-Adapter**: Image-prompt reference conditioning with configurable weight
    - **ControlNet**: Canny edge detection with server-side preview and conditioning scale
- **Schedulers / Samplers**:
    - DPM++ 2M Karras, DPM++ SDE Karras, Euler, Euler A, DDIM, UniPC, Heun, and more
- **Client Features**:
    - Full client-side persistent storage (**IndexedDB** via `localforage`) for generation history, prompts, and gallery
    - Real-time denoising progress tracking and cancel-safe polling
    - One-click configuration via URL query parameters (`?api=...`)
    - Cross-platform support: Web SPA, PWA, and native Desktop/Mobile via Tauri 2.0

---

## Getting Started

### 1. Backend (Python API Server)

#### Requirements

- **OS**: Linux recommended for CUDA inference (macOS supported for development/API testing)
- **GPU**: NVIDIA GPU with CUDA support for diffusion generation
- **Python**: >= 3.10
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (recommended) or `pip`

#### Installation & Launch

```bash
# 1. Install backend dependencies (via uv)
uv sync

# 2. Start the FastAPI server (Default: http://0.0.0.0:8000)
uv run xfusion-server

# Optional environment variables:
# HOST=0.0.0.0 PORT=8000 DEBUG=true DOWNLOAD_PATH=/xfusion_models uv run xfusion-server
```

Once running:

- **Interactive Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **OpenAPI Schema**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)
- **Health Check**: [http://localhost:8000/api/v1/system/health](http://localhost:8000/api/v1/system/health)

#### Environment Variables

| Variable           | Default                         | Description                                                                           |
| :----------------- | :------------------------------ | :------------------------------------------------------------------------------------ |
| `HOST`             | `0.0.0.0`                       | Host IP to bind the API server                                                        |
| `PORT`             | `8000`                          | Port to bind the API server                                                           |
| `DEBUG`            | `false`                         | Enable automatic server reload on code change                                         |
| `MODEL_URL`        | `None`                          | Optional model URL or path to auto-load on startup                                    |
| `MODEL_VERSION`    | `""`                            | Version hint for `MODEL_URL` (`1.5`, `2`, `xl`, `pony`, `3`, `3.5`, `flux`, or empty) |
| `DOWNLOAD_PATH`    | `/xfusion_models`               | Local directory for caching downloaded models and adapters                            |
| `HF_HUB_TOKEN`     | `None`                          | Hugging Face token for downloading gated models                                       |
| `CIVITAI_TOKEN`    | `None`                          | Civitai API key for downloading models and LoRAs                                      |
| `NEED_PROXY`       | Auto                            | Force enable (`true`) or disable (`false`) proxy routing                              |
| `PROXY_URL_PREFIX` | `https://us2.xsolutiontech.com` | Proxy prefix for downloads                                                            |
| `CORS_ORIGINS`     | `*`                             | Comma-separated list of allowed CORS origins                                          |

---

### 2. Frontend (Nextkit Client)

#### Requirements

- **Node.js**: >= 18
- **Package Manager**: npm / pnpm / bun

#### Local Development

```bash
# Install frontend dependencies
npm install

# Start development server (Runs on http://localhost:8888)
npm run dev
```

#### Remote GPU Development (via `rexec`)

If you develop on macOS or a laptop without a dedicated CUDA GPU, you can connect your local frontend to a remote GPU machine seamlessly:

1. Copy `.env.example` to `.env` and fill in your `rexec` credentials:
    ```bash
    cp .env.example .env
    ```
2. Start dev session:
    ```bash
    # Remote GPU workflow (sync + daemon + forward + frontend)
    ./scripts/dev.sh

    # Local fullstack workflow (local backend + frontend)
    ./scripts/dev.sh --local
    ```
    The remote workflow automatically syncs your local code to the remote GPU instance, starts the background backend daemon, creates a secure port forwarding tunnel (localhost:8000 -> remote:8000), and launches your local frontend dev server.

Additional remote helper scripts:

```bash
./scripts/remote/sync.sh      # Sync local files to the remote server
./scripts/remote/backend.sh   # Manage remote backend daemon (start/stop/restart/status/logs)
./scripts/remote/forward.sh   # Open SSH/rexec port forwarding tunnel
```

#### Production Build & CDN Deployment

```bash
# Code verification (Oxlint & TypeScript typecheck)
npm run check

# Build production static bundle
npm run build

# Preview production build locally
npm run start
```

The production bundle is generated in `dist/`. Upload the contents of `dist/` directly to any static web hosting platform (e.g. Cloudflare Pages, Vercel, Netlify, AWS S3).

---

## Connecting Client to Backend

The frontend client communicates with the inference backend over HTTP with full CORS support:

1. **Via UI Settings**:
    - Open the client in your browser.
    - Click the **Settings (gear icon)** at the bottom-right corner.
    - Enter your Python backend URL (e.g. `http://localhost:8000` or `https://gpu.yourdomain.com`).
    - Click **Test** to verify connection. The URL is saved locally in browser storage (`IndexedDB`).

2. **Via URL Parameter (Auto-configuration)**:
    - Share or bookmark URLs with the `?api=` parameter:
        ```text
        https://your-cdn-app.pages.dev/?api=http://gpu.server.com:8000
        ```
    - The client will automatically store and connect to this backend on initial load.

---

## API Endpoints Reference

| Method | Endpoint                      | Description                                                                                                                                     |
| :----- | :---------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET`  | `/api/v1/system/health`       | Service health status and whether a model is loaded                                                                                             |
| `GET`  | `/api/v1/system/info`         | GPU status, loaded model & version, active LoRAs & toggle, available schedulers, ControlNet & IP-Adapter state, Python & OS info                |
| `POST` | `/api/v1/models/load`         | Load or switch base model (`model`, `model_version`: `1.5`, `2`, `xl`, `pony`, `3`, `3.5`, `flux`, or empty for auto-detect)                    |
| `GET`  | `/api/v1/models/current`      | Retrieve currently active base model name and architecture version                                                                              |
| `POST` | `/api/v1/lora/load`           | Load or update a LoRA adapter (`url`, `lora_name`, `strength`)                                                                                  |
| `POST` | `/api/v1/lora/strength`       | Adjust the strength of an active LoRA without redownloading (`lora_name`, `strength`)                                                           |
| `POST` | `/api/v1/lora/enabled`        | Enable or disable all loaded LoRAs globally (`enabled`)                                                                                         |
| `POST` | `/api/v1/lora/delete`         | Remove a LoRA adapter (`lora_name`)                                                                                                             |
| `GET`  | `/api/v1/lora/list`           | List active LoRA adapters and whether they are currently enabled                                                                                |
| `POST` | `/api/v1/ip-adapter/load`     | Load an IP-Adapter (`uri`, `subfolder`, `weight_name`, `image_encoder_folder`, `strength`)                                                      |
| `POST` | `/api/v1/ip-adapter/strength` | Adjust the strength of the loaded IP-Adapter (`strength`)                                                                                       |
| `POST` | `/api/v1/ip-adapter/delete`   | Unload the IP-Adapter                                                                                                                           |
| `POST` | `/api/v1/controlnet/load`     | Load ControlNet weights (`model` or null for pipeline default)                                                                                  |
| `POST` | `/api/v1/controlnet/offload`  | Offload the loaded ControlNet model                                                                                                             |
| `POST` | `/api/v1/controlnet/preview`  | Render the Canny edge preview for a control image (`image`, `low_threshold`, `high_threshold`)                                                  |
| `POST` | `/api/v1/scheduler/set`       | Switch sampler/scheduler on the active pipeline (`scheduler`, `pipeline_type`)                                                                  |
| `GET`  | `/api/v1/generate/progress`   | Live step progress, total steps, and image indices while generation is running                                                                  |
| `POST` | `/api/v1/generate/txt2img`    | Generate images from text prompt (`prompt`, `negative_prompt`, `guidance_scale`, `num_inference_steps`, `width`, `height`, `seed`, `num`, etc.) |
| `POST` | `/api/v1/generate/img2img`    | Image-to-image synthesis (`image`, `prompt`, `strength`, etc.)                                                                                  |
| `POST` | `/api/v1/generate/inpaint`    | Inpaint masked area of an image (`image`, `mask_image`, `prompt`, `strength`, etc.)                                                             |

Generation requests accept optional parameters including `control_image`, `controlnet_conditioning_scale`, `low_threshold`, `high_threshold`, and `ip_adapter_image`. Control images are processed to Canny maps on the server, and ControlNet is activated automatically when a control image is supplied. When multiple GPUs are detected, requested batches are distributed across them.

---

## Zero-Disk Server Policy

All inference output images are streamed and returned as in-memory Base64 Data URIs:

- **No temporary or cached images** are saved to the GPU server's disk during generation.
- All generation history, prompt records, and image blobs are saved on the client side in **IndexedDB** (`localforage`), keeping the backend 100% stateless and safe for multi-tenant or ephemeral container environments.
