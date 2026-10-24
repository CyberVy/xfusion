# XFusion

A decoupled, high-performance generative AI platform specializing in assembling open-source diffusion models (Stable Diffusion, FLUX, etc.).

Designed with a modern **Client-Server Decoupled Architecture**:
- **Frontend / Client**: A cross-platform Web / Desktop client built with **Nextkit** (React 19, Vite, Tailwind CSS 4, Tauri 2.0). Can be deployed as a static Single Page Application (SPA) directly to any CDN (Cloudflare Pages, Vercel, AWS S3).
- **Backend / Inference Server**: A stateless **FastAPI** inference server located in `server/`, executing in-memory image generation without consuming server-side storage.

---

## Architecture & Project Structure

```text
xfusion/
├── src/                  # Frontend Client (React 19 + Nextkit blocks & components)
│   ├── app/              # Entry point & ViewSwitcher layout
│   ├── blocks/           # UI Blocks (Create workspace, Models, Gallery)
│   ├── components/       # Atomic UI primitives & composite modal containers
│   └── core/             # Reactive controllers (Settings, Model, Generation)
├── server/               # Python API Backend & Diffusion Engine
│   ├── api/              # FastAPI application, route handlers, Pydantic schemas, and manager
│   ├── enhancement/      # Diffusers pipeline enhancers (LoRA, IP-Adapter, ControlNet)
│   ├── components/       # Model loaders, architecture configurations
│   ├── const.py          # Environment variables & hardware detection
│   └── utils.py          # Concurrency locks, image processing & memory clearing utilities
├── src-tauri/            # Tauri 2.0 native desktop/mobile integration
├── public/               # Static assets & PWA manifest
├── dist/                 # Production web build output (ready for CDN deployment)
├── package.json          # Frontend scripts & dependencies (Dev Port: 8888)
├── vite.config.ts        # Vite configuration & development proxy
├── pyproject.toml        # Backend Python dependencies (managed via uv)
├── start.py              # Backend API startup script (Default Port: 8000)
├── .oxlintrc.json        # OXC linter configuration
└── .oxfmtrc.json         # OXC formatter configuration
```

---

## Supported Models

- **Stable Diffusion**: SD 1.5 / SD 2.1 / SDXL / SD 3 / SD 3.5 / Pony
- **FLUX**: FLUX.1-dev / Schnell
- **Adapters**: LoRA, IP-Adapter, ControlNet (Canny, etc.)

---

## Getting Started

### 1. Backend (Python API Server)

#### Requirements
- **OS**: Linux recommended for CUDA inference (macOS supported for development/API testing)
- **GPU**: NVIDIA GPU with CUDA support for generation
- **Python**: >= 3.10
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (recommended)

#### Installation & Launch
```bash
# 1. Sync Python dependencies into virtual environment
uv sync

# 2. Start the FastAPI server (Default: http://0.0.0.0:8000)
python start.py

# Optional environment variables:
# PORT=8000 MODEL_VERSION=xl DOWNLOAD_PATH=/xfusion_models python start.py
```

Once running:
- **Interactive Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **OpenAPI Schema**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)
- **Health Check**: [http://localhost:8000/api/v1/system/health](http://localhost:8000/api/v1/system/health)

---

### 2. Frontend (Nextkit Client)

#### Requirements
- **Node.js**: >= 18
- **Package Manager**: npm / pnpm / bun

#### Development
```bash
# Install frontend dependencies
npm install

# Start development server (Runs on http://localhost:8888)
npm run dev
```

#### Production Build & CDN Deployment
```bash
# Typecheck & build production static assets
npm run check
npm run build
```
The output will be placed in the `dist/` directory. Upload the contents of `dist/` directly to your CDN or static hosting platform (e.g. Cloudflare Pages, Vercel, Netlify, GitHub Pages, AWS S3).

---

## Connecting Client to Backend

The frontend client communicates with your backend API over HTTP with built-in CORS support:

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
   - The client will automatically store and connect to this backend on first load.

---

## API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/system/health` | Service health status and whether a model is loaded |
| `GET` | `/api/v1/system/info` | GPU, active model, LoRA, scheduler, ControlNet, IP-Adapter, Python, and OS |
| `POST` | `/api/v1/models/load` | Load or switch the base model. `model_version` may be `1.5`, `2`, `xl`, `pony`, `3`, `3.5`, `flux`, or empty to auto-detect |
| `GET` | `/api/v1/models/current` | Retrieve the currently active base model |
| `POST` | `/api/v1/lora/load` | Load a LoRA adapter with a strength |
| `POST` | `/api/v1/lora/strength` | Change the strength of a loaded LoRA |
| `POST` | `/api/v1/lora/enabled` | Enable or disable every loaded LoRA |
| `POST` | `/api/v1/lora/delete` | Remove a LoRA adapter |
| `GET` | `/api/v1/lora/list` | List loaded LoRA adapters and whether they are enabled |
| `POST` | `/api/v1/ip-adapter/load` | Load an IP-Adapter (not available for FLUX) |
| `POST` | `/api/v1/ip-adapter/strength` | Change the loaded IP-Adapter strength |
| `POST` | `/api/v1/ip-adapter/delete` | Unload the IP-Adapter |
| `POST` | `/api/v1/controlnet/load` | Load a ControlNet. An empty model uses the built-in default |
| `POST` | `/api/v1/controlnet/offload` | Offload the ControlNet |
| `POST` | `/api/v1/controlnet/preview` | Render the Canny edge map for a control image |
| `POST` | `/api/v1/scheduler/set` | Set the sampler on txt2img, img2img, or inpaint. FLUX has no selectable scheduler |
| `GET` | `/api/v1/generate/progress` | Current step and image index while a generation request is running |
| `POST` | `/api/v1/generate/txt2img` | Generate 1–10 images from a prompt |
| `POST` | `/api/v1/generate/img2img` | Transform a source image |
| `POST` | `/api/v1/generate/inpaint` | Inpaint a masked area |

Generation requests accept optional `control_image`, `controlnet_conditioning_scale`, `low_threshold`, `high_threshold`, and `ip_adapter_image`. A control image is converted to a Canny map on the server, and ControlNet is loaded automatically when one is sent. Each returned image includes the seed and the actual width and height. When more than one GPU is loaded, the requested images are split across them. Request and response fields are defined in `server/api/schemas.py` and rendered at `/docs`.

---

## Zero-Disk Server Policy

All inference output images are streamed and returned as in-memory Base64 Data URIs:
- **No temporary or cached images** are saved to the GPU server's disk during generation.
- All generation history and metadata are saved on the client side in **IndexedDB** (`localforage`), keeping the backend 100% stateless.
