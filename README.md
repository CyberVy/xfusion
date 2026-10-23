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
│   ├── blocks/           # UI Blocks (Txt2Img, Img2Img, Inpaint, Models & LoRA, Gallery)
│   ├── components/       # Atomic UI primitives & composite modal containers
│   └── core/             # Reactive controllers (SettingsController, GenerationController)
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
| `GET` | `/api/v1/system/health` | Service health status & model loaded state |
| `GET` | `/api/v1/system/info` | GPU status, device names, active model & available schedulers |
| `POST` | `/api/v1/models/load` | Load or switch base diffusion model |
| `GET` | `/api/v1/models/current` | Retrieve currently active base model info |
| `POST` | `/api/v1/lora/load` | Attach / load a LoRA adapter with weight |
| `POST` | `/api/v1/lora/delete` | Remove an active LoRA adapter |
| `GET` | `/api/v1/lora/list` | List all active LoRA adapters |
| `POST` | `/api/v1/scheduler/set` | Change numerical sampler (DPM++ 2M, Euler, etc.) |
| `POST` | `/api/v1/generate/txt2img` | Generate image from text prompt |
| `POST` | `/api/v1/generate/img2img` | Transform an image from input image & prompt |
| `POST` | `/api/v1/generate/inpaint` | Inpaint image using mask and prompt |

---

## Zero-Disk Server Policy

All inference output images are streamed and returned as in-memory Base64 Data URIs:
- **No temporary or cached images** are saved to the GPU server's disk during generation.
- All generation history and metadata are saved on the client side in **IndexedDB** (`localforage`), keeping the backend 100% stateless.
