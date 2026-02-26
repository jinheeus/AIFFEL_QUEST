# Deployment Guide for AURA Chatbot

This project uses a **Split Deployment Strategy**:
- **Frontend (Next.js)**: Deployed to **Vercel** (Serverless, optimized for React).
- **Backend (FastAPI)**: Deployed to **Render** or **Railway** (Docker container, supports heavy ML libraries).

---

## 1. Backend Deployment (Render/Railway)

The backend must be deployed first to get the public API URL.

### Option A: Railway (Recommended for ease of use)
1.  **Sign up/Login** at [railway.app](https://railway.app/).
2.  **New Project** -> **GitHub Repo** -> `aura/aiffelthon`.
3.  Railway should automatically detect the `Dockerfile` at the root.
4.  **Variables**: Add the following Environment Variables (copy values from your local `.env`):
    - `CLOVASTUDIO_API_KEY`
    - `OPENAI_API_KEY` (if used)
    - `GOOGLE_API_KEY` (if used)
    - `MILVUS_URI` (You may need a cloud Milvus like Zilliz, or Railway's Redis/Postgres add-ons if you adapt the code. For now, if using local file DB, **it won't persist well on serverless containers** without a persistent volume. **Recommendation**: Use Zilliz Cloud free tier for Milvus).
6.  **Deploy**.
7.  **Branch Configuration (Optional)**:
    - By default, Railway deploys the `main` branch.
    - To deploy a specific branch (e.g., `feature/demo`), go to **Settings** -> **Git**.
    - Change **Production Branch** to `feature/demo`.
    - Or, create a new **Environment** and link it to the `feature/demo` branch.
8.  **Copy URL**: Once deployed, copy the provided domain (e.g., `https://aura-backend.up.railway.app`).

### Option B: Render
1.  **Sign up/Login** at [render.com](https://render.com/).
2.  **New** -> **Web Service**.
3.  Connect your GitHub repository.
4.  **Runtime**: Select **Docker**.
5.  **Environment Variables**: Add keys from `.env` as above.
6.  **create Web Service**.

---

## 2. Frontend Deployment (Vercel)

1.  **Sign up/Login** at [vercel.com](https://vercel.com/).
2.  **Add New** -> **Project**.
3.  Import `aura/aiffelthon`.
4.  **Framework Preset**: Next.js (Should be auto-detected).
5.  **Root Directory**: Click "Edit" and select `web_app/frontend`.
6.  **Environment Variables**:
    - `NEXT_PUBLIC_API_URL`: Paste the Backend URL from Step 1 (e.g., `https://aura-backend.up.railway.app` without the trailing slash).
7.  **Deploy**.

---

## 3. Verification
1.  Open the Vercel deployment URL.
2.  Type a message in the chat.
3.  Ensure the response streams back correctly.

## Troubleshooting
- **CORS Error**: If the frontend cannot talk to the backend, check `web_app/backend/main.py`. It is currently configured to allow `["*"]` which is open, but ensure your backend service isn't blocking origins.
- **Timeout**: If the backend takes too long to start (loading heavy models), increase the timeout settings in your cloud provider or upgrade the plan.
