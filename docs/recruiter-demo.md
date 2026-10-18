# Free recruiter demo

This deployment uses GitHub Pages for the UI and one free Render Docker web
service for the API. SQLite stores vectors, metadata, and conversation history.
Redis is optional and omitted; response caching is disabled. NVIDIA API calls
handle generation and embeddings, so the server does not load local ML models.
Provider credits/quotas are separate from free hosting.

## Deploy

1. Commit and push the intended application changes plus `Dockerfile.demo`,
   `requirements-demo.txt`, `render.yaml`, and `.github/workflows/pages.yml`.
   Put the Pages workflow on the repository default branch. Do not commit `.env`
   or local uploads. Review the existing working-tree changes before committing.
2. Sign in at https://dashboard.render.com using GitHub. Choose **New > Blueprint**,
   connect `shreevm/CareMind`, and select the branch containing `render.yaml`.
   Confirm that the API service plan is **Free**.
3. Enter `NVIDIA_API_KEY` when prompted, using a key with available provider quota.
   Keep it in Render's secret environment settings, never GitHub Pages variables.
   The default chat model comes from `backend/caremind/config.py`; override
   `NVIDIA_CHAT_MODEL` in Render if your account requires another supported model.
4. Deploy and open `https://YOUR-SERVICE.onrender.com/health`. The API root `/`
   does not serve the frontend in this slim image. A successful health request
   checks initialization, not provider credentials; test a seeded chat as well.
5. Follow [the GitHub Pages instructions](github-pages.md), setting the repository
   variable `NEXT_PUBLIC_API_BASE_URL` to your Render URL without `/health`.
   Run **Deploy frontend to GitHub Pages** and open its deployment URL.

## Demonstrate

- Open the site before the interview and allow the API to wake up.
- Click **Seed demo reports**, then ask about the key findings with citations.
- Try a follow-up question and select both reports for comparison.
- Use synthetic text/PDF documents only. All visitors share the demo workspace;
  this deployment deliberately has no login or private user data isolation.
- Uploads, SQLite records, and seeded reports disappear when the free backend
  restarts or redeploys. Seed again and start a new chat after a reset; browser
  history may still refer to documents that no longer exist.
- Voice needs an additional `ELEVENLABS_API_KEY`. Local Qwen/MedGemma models and
  local Tesseract OCR are not included in the slim image; remote vision/OCR needs
  provider support and configuration. Text/PDF chat is the basic demo path.

Render's free service sleeps after inactivity and can take about a minute to
wake up. See https://render.com/docs/free for current limits. Keep a short screen
recording available for the interview if the host or model provider is slow.

The latest saved quality evaluation has failures; this is a portfolio prototype,
not a production or clinical deployment. After deploying, verify seed, upload,
chat, citations, comparison, and browser CORS before sending the link.
