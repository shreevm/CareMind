# Deploy CareMind's frontend to GitHub Pages

GitHub Pages serves the static frontend. Run the Python backend separately on an HTTPS server; run MedGemma on a separate GPU endpoint when needed. Model API keys belong on those servers, never in NEXT_PUBLIC variables.

1. Deploy the Python backend and verify its `/health` endpoint. `backend/app.py` now allows the frontend origin `https://shreevm.github.io`; deploy that change with the backend. The origin does not include `/CareMind`.
2. Commit `frontend/next.config.js` and `.github/workflows/pages.yml` to the repository's default branch (merge a PR if currently on another branch). The manual Actions workflow needs to exist on the default branch to appear in GitHub.
3. In GitHub repository **Settings > Pages**, choose **GitHub Actions** as the build source.
4. In **Settings > Secrets and variables > Actions > Variables**, create `NEXT_PUBLIC_API_BASE_URL` with your public HTTPS backend base URL. This is a public address, not a secret or token.
5. Open **Actions > Deploy frontend to GitHub Pages > Run workflow** and select the intended branch. Wait for both build and deploy to pass.
6. Open the URL shown by the deployment, normally `https://shreevm.github.io/CareMind/`.

## Verify

- Check that the page and CSS load. In browser developer tools, confirm `_next` assets load under `/CareMind/` without 404 responses.
- Open the backend `/health` URL separately. GitHub Pages itself does not have a Python health endpoint.
- Use a synthetic report to test upload, chat, citations, and a follow-up question.
- In the browser Network panel, API calls must target your backend domain, not `github.io` or `localhost`. CORS errors mean the backend origin allowlist needs updating.
- Test MedGemma only after its endpoint is configured on the backend. Publishing the frontend does not start the model or keep a Colab runtime alive.

The workflow stops if no public HTTPS backend URL is configured. Re-run it after changing that variable; the API URL is embedded during the build.
