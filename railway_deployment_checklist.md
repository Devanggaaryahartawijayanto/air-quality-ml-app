# Railway Deployment Checklist

1.  **Push Changes to GitHub**
    - Ensure `Procfile` is included in your commit.
    - `render.yaml` should be removed.

2.  **Create Project on Railway**
    - Go to [Railway Dashboard](https://railway.app/).
    - Click **+ New Project** > **Deploy from GitHub repo**.
    - Select your repository: `Devanggaaryahartawijayanto/air-quality-ml-app`.

3.  **Configure Service**
    - Railway should automatically detect the `Procfile` and Python environment.
    - If prompted for a start command, ensure it is `gunicorn app:app`.

4.  **Environment Variables**
    - Go to the **Variables** tab in your service dashboard.
    - Add `PORT` = `5000` (Optional, Railway assigns one automatically but good to be explicit or handle it in code).
    - Add `PYTHON_VERSION` = `3.10` (if you want to pin it via `runtime.txt` or similar, otherwise defaults apply).

5.  **Deploy & Monitor**
    - Railway automatically deploys on push.
    - Check the **DeployLogs** to ensure `gunicorn` starts successfully.
    - Click the generated URL to test the app.
