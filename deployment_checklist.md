# Deployment Checklist (Render.com)

1.  **Push Changes to GitHub**
    - Ensure your latest commit includes `app.py`, `requirements.txt`, and `render.yaml`.

2.  **Create Service on Render**
    - Go to [Render Dashboard](https://dashboard.render.com/).
    - Click **New +** and select **Web Service**.
    - Connect your GitHub repository: `Devanggaaryahartawijayanto/air-quality-ml-app`.

3.  **Automatic Configuration (IaC)**
    - Render should detect `render.yaml` automatically.
    - Confirm the settings:
        - **Name**: `air-quality-ml-app`
        - **Environment**: `Python`
        - **Start Command**: `gunicorn app:app`

4.  **Manual Configuration (Fallback)**
    - If `render.yaml` is ignored, fill in manually:
        - **Build Command**: `pip install -r requirements.txt`
        - **Start Command**: `gunicorn app:app`
        - **Environment Variables**:
            - `PYTHON_VERSION`: `3.10.0` (or match your local version)

5.  **Deploy & Monitor**
    - Click **Create Web Service**.
    - Watch the logs for "Build successful" and "Starting service".
    - Once deployed, your URL will be `https://air-quality-ml-app.onrender.com` (or similar).
