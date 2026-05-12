# Deploy To Render

This project is prepared to deploy as a public website on Render.

## What is already prepared

- `ids_dashboard_server.py` now supports hosted deployment through environment variables
- `render.yaml` tells Render how to build and run the app
- `deployment_assets/stream_source.csv` is a smaller packaged stream dataset for the live dashboard
- `deployment_assets/models/` contains the centralized and federated model files used by the website

## Step 1: Make sure Git is installed

Open Terminal and run:

```bash
git --version
```

If you get an error on your Mac, run:

```bash
xcode-select --install
```

## Step 2: Create a GitHub account

Go to [https://github.com](https://github.com) and create an account if you do not already have one.

## Step 3: Create a new GitHub repository

After logging in to GitHub:

1. Click the `+` icon in the top-right corner.
2. Click `New repository`.
3. Name it something like `ml-vs-fl-dashboard`.
4. Leave it empty.
5. Click `Create repository`.

## Step 4: Upload this project to GitHub

Open Terminal in this folder:

```bash
cd "/Users/ayushimishra/Documents/Codex/2026-04-26/files-mentioned-by-the-user-w"
```

Then run these commands one by one:

```bash
git init
git add .
git commit -m "Prepare ML vs FL dashboard for deployment"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/ml-vs-fl-dashboard.git
git push -u origin main
```

Replace `YOUR_GITHUB_USERNAME` with your actual GitHub username.

If Git asks for your name and email first, run:

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

Then repeat the `git add`, `git commit`, and `git push` commands.

## Step 5: Create a Render account

Go to [https://render.com](https://render.com) and sign up.

The easiest option is `Continue with GitHub`.

## Step 6: Deploy the website on Render

After logging into Render:

1. Click `New +`
2. Click `Blueprint`
3. Connect your GitHub account if Render asks
4. Select your repository `ml-vs-fl-dashboard`
5. Render will detect the `render.yaml` file automatically
6. Review the service settings
7. Click `Apply`

Render will then:

1. install the Python dependencies from `requirements.txt`
2. start the dashboard server with `python ids_dashboard_server.py`
3. give you a public website URL ending in `.onrender.com`

## Step 7: Open the live site

When the deploy finishes, Render shows a public URL like:

```text
https://ml-vs-fl-dashboard.onrender.com
```

Open that URL in your browser. Your live dashboard should be there.

## Step 8: Update the website later

Whenever you change the code locally, run:

```bash
cd "/Users/ayushimishra/Documents/Codex/2026-04-26/files-mentioned-by-the-user-w"
git add .
git commit -m "Describe your change"
git push
```

Render will automatically redeploy the website after each push.

## Step 9: If Render says `free` is unavailable

The `render.yaml` file currently asks for the `free` plan. If your Render account does not offer it, change the service plan in the Render setup screen to the cheapest available option and continue.

## Troubleshooting

If the build fails:

1. Open the failed deploy in Render
2. Read the build logs
3. Fix the problem locally
4. Run:

```bash
git add .
git commit -m "Fix deployment issue"
git push
```

If the site opens but the dashboard does not work:

1. Open the Render service logs
2. Look for Python errors
3. Make sure the deploy included:
   - `deployment_assets/stream_source.csv`
   - `deployment_assets/models/centralized_best_model.npz`
   - `deployment_assets/models/federated_best_model.npz`
