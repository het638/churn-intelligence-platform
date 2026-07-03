# Deployment Guide: GitHub + Streamlit Community Cloud

This machine doesn't have Git installed, so these steps need to be run by you.
Everything else (the code, the trained model, the data snapshot) is already
committed-ready in the project folder.

## 1. Install Git

Download and install from the official site:
https://git-scm.com/download/win

Accept the defaults during install. Then restart your terminal so `git` is on
PATH.

## 2. Initialize the repo and make the first commit

```powershell
cd C:\Users\hetsa\Projects\churn-intelligence-platform
git init
git add .
git commit -m "Customer Churn & Revenue Intelligence Platform: ETL + SQL + ML + R + Streamlit"
```

Double-check `.env` is NOT staged (it's gitignored, but verify with
`git status` — you should not see it listed).

## 3. Create the GitHub repository

1. Go to https://github.com/new
2. Name it e.g. `churn-intelligence-platform`, keep it **Public** (so
   recruiters and Streamlit Cloud can access it), don't initialize with a
   README (you already have one).
3. Copy the commands GitHub shows you under "…or push an existing
   repository from the command line", something like:

```powershell
git remote add origin https://github.com/<your-username>/churn-intelligence-platform.git
git branch -M main
git push -u origin main
```

You'll be prompted to authenticate — use a GitHub Personal Access Token (not
your password) if prompted, or sign in via the browser popup if you have
Git Credential Manager installed (it usually is, by default, with Git for
Windows).

## 4. Deploy to Streamlit Community Cloud

1. Go to https://streamlit.io/cloud and sign in with your GitHub account.
2. Click **New app**.
3. Repository: `<your-username>/churn-intelligence-platform`, branch `main`.
4. Main file path: `streamlit_app/app.py`
5. Click **Deploy**. First build takes a few minutes (it installs everything
   in `requirements.txt`).
6. Once live, you'll get a URL like
   `https://<your-app-name>.streamlit.app` — put this in your CV/LinkedIn
   and at the top of the README.

### If the deploy fails

- **Missing module error**: check `requirements.txt` at the repo root — that's
  the one Streamlit Cloud installs from.
- **FileNotFoundError for model/data files**: confirm
  `data/processed/churn_predictions_snapshot.csv` and everything under
  `ml/models/*.joblib` actually got committed (check on GitHub.com that they
  exist in the repo — they're deliberately excluded from `.gitignore` for
  exactly this reason).
- The app never tries to connect to your local MySQL — it only reads the
  committed snapshot CSV and model files, so your local database being
  offline doesn't affect the deployed app.

## 5. Add the live link + screenshots to the README

Once deployed, edit `README.md` and add near the top:

```markdown
**Live app:** https://<your-app-name>.streamlit.app
```

Also add 1-2 screenshots of the Streamlit app and the Power BI dashboard
pages (drag PNGs into `docs/screenshots/` and reference them with normal
markdown image syntax) — recruiters skim, so a visual at the top of the
README does more work than a paragraph of text.
