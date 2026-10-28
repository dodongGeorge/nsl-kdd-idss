# Deploying to Streamlit Community Cloud

1. Create a new empty GitHub repo, then from inside this folder:
   ```
   git init -b main
   git add -A          # includes figures/*.png
   git commit -m "NSL-KDD IDS Streamlit app"
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
   Check on GitHub that `figures/` shows 3 PNGs and 2 CSVs before deploying.
2. On share.streamlit.io choose "Create app", pick the repo, branch `main`, main file `app.py`.
3. Under Advanced settings choose Python 3.11 or newer (scikit-learn 1.8.0 needs it).
4. Deploy. `data/` and `imbalanced-learn` are only needed for retraining, not for the app.
