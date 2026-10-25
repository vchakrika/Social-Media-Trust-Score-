"""
train_models.py
Run this once to train and save both ML models.
Usage: python train_models.py
Place your CSVs in the same folder or update the paths below.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Instagram Model ─────────────────────────────────────
print("=" * 50)
print("Training Instagram Fake Account Detector...")
train_df = pd.read_csv(os.path.join(BASE, "data", "instagram.csv"))
test_df  = pd.read_csv(os.path.join(BASE, "data", "test.csv"))

IG_FEATURES = [
    'profile pic', 'nums/length username', 'fullname words',
    'nums/length fullname', 'name==username', 'description length',
    'external URL', 'private', '#posts', '#followers', '#follows'
]
IG_LABEL = 'fake'

X_train, y_train = train_df[IG_FEATURES], train_df[IG_LABEL]
X_test,  y_test  = test_df[IG_FEATURES],  test_df[IG_LABEL]

ig_pipe = Pipeline([
    ('imp',    SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf',    RandomForestClassifier(
        n_estimators=200, max_depth=10,
        random_state=42, class_weight='balanced', n_jobs=-1
    ))
])
ig_pipe.fit(X_train, y_train)
y_pred = ig_pipe.predict(X_test)
print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"  AUC      : {roc_auc_score(y_test, ig_pipe.predict_proba(X_test)[:,1]):.4f}")
print(classification_report(y_test, y_pred))

with open(os.path.join(MODEL_DIR, "instagram_model.pkl"), "wb") as f:
    pickle.dump(ig_pipe, f)
print("  ✅ Saved instagram_model.pkl")

# ─── Facebook Model ───────────────────────────────────────
print("=" * 50)
print("Training Facebook Spam Detector...")
fb_df = pd.read_csv(os.path.join(BASE, "data", "Facebook_Spam_Dataset.csv"))

FB_FEATURES = [
    '#friends', '#following', '#community', 'age',
    '#postshared', '#urlshared', '#photos/videos',
    'fpurls', 'fpphotos/videos', 'avgcomment/post',
    'likes/post', 'tags/post', '#tags/post'
]
FB_LABEL = 'Label'

Xf, yf = fb_df[FB_FEATURES], fb_df[FB_LABEL]
fb_pipe = Pipeline([
    ('imp',    SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf',    RandomForestClassifier(
        n_estimators=200, max_depth=10,
        random_state=42, class_weight='balanced', n_jobs=-1
    ))
])
fb_pipe.fit(Xf, yf)
print(f"  Train Accuracy: {fb_pipe.score(Xf, yf):.4f}")

with open(os.path.join(MODEL_DIR, "facebook_model.pkl"), "wb") as f:
    pickle.dump(fb_pipe, f)
print("  ✅ Saved facebook_model.pkl")

# ─── Feature Importances ──────────────────────────────────
ig_imp = dict(zip(IG_FEATURES, ig_pipe.named_steps['clf'].feature_importances_))
fb_imp = dict(zip(FB_FEATURES, fb_pipe.named_steps['clf'].feature_importances_))
with open(os.path.join(MODEL_DIR, "feature_importances.pkl"), "wb") as f:
    pickle.dump({'instagram': ig_imp, 'facebook': fb_imp}, f)
print("  ✅ Saved feature_importances.pkl")
print("=" * 50)
print("All models trained and saved to /model/")
