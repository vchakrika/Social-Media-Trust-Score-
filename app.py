from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

BASE = os.path.dirname(__file__)

with open(os.path.join(BASE, "model/instagram_model.pkl"), "rb") as f:
    ig_model = pickle.load(f)
with open(os.path.join(BASE, "model/facebook_model.pkl"), "rb") as f:
    fb_model = pickle.load(f)
with open(os.path.join(BASE, "model/feature_importances.pkl"), "rb") as f:
    feature_importances = pickle.load(f)


def compute_trust_score(prob_fake, features_dict, platform):
    """Convert fake probability + signals into a 0–100 trust score."""
    base = (1 - prob_fake) * 100

    bonus = 0
    if platform == "instagram":
        if features_dict.get("profile_pic") == 1:
            bonus += 3
        if features_dict.get("external_url") == 1:
            bonus += 2
        followers = features_dict.get("followers", 0)
        posts = features_dict.get("posts", 0)
        if followers > 500:
            bonus += 3
        if posts > 20:
            bonus += 2
        ratio = features_dict.get("nums_length_username", 0)
        if ratio < 0.1:
            bonus += 2
        if features_dict.get("name_eq_username") == 1:
            bonus += 2  # Name matching username = consistent real identity
    elif platform == "facebook":
        if features_dict.get("friends", 0) > 200:
            bonus += 3
        if features_dict.get("community", 0) > 100:
            bonus += 2

    score = min(100, max(0, base + bonus))
    return round(score, 1)


def get_risk_flags(features_dict, platform, prob_fake):
    flags = []
    if platform == "instagram":
        ratio = features_dict.get("nums_length_username", 0)
        if ratio > 0.4:
            flags.append({"icon": "⚠️", "text": "Username contains many numbers (bot-like pattern)", "severity": "high"})
        if features_dict.get("profile_pic") == 0:
            flags.append({"icon": "⚠️", "text": "No profile picture detected", "severity": "medium"})
        followers = features_dict.get("followers", 0)
        following = features_dict.get("following", 0)
        if followers > 0 and following / max(followers, 1) > 10:
            flags.append({"icon": "⚠️", "text": "Follows far more accounts than followers (suspicious ratio)", "severity": "high"})
        if features_dict.get("posts", 0) == 0:
            flags.append({"icon": "⚠️", "text": "No posts on account", "severity": "medium"})
        if features_dict.get("description_length", 0) < 5:
            flags.append({"icon": "ℹ️", "text": "Very short or empty bio", "severity": "low"})
    elif platform == "facebook":
        if features_dict.get("fpurls", 0) > 0.7:
            flags.append({"icon": "⚠️", "text": "High proportion of URL-only posts (spam indicator)", "severity": "high"})
        if features_dict.get("urlshared", 0) > 500:
            flags.append({"icon": "⚠️", "text": "Extremely high number of URLs shared", "severity": "high"})
        if features_dict.get("friends", 0) < 20:
            flags.append({"icon": "⚠️", "text": "Very few friends on account", "severity": "medium"})
        if features_dict.get("avgcomment", 0) < 0.1:
            flags.append({"icon": "ℹ️", "text": "Very low average comments per post", "severity": "low"})

    if prob_fake > 0.8:
        flags.insert(0, {"icon": "🚨", "text": "Model confidence of fake account is very high (>80%)", "severity": "critical"})

    return flags


@app.route("/")
def index():
    ig_importances = sorted(feature_importances["instagram"].items(), key=lambda x: -x[1])
    fb_importances = sorted(feature_importances["facebook"].items(), key=lambda x: -x[1])
    return render_template("index.html", ig_importances=ig_importances, fb_importances=fb_importances)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    platform = data.get("platform", "instagram")

    if platform == "instagram":
        features = [
            float(data.get("profile_pic", 0)),
            float(data.get("nums_length_username", 0)),
            float(data.get("fullname_words", 0)),
            float(data.get("nums_length_fullname", 0)),
            float(data.get("name_eq_username", 0)),
            float(data.get("description_length", 0)),
            float(data.get("external_url", 0)),
            float(data.get("private", 0)),
            float(data.get("posts", 0)),
            float(data.get("followers", 0)),
            float(data.get("following", 0)),
        ]
        X = np.array(features).reshape(1, -1)
        prob_fake = ig_model.predict_proba(X)[0][1]
        feat_dict = {
            "profile_pic": float(data.get("profile_pic", 0)),
            "nums_length_username": float(data.get("nums_length_username", 0)),
            "external_url": float(data.get("external_url", 0)),
            "followers": float(data.get("followers", 0)),
            "following": float(data.get("following", 0)),
            "posts": float(data.get("posts", 0)),
            "description_length": float(data.get("description_length", 0)),
            "name_eq_username": float(data.get("name_eq_username", 0)),
        }
    else:
        features = [
            float(data.get("friends", 0)),
            float(data.get("following", 0)),
            float(data.get("community", 0)),
            float(data.get("age", 0)),
            float(data.get("postshared", 0)),
            float(data.get("urlshared", 0)),
            float(data.get("photos_videos", 0)),
            float(data.get("fpurls", 0)),
            float(data.get("fpphotos", 0)),
            float(data.get("avgcomment", 0)),
            float(data.get("likes_post", 0)),
            float(data.get("tags_post", 0)),
            float(data.get("num_tags", 0)),
        ]
        X = np.array(features).reshape(1, -1)
        prob_fake = fb_model.predict_proba(X)[0][1]
        feat_dict = {
            "friends": float(data.get("friends", 0)),
            "community": float(data.get("community", 0)),
            "fpurls": float(data.get("fpurls", 0)),
            "urlshared": float(data.get("urlshared", 0)),
            "avgcomment": float(data.get("avgcomment", 0)),
        }

    is_fake = bool(prob_fake >= 0.5)
    trust_score = compute_trust_score(prob_fake, feat_dict, platform)
    risk_flags = get_risk_flags(feat_dict, platform, prob_fake)

    if trust_score >= 75:
        verdict = "Likely Genuine"
        verdict_class = "genuine"
    elif trust_score >= 45:
        verdict = "Suspicious"
        verdict_class = "suspicious"
    else:
        verdict = "Likely Fake"
        verdict_class = "fake"

    return jsonify({
        "trust_score": trust_score,
        "prob_fake": round(float(prob_fake) * 100, 1),
        "prob_real": round((1 - float(prob_fake)) * 100, 1),
        "is_fake": is_fake,
        "verdict": verdict,
        "verdict_class": verdict_class,
        "risk_flags": risk_flags,
        "platform": platform
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
