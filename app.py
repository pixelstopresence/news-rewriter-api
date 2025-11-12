import sys, types
if 'cgi' not in sys.modules:
    sys.modules['cgi'] = types.SimpleNamespace(
        escape=lambda s, quote=True: s
    )


from flask import Flask, jsonify
import feedparser
import requests
import random

app = Flask(__name__)

# --- Step 1: Choose your news sources (AI, UI/UX, Digital Marketing) ---
RSS_FEEDS = [
    "https://techcrunch.com/tag/artificial-intelligence/feed/",
    "https://venturebeat.com/category/ai/feed/",
    "https://uxdesign.cc/feed",
    "https://www.smashingmagazine.com/category/uxdesign/feed/",
    "https://neilpatel.com/blog/feed/",
    "https://moz.com/blog/feed/",
    "https://blog.hubspot.com/marketing/rss.xml"
]

# --- Step 2: Hugging Face public model (no API key needed) ---
HF_MODEL_URL = "https://api-inference.huggingface.co/models/humarin/chatgpt_paraphraser_on_T5_base"

def fetch_random_article():
    """Fetch a random article from the RSS feeds."""
    articles = []
    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:3]:
                title = entry.get("title", "")
                summary = entry.get("summary", "") or entry.get("description", "")
                link = entry.get("link", "")
                if title and summary:
                    articles.append({"title": title, "content": summary, "link": link})
        except Exception as e:
            print(f"Error fetching from {feed_url}: {e}")
    return random.choice(articles) if articles else None

def rewrite_content(title, content):
    """Rewrite the content using Hugging Face public model (no key)."""
    prompt = (
        f"Paraphrase and expand the following blog post to sound natural, human-like, "
        f"and engaging with around 1000 to 1500 words:\n\n"
        f"Title: {title}\n\nContent: {content}"
    )
    payload = {"inputs": prompt}
    try:
        response = requests.post(HF_MODEL_URL, json=payload, timeout=90)
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        if isinstance(result, dict) and "generated_text" in result:
            return result["generated_text"]
        if isinstance(result, str):
            return result
    except Exception as e:
        print(f"Rewrite error: {e}")
    return content

@app.route("/news/latest", methods=["GET"])
def latest_news():
    article = fetch_random_article()
    if not article:
        return jsonify({"error": "No articles found"}), 500

    rewritten = rewrite_content(article["title"], article["content"])
    final_content = f"{rewritten}\n\nSource: {article['link']}"

    return jsonify({
        "articles": [
            {
                "title": article["title"],
                "content": final_content
            }
        ]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

