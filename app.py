# app.py
from flask import Flask, jsonify, request
import os
import openai
import feedparser
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

app = Flask(__name__)
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Default sources (RSS feeds and example site homepages)
DEFAULT_SOURCES = [
    # RSS feeds (recommended)
    "https://uxdesign.cc/feed",
    "https://www.marketingaiinstitute.com/feed",
    "https://www.smashingmagazine.com/feed/",
    "https://www.creativebloq.com/feed",
    "https://towardsdatascience.com/feed",
    "https://medium.com/topic/artificial-intelligence/feed",
    "https://medium.com/topic/design/feed",
    # Example website homepages (will attempt to scrape article content)
    "https://uxdesign.cc/",
    "https://www.smashingmagazine.com/",
    "https://www.creativebloq.com/"
]

TOPIC_KEYWORDS = ["ui/ux", "user interface", "ux design",
                  "artificial intelligence", "ai",
                  "digital marketing", "tech", "technology"]
BLACKLIST_KEYWORDS = ["privacy policy", "terms & conditions",
                      "services", "pricing", "about us", "why choose", "features"]


# --- helpers ---
def looks_like_rss(url: str) -> bool:
    return url.lower().endswith("/feed") or "rss" in url.lower()

def is_blacklisted(text: str) -> bool:
    lc = text.lower()
    return any(kw in lc for kw in BLACKLIST_KEYWORDS)

def fetch_full_article_from_url(url: str, min_len=300):
    """Try to scrape the target URL and return main text (or None)."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, timeout=10, headers=headers)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        # Try common article selectors first
        selectors = [
            {"name": "article"},
            {"name": "div", "attrs": {"class": lambda v: v and "article" in v.lower()}},
            {"name": "div", "attrs": {"class": lambda v: v and "post" in v.lower()}},
            {"name": "main"}
        ]
        text_blocks = []
        for sel in selectors:
            found = soup.find_all(sel.get("name"), sel.get("attrs")) if sel.get("attrs") is not None else soup.find_all(sel.get("name"))
            for f in found:
                paragraphs = f.find_all("p")
                if paragraphs:
                    text = " ".join(p.get_text(strip=True) for p in paragraphs)
                    if len(text) >= min_len and not is_blacklisted(text[:200]):
                        return text
        # fallback: gather all <p> on page
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs)
        if len(text) >= min_len and not is_blacklisted(text[:200]):
            return text
        return None
    except Exception as e:
        print(f"[fetch_full_article_from_url] error for {url}: {e}")
        return None


def fetch_from_rss(feed_url: str, max_entries=5):
    """Parse RSS and return list of (title, summary, link)."""
    try:
        feed = feedparser.parse(feed_url)
        entries = []
        for entry in feed.entries[:max_entries]:
            title = getattr(entry, "title", "")
            summary = getattr(entry, "summary", "")
            link = getattr(entry, "link", "")
            if is_blacklisted(title) or is_blacklisted(summary):
                continue
            entries.append({"title": title, "summary": summary, "link": link})
        return entries
    except Exception as e:
        print(f"[fetch_from_rss] {feed_url} error: {e}")
        return []


# --- AI writer using gpt-3.5-turbo (low cost) ---
def ai_generate_article_from_seed(seed_text: str, topic: str, min_words=1000):
    """
    Use GPT-3.5 to generate an original SEO-optimized article based on seed_text/topic.
    Will not include links and should be > min_words.
    """
    # We give the model both the topic and a short seed (summary/full content) to preserve context,
    # but we instruct it to produce an original article without source links.
    system_msg = (
        "You are a professional SEO content writer. "
        "Given a topic and a short seed from an article, produce a completely original, "
        "human-like, SEO-optimized article. Do NOT include any source links or mention original sources. "
        "Make the content unique, engaging, and at least the requested length."
    )
    user_prompt = f"""
Topic: {topic}

Seed (short): {seed_text[:2000]}

Instructions:
- Produce an SEO-optimized title and article content.
- Use headings, subheadings, bullet points where helpful.
- Include practical examples and actionable tips.
- Ensure the article reads like original human writing and is plagiarism-free.
- Target length: at least {min_words} words.
- Do not include source links or mention the original source.
- Avoid copying sentences verbatim from the seed; rewrite and expand freely.
"""
    try:
        # estimate tokens: 1000 words ≈ 1400+ tokens, set max_tokens accordingly (adjust per need)
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=2200  # increase if you want longer output (and cost)
        )
        text = resp.choices[0].message.content
        return text
    except Exception as e:
        print(f"[ai_generate_article_from_seed] OpenAI error: {e}")
        return None


# --- Main endpoint ---
@app.route("/generate", methods=["POST"])
def generate():
    """
    POST JSON body:
    {
      "topic": "Your topic/title/keyword",
      "sources": ["optional list of RSS feed urls or site urls"],   # optional
      "max_articles": 1   # how many generated articles to return
    }
    """
    data = request.get_json(force=True)
    topic = data.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Please provide a topic"}), 400

    sources = data.get("sources") or DEFAULT_SOURCES
    max_articles = int(data.get("max_articles", 1))

    generated = []
    # iterate through sources, try RSS first if looks like rss
    for src in sources:
        if len(generated) >= max_articles:
            break

        if looks_like_rss(src):
            entries = fetch_from_rss(src, max_entries=5)
            for e in entries:
                if len(generated) >= max_articles:
                    break
                # Try to fetch full article from entry.link for richer seed
                seed = None
                if e.get("link"):
                    seed = fetch_full_article_from_url(e["link"]) or e.get("summary") or e.get("title")
                else:
                    seed = e.get("summary") or e.get("title")
                if not seed or is_blacklisted(seed[:200]):
                    continue
                article_text = ai_generate_article_from_seed(seed_text=seed, topic=topic)
                if article_text:
                    generated.append({
                        "topic": topic,
                        "generated_article": article_text
                    })
        else:
            # try scraping the site root for recent article(s)
            seed = fetch_full_article_from_url(src)
            if not seed:
                continue
            article_text = ai_generate_article_from_seed(seed_text=seed, topic=topic)
            if article_text:
                generated.append({
                    "topic": topic,
                    "generated_article": article_text
                })

    # If nothing generated from sources, fallback to generating from topic alone
    while len(generated) < max_articles:
        article_text = ai_generate_article_from_seed(seed_text=topic, topic=topic)
        if not article_text:
            break
        generated.append({"topic": topic, "generated_article": article_text})

    if not generated:
        return jsonify({"error": "Failed to generate article"}), 500

    return jsonify({"results": generated})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
