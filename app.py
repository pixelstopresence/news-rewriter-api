from flask import Flask, jsonify
import requests
import random
from bs4 import BeautifulSoup

app = Flask(__name__)

# List of AI, UX, and Marketing blogs
BLOG_SOURCES = [
    "https://uxdesign.cc/",
    "https://www.marketingaiinstitute.com/blog",
    "https://www.smashingmagazine.com/category/uxdesign/",
    "https://www.creativebloq.com/ux"
]

# Keywords to filter relevant articles
TOPIC_KEYWORDS = ["ui/ux", "user interface", "ux design",
                  "artificial intelligence", "ai",
                  "digital marketing", "tech", "technology"]

# Blacklist to ignore irrelevant pages
BLACKLIST_KEYWORDS = ["privacy policy", "terms & conditions", 
                      "services", "pricing", "about us", "why choose", "features"]

def is_valid_article(title: str, content: str) -> bool:
    """Check if the article is relevant."""
    lc_title = title.lower()
    lc_content = content.lower()
    for kw in BLACKLIST_KEYWORDS:
        if kw in lc_title or kw in lc_content[:200]:
            return False
    if any(topic in lc_title for topic in TOPIC_KEYWORDS) and len(content.strip()) > 200:
        return True
    return False

def fetch_article_from_site(url):
    """Scrape one valid article link and content from a blog."""
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        links = [a["href"] for a in soup.find_all("a", href=True) if "http" in a["href"]]

        random.shuffle(links)  # randomize links
        for article_link in links:
            try:
                article_html = requests.get(article_link, timeout=10, headers={"User-Agent": "Mozilla/5.0"}).text
                article_soup = BeautifulSoup(article_html, "html.parser")
                title = article_soup.find("title").text if article_soup.find("title") else "Untitled"
                paragraphs = article_soup.find_all("p")
                content = " ".join([p.text for p in paragraphs])

                if is_valid_article(title, content):
                    return {"title": title, "content": content[:3000]}  # Limit raw text length
            except:
                continue  # skip broken links
        return None
    except Exception as e:
        print(f"Error fetching from {url}: {e}")
        return None

def rewrite_content(content):
    """Rewrite text in a more natural, human-like style using T5 paraphraser."""
    prompt = f"paraphrase: {content} </s>"
    payload = {"inputs": prompt}
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/Vamsi/T5_Paraphrase_Paws",
            json=payload,
            timeout=60
        )
        data = response.json()
        text = data[0]["generated_text"]
        return text.replace(" .", ".").replace(" ,", ",")
    except Exception as e:
        print("Rewrite error:", e)
        return content

@app.route("/news/latest", methods=["GET"])
def get_latest():
    """Fetch a random valid article and rewrite it."""
    article = None
    for site in BLOG_SOURCES:
        article = fetch_article_from_site(site)
        if article:
            break

    if not article:
        return jsonify({"error": "No valid article found"}), 500

    rewritten = rewrite_content(article["content"])
    return jsonify({
        "articles": [
            {
                "title": article["title"],
                "content": rewritten
            }
        ]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
