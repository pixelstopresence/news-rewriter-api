from flask import Flask, jsonify
import requests
import random
from bs4 import BeautifulSoup

app = Flask(__name__)

# List of AI, UX, and Marketing blogs (you can add more)
BLOG_SOURCES = [
    "https://uxdesign.cc/",
    "https://www.marketingaiinstitute.com/blog",
    "https://www.smashingmagazine.com/category/uxdesign/",
    "https://www.creativebloq.com/ux"
]

def fetch_article_from_site(url):
    """Scrape one article link and content from a blog."""
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        links = [a["href"] for a in soup.find_all("a", href=True) if "http" in a["href"]]
        article_link = random.choice(links)
        article_html = requests.get(article_link, timeout=10, headers={"User-Agent": "Mozilla/5.0"}).text
        article_soup = BeautifulSoup(article_html, "html.parser")
        title = article_soup.find("title").text if article_soup.find("title") else "Untitled"
        paragraphs = article_soup.find_all("p")
        content = " ".join([p.text for p in paragraphs])
        return {"title": title, "content": content[:3000]}  # Limit raw text length
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
        # Add simple punctuation fixes
        text = text.replace(" .", ".").replace(" ,", ",")
        return text
    except Exception as e:
        print("Rewrite error:", e)
        return content


@app.route("/news/latest", methods=["GET"])
def get_latest():
    """Fetch a random article and rewrite it."""
    article = None
    for site in BLOG_SOURCES:
        article = fetch_article_from_site(site)
        if article:
            break

    if not article:
        return jsonify({"error": "No article found"}), 500

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


