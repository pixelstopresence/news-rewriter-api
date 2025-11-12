from flask import Flask, jsonify
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

BLOG_SOURCES = [
    "https://uxdesign.cc/",
    "https://www.marketingaiinstitute.com/blog",
    "https://www.smashingmagazine.com/category/uxdesign/",
    "https://www.creativebloq.com/ux"
]

TOPIC_KEYWORDS = ["ui/ux", "user interface", "ux design",
                  "artificial intelligence", "ai",
                  "digital marketing", "tech", "technology"]

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


def fetch_article_uxdesign(url):
    """Fetch the first valid article from UXDesign.cc"""
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        articles = soup.find_all("div", class_="postArticle-content")
        for article in articles:
            link_tag = article.find_parent("a", href=True)
            if link_tag:
                link = link_tag["href"]
                article_html = requests.get(link, timeout=10, headers={"User-Agent": "Mozilla/5.0"}).text
                article_soup = BeautifulSoup(article_html, "html.parser")
                title_tag = article_soup.find("h1") or article_soup.find("title")
                title = title_tag.get_text(strip=True) if title_tag else "Untitled"
                paragraphs = article_soup.find_all("p")
                content = " ".join([p.get_text(strip=True) for p in paragraphs])
                if is_valid_article(title, content):
                    return {"title": title, "content": content[:5000]}
        return None
    except Exception as e:
        print(f"Error fetching from UXDesign: {e}")
        return None


def fetch_article_marketingai(url):
    """Fetch first valid article from Marketing AI Institute"""
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        articles = soup.find_all("article")
        for art in articles:
            link_tag = art.find("a", href=True)
            if link_tag:
                link = link_tag["href"]
                article_html = requests.get(link, timeout=10, headers={"User-Agent": "Mozilla/5.0"}).text
                article_soup = BeautifulSoup(article_html, "html.parser")
                title_tag = article_soup.find("h1") or article_soup.find("title")
                title = title_tag.get_text(strip=True) if title_tag else "Untitled"
                paragraphs = article_soup.find_all("p")
                content = " ".join([p.get_text(strip=True) for p in paragraphs])
                if is_valid_article(title, content):
                    return {"title": title, "content": content[:5000]}
        return None
    except Exception as e:
        print(f"Error fetching from Marketing AI: {e}")
        return None


def fetch_article_smashing(url):
    """Fetch first valid article from Smashing Magazine"""
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        articles = soup.find_all("article")
        for art in articles:
            link_tag = art.find("a", href=True)
            if link_tag:
                link = link_tag["href"]
                article_html = requests.get(link, timeout=10, headers={"User-Agent": "Mozilla/5.0"}).text
                article_soup = BeautifulSoup(article_html, "html.parser")
                title_tag = article_soup.find("h1") or article_soup.find("title")
                title = title_tag.get_text(strip=True) if title_tag else "Untitled"
                paragraphs = article_soup.find_all("p")
                content = " ".join([p.get_text(strip=True) for p in paragraphs])
                if is_valid_article(title, content):
                    return {"title": title, "content": content[:5000]}
        return None
    except Exception as e:
        print(f"Error fetching from Smashing: {e}")
        return None


def fetch_article_creativebloq(url):
    """Fetch first valid article from CreativeBloq"""
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        articles = soup.find_all("h2")
        for art in articles:
            link_tag = art.find_parent("a", href=True)
            if link_tag:
                link = link_tag["href"]
                article_html = requests.get(link, timeout=10, headers={"User-Agent": "Mozilla/5.0"}).text
                article_soup = BeautifulSoup(article_html, "html.parser")
                title_tag = article_soup.find("h1") or article_soup.find("title")
                title = title_tag.get_text(strip=True) if title_tag else "Untitled"
                paragraphs = article_soup.find_all("p")
                content = " ".join([p.get_text(strip=True) for p in paragraphs])
                if is_valid_article(title, content):
                    return {"title": title, "content": content[:5000]}
        return None
    except Exception as e:
        print(f"Error fetching from CreativeBloq: {e}")
        return None


def rewrite_content(content):
    """Rewrite text in a more natural, human-like style using T5 paraphraser."""
    if not content.strip():
        return content
    payload = {"inputs": f"paraphrase: {content} </s>"}
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
    """Fetch a valid article and rewrite it."""
    article = (
        fetch_article_uxdesign(BLOG_SOURCES[0])
        or fetch_article_marketingai(BLOG_SOURCES[1])
        or fetch_article_smashing(BLOG_SOURCES[2])
        or fetch_article_creativebloq(BLOG_SOURCES[3])
    )

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
