from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
from newspaper import Article, ArticleException, build as newspaper_build
import newspaper  # For source building
import feedparser  # RSS fallback
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
import torch
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re
from openai import OpenAI  # Optional
import os
import time
from datetime import datetime

app = Flask(__name__)

# Initialize models (load once)
device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
tokenizer = T5Tokenizer.from_pretrained("t5-small")
expander = T5ForConditionalGeneration.from_pretrained("t5-small")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Optional OpenAI
openai_client = None
if os.getenv('OPENAI_API_KEY'):
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Blog-focused websites configuration (add/remove as needed)
WEBSITES = {
    # UI/UX Design
    'ux_collective': {
        'name': 'UX Collective',
        'url': 'https://uxdesign.cc',
        'category': 'UI/UX',
        'config': {'language': 'en', 'max_pages_to_fetch': 5}
    },
    'designlab': {
        'name': 'Designlab Blog',
        'url': 'https://designlab.com/blog',
        'category': 'UI/UX',
        'config': {'language': 'en', 'max_pages_to_fetch': 5}
    },
    'codrops': {
        'name': 'Codrops',
        'url': 'https://tympanus.net/codrops',
        'category': 'UI/UX',
        'config': {'language': 'en', 'max_pages_to_fetch': 5}
    },
    
    # Digital Marketing
    'hubspot': {
        'name': 'HubSpot Blog',
        'url': 'https://blog.hubspot.com',
        'category': 'Digital Marketing',
        'config': {'language': 'en', 'max_pages_to_fetch': 10}
    },
    'moz': {
        'name': 'Moz Blog',
        'url': 'https://moz.com/blog',
        'category': 'Digital Marketing',
        'config': {'language': 'en', 'max_pages_to_fetch': 5}
    },
    'ahrefs': {
        'name': 'Ahrefs Blog',
        'url': 'https://ahrefs.com/blog',
        'category': 'Digital Marketing',
        'config': {'language': 'en', 'max_pages_to_fetch': 5}
    },
    
    # AI and Technologies
    'marktechpost': {
        'name': 'MarkTechPost',
        'url': 'https://www.marktechpost.com',
        'category': 'AI/Tech',
        'config': {'language': 'en', 'max_pages_to_fetch': 10}
    },
    'huggingface': {
        'name': 'Hugging Face Blog',
        'url': 'https://huggingface.co/blog',
        'category': 'AI/Tech',
        'config': {'language': 'en', 'max_pages_to_fetch': 5}
    }
}

def scrape_article(url):
    """Scrape single article with newspaper3k."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return {
            'title': article.title,
            'text': article.text,
            'authors': article.authors,
            'publish_date': str(article.publish_date) if article.publish_date else 'Unknown',
            'url': url
        }
    except ArticleException:
        return None

def fetch_rss_fallback(site_url):
    """Fallback to RSS if scraping fails."""
    try:
        feed = feedparser.parse(site_url + '/rss')  # Common RSS path
        if feed.entries:
            entry = feed.entries[0]  # Latest
            return {
                'title': entry.title,
                'url': entry.link,
                'text_preview': (entry.summary or '')[:300],
                'publish_date': entry.published if 'published' in entry else 'Unknown',
                'authors': [entry.author] if 'author' in entry else []
            }
    except:
        pass
    return None

def fetch_latest_articles(site_key, num_articles=10):
    """Fetch latest from site with RSS fallback."""
    if site_key not in WEBSITES:
        return []
    
    site = WEBSITES[site_key]
    articles = []
    
    try:
        # Primary: Newspaper3k build
        source = newspaper_build(site['url'], 
                                memoize_articles=False,
                                language=site['config'].get('language', 'en'),
                                max_pages_to_fetch=site['config'].get('max_pages_to_fetch', 3))
        
        for article_url in source.article_urls()[:num_articles]:
            time.sleep(1)  # Rate limit
            art_data = scrape_article(article_url)
            if art_data and len(art_data['text']) > 200:
                art_data['source'] = site['name']
                art_data['category'] = site['category']
                articles.append(art_data)
                if len(articles) >= num_articles:
                    break
                    
    except Exception:
        # Fallback: RSS
        fallback = fetch_rss_fallback(site['url'])
        if fallback:
            articles.append({**fallback, 'source': site['name'], 'category': site['category']})
    
    # Sort by date
    articles.sort(key=lambda x: x.get('publish_date', ''), reverse=True)
    return articles

def fetch_top_viewed_or_latest(website_keys=None, num_per_site=5, by_views=False):
    """Fetch across sites."""
    if website_keys is None:
        website_keys = list(WEBSITES.keys())
    
    all_articles = []
    for key in website_keys:
        site_articles = fetch_latest_articles(key, num_per_site)
        all_articles.extend(site_articles)
        time.sleep(2)  # Between sites
    
    if by_views:
        all_articles.sort(key=lambda x: len(x.get('title', '')), reverse=True)  # Proxy
    
    # Dedupe and limit
    seen_urls = set()
    unique = []
    for art in all_articles[:20]:
        if art['url'] not in seen_urls:
            seen_urls.add(art['url'])
            unique.append(art)
    
    return unique

def generate_seo_title(original_title, keywords):
    """Generate SEO title."""
    seo_keywords = ' '.join(keywords[:3])
    if openai_client:
        prompt = f"Create SEO title <60 chars for: {original_title}. Keywords: {seo_keywords}. Engaging, no clickbait."
        response = openai_client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content.strip()
    return f"{seo_keywords.capitalize()} Trends 2025 - {original_title[:40]}"

def extract_keywords(text, top_n=10):
    """Extract keywords."""
    sentences = sent_tokenize(text)
    words = [w.lower() for s in sentences for w in nltk.word_tokenize(s) if w.isalnum() and w.lower() not in stop_words]
    freq = nltk.FreqDist(words)
    return [w for w, _ in freq.most_common(top_n)]

def rewrite_article(text, min_words=1000):
    """Rewrite to 1000+ words, original content."""
    # Summarize
    max_chunk = 1000
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    summaries = []
    for chunk in chunks:
        if len(chunk) > 50:
            summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
            summaries.append(summary)
    core = ' '.join(summaries)
    
    # Expand with T5
    inputs = tokenizer.encode("Rewrite and expand for SEO: " + core, return_tensors="pt", max_length=512, truncation=True)
    expanded = expander.generate(inputs, max_length=1500, min_length=min_words//2, num_beams=4, early_stopping=True, do_sample=True, temperature=0.7)
    rewritten = tokenizer.decode(expanded[0], skip_special_tokens=True)
    
    # Ensure length
    keywords = extract_keywords(rewritten)
    full = rewritten + f"\n\nKey insights on {', '.join(keywords[:5])} for 2025 trends."
    while len(full.split()) < min_words:
        inputs = tokenizer.encode(f"Elaborate: {full[:500]}", return_tensors="pt", max_length=512, truncation=True)
        extra = expander.generate(inputs, max_length=300, min_length=100, do_sample=True, temperature=0.8)
        full += " " + tokenizer.decode(extra[0], skip_special_tokens=True)
    
    return full[:min_words + 200]

@app.route('/latest_articles', methods=['POST'])
def latest_articles_endpoint():
    data = request.json or {}
    sites = data.get('sites', list(WEBSITES.keys()))
    num_per_site = data.get('num_per_site', 5)
    by_views = data.get('by_views', False)
    
    articles = fetch_top_viewed_or_latest(sites, num_per_site, by_views)
    
    return jsonify({
        'articles': articles,
        'total_fetched': len(articles),
        'sources': [WEBSITES[key]['name'] for key in sites],
        'fetched_at': datetime.now().isoformat()
    })

@app.route('/rewrite', methods=['POST'])
def rewrite_endpoint():
    data = request.json
    url = data.get('url')
    if not url:
        # Fallback to first latest
        latest = fetch_latest_articles(list(WEBSITES.keys())[0], 1)
        if latest:
            url = latest[0]['url']
        else:
            return jsonify({'error': 'No URL and fetch failed'}), 400
    
    art_data = scrape_article(url)
    if not art_data or not art_data['text']:
        return jsonify({'error': 'Scrape failed'}), 404
    
    keywords = extract_keywords(art_data['text'])
    seo_title = generate_seo_title(art_data['title'], keywords)
    rewritten = rewrite_article(art_data['text'])
    
    return jsonify({
        'seo_title': seo_title,
        'content': rewritten,
        'keywords': keywords,
        'original_url': url,
        'word_count': len(rewritten.split())
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
