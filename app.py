import os
import re
import time
from datetime import datetime
from flask import Flask, request, jsonify
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
import torch

# Core imports
import requests
from bs4 import BeautifulSoup
from newspaper import Article, ArticleException, build as newspaper_build
import feedparser

# NLTK imports (with safe handling)
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available, using fallback")

app = Flask(__name__)

# Global variables for models (lazy loading)
summarizer = None
tokenizer = None
expander = None
stop_words = None

# Blog configuration (UI/UX, Digital Marketing, AI/Tech)
WEBSITES = {
    # UI/UX Design Blogs
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
    
    # Digital Marketing Blogs
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
    
    # AI and Technologies Blogs
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

# Initialize AI models (lazy loading for faster startup)
def load_models():
    global summarizer, tokenizer, expander
    if summarizer is None:
        device = 0 if torch.cuda.is_available() else -1
        print(f"Loading models on device: {device}")
        summarizer = pipeline("summarization", 
                             model="facebook/bart-large-cnn", 
                             device=device)
        
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        expander = T5ForConditionalGeneration.from_pretrained("t5-small")
        print("All models loaded successfully")

# Safe NLTK initialization
def init_nltk():
    global stop_words, NLTK_AVAILABLE
    
    if NLTK_AVAILABLE:
        try:
            # Check if NLTK data exists
            nltk_path = os.environ.get('NLTK_DATA', '/tmp/nltk_data')
            if not nltk.data.find('tokenizers/punkt', paths=[nltk_path]):
                print("Downloading NLTK data...")
                os.makedirs(nltk_path, exist_ok=True)
                nltk.download('punkt', download_dir=nltk_path, quiet=False)
                nltk.download('stopwords', download_dir=nltk_path, quiet=False)
            
            stop_words = set(stopwords.words('english'))
            print(f"NLTK initialized with {len(stop_words)} stopwords")
            return True
        except Exception as e:
            print(f"NLTK setup failed: {e}")
            NLTK_AVAILABLE = False
            return False
    else:
        # Fallback stopwords
        stop_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'she', 'or', 'an', 'will'
        }
        print("Using fallback stopwords")
        return True

# Initialize on startup
print("Starting News Rewriter API...")
init_nltk()
load_models()

def scrape_article(url):
    """Scrape single article using newspaper3k."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        # Validate content
        if len(article.text) < 100:
            return None
            
        return {
            'title': article.title or 'Untitled Article',
            'text': article.text,
            'authors': article.authors,
            'publish_date': str(article.publish_date) if article.publish_date else 'Unknown',
            'url': url,
            'word_count': len(article.text.split())
        }
    except Exception as e:
        print(f"Scrape error for {url}: {e}")
        return None

def fetch_rss_fallback(site_url, limit=3):
    """RSS fallback for sites that block scraping."""
    try:
        # Try common RSS paths
        for rss_path in ['/rss', '/feed', '/atom.xml', '']:
            rss_url = f"{site_url.rstrip('/')}{rss_path}"
            feed = feedparser.parse(rss_url)
            if feed.entries:
                articles = []
                for entry in feed.entries[:limit]:
                    article = {
                        'title': entry.title or 'Untitled',
                        'url': entry.link,
                        'text_preview': (getattr(entry, 'summary', '') or '')[:300] + '...',
                        'publish_date': getattr(entry, 'published', 'Unknown'),
                        'authors': [getattr(entry, 'author', '')] if hasattr(entry, 'author') else [],
                        'word_count': 100  # Estimate
                    }
                    articles.append(article)
                return articles
    except Exception as e:
        print(f"RSS fallback failed for {site_url}: {e}")
    return []

def fetch_latest_articles(site_key, num_articles=5):
    """Fetch latest articles from a specific blog."""
    if site_key not in WEBSITES:
        return []
    
    site = WEBSITES[site_key]
    print(f"Fetching from {site['name']}...")
    
    articles = []
    
    # Primary method: Newspaper3k
    try:
        source = newspaper_build(
            site['url'],
            memoize_articles=False,
            language=site['config'].get('language', 'en'),
            max_pages_to_fetch=site['config'].get('max_pages_to_fetch', 3)
        )
        
        for i, article_url in enumerate(source.article_urls()[:num_articles * 2]):  # Fetch extra for filtering
            if len(articles) >= num_articles:
                break
                
            time.sleep(1)  # Rate limiting
            art_data = scrape_article(article_url)
            
            if art_data and art_data['word_count'] > 200:
                art_data['source'] = site['name']
                art_data['category'] = site['category']
                art_data['text_preview'] = art_data['text'][:300] + '...' if len(art_data['text']) > 300 else art_data['text']
                articles.append(art_data)
                
    except Exception as e:
        print(f"Newspaper3k failed for {site_key}: {e}")
        # Fallback to RSS
        rss_articles = fetch_rss_fallback(site['url'], num_articles)
        for art in rss_articles:
            art['source'] = site['name']
            art['category'] = site['category']
            articles.append(art)
    
    # Sort by publish date (most recent first)
    try:
        articles.sort(key=lambda x: x.get('publish_date', ''), reverse=True)
    except:
        pass  # If dates are malformed, keep original order
    
    print(f"Fetched {len(articles)} articles from {site['name']}")
    return articles[:num_articles]

def fetch_top_articles(sites=None, num_per_site=3):
    """Fetch articles across multiple sites."""
    if sites is None:
        sites = list(WEBSITES.keys())
    
    all_articles = []
    for site_key in sites:
        site_articles = fetch_latest_articles(site_key, num_per_site)
        all_articles.extend(site_articles)
        time.sleep(2)  # Rate limit between sites
    
    # Deduplicate by URL
    seen_urls = set()
    unique_articles = []
    for article in all_articles:
        if article['url'] not in seen_urls:
            seen_urls.add(article['url'])
            unique_articles.append(article)
    
    return unique_articles[:15]  # Limit total results

def extract_keywords(text, top_n=10):
    """Extract SEO keywords from text."""
    if not NLTK_AVAILABLE or stop_words is None:
        # Simple fallback
        words = re.findall(r'\b[a-zA-Z]{4,15}\b', text.lower())
        from collections import Counter
        common = Counter(words).most_common(top_n)
        return [word for word, _ in common if len(word) > 3]
    
    try:
        sentences = sent_tokenize(text)
        words = [
            word.lower() for sentence in sentences 
            for word in nltk.word_tokenize(sentence)
            if word.isalnum() and word.lower() not in stop_words and len(word) > 3
        ]
        from nltk import FreqDist
        freq = FreqDist(words)
        return [word for word, _ in freq.most_common(top_n)]
    except Exception:
        # Fallback
        return extract_keywords(text, top_n)  # Recursive fallback

def generate_seo_title(original_title, keywords):
    """Generate SEO-optimized title."""
    if not keywords:
        keywords = extract_keywords(original_title)
    
    # Simple SEO formula: Keywords + Core + Year
    seo_keywords = ' '.join(keywords[:4])
    base_title = re.sub(r'[^\w\s]', '', original_title)[:60]
    
    # Create engaging title under 60 characters
    seo_title = f"{seo_keywords.title()} Guide 2025: {base_title[:40]}"
    
    # Trim to 60 chars for SEO
    return (seo_title[:60] + '...') if len(seo_title) > 60 else seo_title

def rewrite_article(text, min_words=1000):
    """Rewrite article to human-like, SEO-optimized content."""
    if len(text) < 100:
        return "Content too short for rewriting."
    
    try:
        # Step 1: Summarize to core points
        max_chunk = 1000
        chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
        summaries = []
        
        for chunk in chunks:
            if len(chunk) > 50:
                summary = summarizer(
                    chunk, 
                    max_length=150, 
                    min_length=50, 
                    do_sample=False
                )[0]['summary_text']
                summaries.append(summary)
        
        core_summary = ' '.join(summaries)
        
        # Step 2: Expand with T5 model
        if tokenizer and expander:
            inputs = tokenizer.encode(
                f"rewrite and expand for SEO: {core_summary}", 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            )
            
            with torch.no_grad():
                expanded = expander.generate(
                    inputs, 
                    max_length=1500, 
                    min_length=min_words//2, 
                    num_beams=4, 
                    early_stopping=True, 
                    do_sample=True, 
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            rewritten = tokenizer.decode(expanded[0], skip_special_tokens=True)
        else:
            # Fallback: Simple expansion
            rewritten = core_summary * 2  # Basic repetition
        
        # Step 3: Ensure minimum length and add SEO elements
        keywords = extract_keywords(rewritten)
        full_content = rewritten
        
        # Add SEO paragraphs
        full_content += f"\n\nIn 2025, key trends in {', '.join(keywords[:3])} are shaping the industry. "
        full_content += "This comprehensive guide explores the latest developments and best practices."
        
        # Iterative expansion if needed
        while len(full_content.split()) < min_words:
            expansion_prompt = f"elaborate on: {full_content[-500:]}"
            if tokenizer and expander:
                inputs = tokenizer.encode(expansion_prompt, return_tensors="pt", max_length=512, truncation=True)
                with torch.no_grad():
                    extra = expander.generate(inputs, max_length=200, do_sample=True, temperature=0.8)
                full_content += tokenizer.decode(extra[0], skip_special_tokens=True)
            else:
                full_content += " This topic continues to evolve with new insights and applications."
        
        return full_content[:min_words + 200].strip()
        
    except Exception as e:
        print(f"Rewrite error: {e}")
        # Fallback: Simple paraphrasing
        return f"Comprehensive Analysis of: {text[:300]}...\n\nThis topic explores key aspects of modern digital trends, offering insights for 2025 and beyond."

# API Routes

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'nltk_available': NLTK_AVAILABLE,
        'models_loaded': summarizer is not None,
        'timestamp': datetime.now().isoformat(),
        'websites_configured': len(WEBSITES)
    })

@app.route('/latest_articles', methods=['POST'])
def latest_articles():
    """Fetch latest articles from blogs."""
    try:
        data = request.get_json() or {}
        sites = data.get('sites', list(WEBSITES.keys()))
        num_per_site = data.get('num_per_site', 3)
        category_filter = data.get('category')  # Optional: 'UI/UX', 'Digital Marketing', 'AI/Tech'
        
        if category_filter:
            sites = [k for k, v in WEBSITES.items() if v['category'] == category_filter]
        
        articles = fetch_top_articles(sites, num_per_site)
        
        return jsonify({
            'success': True,
            'articles': articles,
            'total_articles': len(articles),
            'sources': list(set([a['source'] for a in articles])),
            'fetched_at': datetime.now().isoformat(),
            'config': {
                'sites_requested': sites,
                'num_per_site': num_per_site,
                'category_filter': category_filter
            }
        })
        
    except Exception as e:
        print(f"Latest articles error: {e}")
        return jsonify({'error': 'Failed to fetch articles', 'details': str(e)}), 500

@app.route('/rewrite', methods=['POST'])
def rewrite():
    """Rewrite a single article."""
    try:
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            # Fallback: Get latest article
            latest = fetch_latest_articles(list(WEBSITES.keys())[0], 1)
            if not latest:
                return jsonify({'error': 'No URL provided and unable to fetch latest article'}), 400
            url = latest[0]['url']
        
        # Scrape article
        article_data = scrape_article(url)
        if not article_data or not article_data['text']:
            return jsonify({'error': 'Failed to scrape or extract content from URL'}), 404
        
        # Process and rewrite
        keywords = extract_keywords(article_data['text'])
        seo_title = generate_seo_title(article_data['title'], keywords)
        rewritten_content = rewrite_article(article_data['text'])
        
        result = {
            'success': True,
            'original': {
                'title': article_data['title'],
                'url': article_data['url'],
                'word_count': article_data['word_count']
            },
            'rewritten': {
                'seo_title': seo_title,
                'content': rewritten_content,
                'word_count': len(rewritten_content.split()),
                'keywords': keywords[:10],
                'estimated_reading_time': f"{len(rewritten_content.split()) // 200} min"
            },
            'processed_at': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Rewrite error: {e}")
        return jsonify({'error': 'Processing failed', 'details': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """Simple home endpoint."""
    return jsonify({
        'message': 'News Rewriter API - Ready for UI/UX, Digital Marketing, and AI content',
        'endpoints': {
            'health': '/health',
            'latest_articles': '/latest_articles (POST)',
            'rewrite': '/rewrite (POST)'
        },
        'status': 'active'
    })

if __name__ == '__main__':
    # Production: Use gunicorn (handles $PORT)
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    app.run(host=host, port=port, debug=False)
