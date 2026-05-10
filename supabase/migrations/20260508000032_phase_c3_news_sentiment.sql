-- Phase C.3 — News RSS Scraper + Sentiment Classification
-- Stores scraped articles from Indian financial RSS feeds and
-- per-symbol sentiment scores derived from rule-based lexicon analysis.

CREATE TABLE IF NOT EXISTS news_sources (
  id SERIAL PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  rss_url TEXT NOT NULL,
  category TEXT,                              -- 'general'|'markets'|'corporate'
  active BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS news_articles (
  id BIGSERIAL PRIMARY KEY,
  source TEXT NOT NULL,
  title TEXT NOT NULL,
  link TEXT,
  published_at TIMESTAMPTZ,
  description TEXT,
  body TEXT,
  scraped_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (source, link)
);
CREATE INDEX IF NOT EXISTS idx_news_pub ON news_articles(published_at DESC);

CREATE TABLE IF NOT EXISTS news_sentiment_symbol (
  article_id BIGINT REFERENCES news_articles(id) ON DELETE CASCADE,
  symbol TEXT NOT NULL,
  sentiment NUMERIC NOT NULL,                 -- -1.0 to 1.0
  classification TEXT NOT NULL CHECK (classification IN ('POSITIVE','NEGATIVE','NEUTRAL')),
  confidence NUMERIC,                         -- 0-1
  matched_keywords JSONB,
  PRIMARY KEY (article_id, symbol)
);
CREATE INDEX IF NOT EXISTS idx_news_sentiment_symbol ON news_sentiment_symbol(symbol);

INSERT INTO schema_version (version, description)
VALUES ('phase_c3_news_sentiment', 'Phase C.3 — RSS news + per-symbol sentiment')
ON CONFLICT (version) DO NOTHING;

INSERT INTO news_sources (name, rss_url, category) VALUES
  ('moneycontrol_business', 'https://www.moneycontrol.com/rss/business.xml', 'general'),
  ('moneycontrol_markets',  'https://www.moneycontrol.com/rss/marketreports.xml', 'markets'),
  ('et_markets',            'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms', 'markets'),
  ('et_companies',          'https://economictimes.indiatimes.com/industry/rssfeeds/13352306.cms', 'corporate'),
  ('businessline_markets',  'https://www.thehindubusinessline.com/markets/feeder/default.rss', 'markets'),
  ('reuters_india',         'https://www.reuters.com/markets/asia/rss', 'general')
ON CONFLICT (name) DO UPDATE SET rss_url = EXCLUDED.rss_url, active = TRUE;
