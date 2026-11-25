"""Tools the analyst LLM can call.

Every tool has 3 parts:
  1. a Pydantic input schema  -> the JSON the LLM must send
  2. a docstring              -> how the LLM decides WHEN to use the tool
  3. a function body          -> fetches real data and returns a plain dict
"""

import math
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import numpy as np
import requests
import yfinance as yf
from langchain_core.tools import tool
from pydantic import BaseModel, Field


def _r(x, nd: int = 4):
    """Round a number; turn NaN / inf / missing into None so the output is clean JSON."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else round(x, nd)


# ============================================================
# 1. INPUT SCHEMAS  (what the LLM is allowed to send)
# ============================================================

class TickerInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. NVDA, AAPL, MSFT")


class NewsInput(TickerInput):
    days: int = Field(default=7, ge=1, le=30, description="Only news from the last N days")
    limit: int = Field(default=10, ge=1, le=25, description="Max number of headlines")


# ============================================================
# 2. TOOLS
# ============================================================

@tool(args_schema=TickerInput)
def get_price_stats(ticker: str) -> dict:
    """Get measured facts about a stock: price performance (returns, volatility,
    drawdown), technical indicators (RSI, MACD, moving averages, 52-week range)
    and valuation/fundamentals (P/E, margins, growth, debt, analyst target).
    Use this first for any question about whether a stock is attractive."""
    try:
        t = yf.Ticker(ticker)

        # ---------- prices: ~13 months so a full 1-year return is possible ----------
        start = (datetime.now() - timedelta(days=395)).strftime("%Y-%m-%d")
        df = t.history(start=start, auto_adjust=True)
        if df.empty:
            return {"error": f"No price data for '{ticker}'. Check the ticker symbol."}
        close = df["Close"]
        last = close.iloc[-1]

        def ret(days: int):
            """Return over the last `days` trading days (21 ≈ 1 month, 252 ≈ 1 year)."""
            return _r(last / close.iloc[-days - 1] - 1) if len(close) > days else None

        # RSI-14 (Wilder): >70 overbought, <30 oversold
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rsi = 100 - 100 / (1 + gain / loss)

        # MACD (12/26 EMA) and its 9-day signal line
        macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        macd_signal = macd.ewm(span=9, adjust=False).mean()

        sma_50 = close.rolling(50).mean().iloc[-1]
        sma_200 = close.rolling(200).mean().iloc[-1]
        last_year = close.tail(252)

        # ---------- fundamentals: one call to .info ----------
        info = t.info or {}
        mcap, fcf = info.get("marketCap"), info.get("freeCashflow")
        target = info.get("targetMeanPrice")
        d2e = info.get("debtToEquity")  # yfinance gives this as a percent (14.3 = 0.143x)

        return {
            "ticker": ticker.upper(),
            "name": info.get("shortName"),
            "sector": info.get("sector"),
            "as_of": str(close.index[-1].date()),
            "price": _r(last, 2),

            # performance
            "return_1m": ret(21),
            "return_3m": ret(63),
            "return_6m": ret(126),
            "return_1y": ret(252),
            "volatility_annual": _r(close.pct_change().tail(252).std() * np.sqrt(252)),
            "max_drawdown_1y": _r((last_year / last_year.cummax() - 1).min()),

            # technicals
            "rsi_14": _r(rsi.iloc[-1], 2),
            "macd": _r(macd.iloc[-1], 3),
            "macd_signal": _r(macd_signal.iloc[-1], 3),
            "sma_50": _r(sma_50, 2),
            "sma_200": _r(sma_200, 2),
            "price_vs_sma_200": _r(last / sma_200 - 1),
            "high_52w": _r(last_year.max(), 2),
            "low_52w": _r(last_year.min(), 2),
            "pct_from_52w_high": _r(last / last_year.max() - 1),

            # valuation
            "market_cap": mcap,
            "trailing_pe": _r(info.get("trailingPE"), 2),
            "forward_pe": _r(info.get("forwardPE"), 2),
            "peg_ratio": _r(info.get("trailingPegRatio"), 2),
            "price_to_book": _r(info.get("priceToBook"), 2),
            "price_to_sales": _r(info.get("priceToSalesTrailing12Months"), 2),
            "ev_to_ebitda": _r(info.get("enterpriseToEbitda"), 2),
            "fcf_yield": _r(fcf / mcap) if fcf and mcap else None,
            "dividend_yield": _r(info.get("dividendRate", 0) / last) if info.get("dividendRate") else 0.0,

            # quality & growth
            "gross_margin": _r(info.get("grossMargins")),
            "profit_margin": _r(info.get("profitMargins")),
            "return_on_equity": _r(info.get("returnOnEquity")),
            "revenue_growth_yoy": _r(info.get("revenueGrowth")),
            "earnings_growth_yoy": _r(info.get("earningsGrowth")),

            # balance sheet & risk
            "debt_to_equity": _r(d2e / 100) if d2e is not None else None,
            "current_ratio": _r(info.get("currentRatio"), 2),
            "beta": _r(info.get("beta"), 2),

            # what analysts think
            "analyst_target_mean": _r(target, 2),
            "upside_to_target": _r(target / last - 1) if target else None,
            "analyst_rating": info.get("recommendationKey"),
            "analyst_count": info.get("numberOfAnalystOpinions"),

            "units": "returns, volatility, drawdown, margins, growth and yields are decimals "
                     "(0.12 = 12%); debt_to_equity is a ratio (0.5 = 50%)",
        }
    except Exception as e:
        return {"error": f"get_price_stats failed: {e}"}


def _yahoo_news(ticker: str) -> list[dict]:
    """~10 recent stories from Yahoo Finance (has summaries, but mixes in market-wide news)."""
    articles = []
    for item in yf.Ticker(ticker).news or []:
        c = item.get("content", item)  # yfinance nests the story under "content"
        articles.append({
            "title": c.get("title"),
            "summary": c.get("summary"),
            "source": (c.get("provider") or {}).get("displayName", "Yahoo Finance"),
            "published": datetime.fromisoformat(c["pubDate"].replace("Z", "+00:00")),
            "url": (c.get("canonicalUrl") or {}).get("url"),
        })
    return articles


def _google_news(query: str, days: int) -> list[dict]:
    """Headlines from Google News RSS (free, no API key, broad coverage, no summaries)."""
    resp = requests.get(
        "https://news.google.com/rss/search",
        params={"q": f"{query} when:{days}d", "hl": "en-US", "gl": "US", "ceid": "US:en"},
        timeout=10,
    )
    resp.raise_for_status()
    articles = []
    for item in ET.fromstring(resp.content).findall("./channel/item"):
        articles.append({
            "title": item.findtext("title"),
            "summary": None,
            "source": item.findtext("source"),
            "published": parsedate_to_datetime(item.findtext("pubDate")),
            "url": item.findtext("link"),
        })
    return articles


@tool(args_schema=NewsInput)
def get_company_news(ticker: str, days: int = 7, limit: int = 10) -> dict:
    """Get recent news headlines about the company (Yahoo Finance + Google News),
    newest first, with source and publish date. Use this to judge sentiment,
    catalysts (earnings, products, deals) and risks (lawsuits, downgrades)."""
    try:
        name = (yf.Ticker(ticker).info or {}).get("shortName") or ticker.upper()
    except Exception:
        name = ticker.upper()

    # 1. collect from both sources; one failing must not break the other
    articles, errors = [], []
    for source, fetch in [
        ("yahoo", lambda: _yahoo_news(ticker)),
        ("google", lambda: _google_news(f'"{name}" OR {ticker.upper()} stock', days)),
    ]:
        try:
            articles += fetch()
        except Exception as e:
            errors.append(f"{source}: {e}")

    # 2. keep only the last `days`, drop duplicate headlines across sources
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    seen, recent = set(), []
    for a in articles:
        key = (a["title"] or "").split(" - ")[0].lower().strip()[:80]
        if key and key not in seen and a["published"] >= cutoff:
            seen.add(key)
            recent.append(a)

    # 3. flag stories that actually mention the company ("NVIDIA Corporation" -> "nvidia")
    keywords = {ticker.lower(), name.split()[0].lower().strip(",.")}
    for a in recent:
        text = f"{a['title']} {a['summary'] or ''}".lower()
        a["mentions_company"] = any(k in text for k in keywords)

    # 4. company stories first, then newest first; trim; make dates JSON-friendly
    recent.sort(key=lambda a: (a["mentions_company"], a["published"]), reverse=True)
    recent = recent[:limit]
    for a in recent:
        a["published"] = a["published"].strftime("%Y-%m-%d %H:%M UTC")

    if not recent:
        return {"ticker": ticker.upper(), "company": name, "articles": [],
                "error": "No recent news found", "source_errors": errors}

    return {
        "ticker": ticker.upper(),
        "company": name,
        "days": days,
        "count": len(recent),
        "articles": recent,
        "source_errors": errors,
        "note": "Some headlines may be about the whole market or sector, not this company.",
    }


# The graph imports this list
TOOLS = [get_price_stats, get_company_news]
