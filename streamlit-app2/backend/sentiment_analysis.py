#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
reglaw_social_sentiment_temporal_v3_llmdate_cap50.py

Version TEMPORELLE + LLM pour la date + CAP PAR ENTREPRISE

Objectifs ajoutés par rapport à v2 :
- limiter le nombre de posts/commentaires/articles CONSERVÉS par entreprise (ex: 50)
- scorer les posts par pertinence et ne garder que les plus liés à la directive
- recalculer les agrégats de sentiment APRÈS filtrage
- éviter de continuer à scraper si on a déjà assez de bons posts pour une entreprise

ENV nouveaux :
- MAX_POSTS_PER_COMPANY (default=50)
- EARLY_STOP_FACTOR (default=1.5) → on arrête de requêter quand on a 1.5 × max dans le buffer
"""

import os
import re
import io
import glob
import json
import time
import datetime
import unicodedata
from hashlib import md5
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

import boto3

# =========================================================
# CONFIG GLOBALE
# =========================================================

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
DIRECTIVES_DIR = os.getenv("DIRECTIVES_DIR", "/home/sagemaker-user/shared/directives")
SP500_PATH = os.getenv("SP500_PATH", "/home/sagemaker-user/shared/2025-08-15_composition_sp500.csv")
OUTPUT_BASE = os.getenv("OUTPUT_BASE", "/home/sagemaker-user/shared/sentiment_analysis")

MAX_COMPANIES = int(os.getenv("MAX_COMPANIES", "50"))     # 0 = toutes
POSTS_PER_QUERY = int(os.getenv("POSTS_PER_QUERY", "30"))

MAX_POSTS_PER_COMPANY = int(os.getenv("MAX_POSTS_PER_COMPANY", "30"))
EARLY_STOP_FACTOR = float(os.getenv("EARLY_STOP_FACTOR", "1.5"))  # on arrête de requêter quand buffer >= factor * max

# mode temporel : par défaut on veut la date de la loi
TEMPORAL_MODE = os.getenv("TEMPORAL_MODE", "law_date").lower()

# fenêtre autour de la loi
IMPLEMENTATION_BEFORE_DAYS = int(os.getenv("IMPLEMENTATION_BEFORE_DAYS", "7"))
IMPLEMENTATION_AFTER_DAYS = int(os.getenv("IMPLEMENTATION_AFTER_DAYS", "14"))

# fallback "continu" si rien
DAYS_BACK = int(os.getenv("DAYS_BACK", "30"))
PUSHSHIFT_RECENT_AFTER = f"{DAYS_BACK}d"

ENABLE_REDDIT = os.getenv("ENABLE_REDDIT", "1") == "1"
ENABLE_HN = os.getenv("ENABLE_HN", "1") == "1"
ENABLE_MASTODON = os.getenv("ENABLE_MASTODON", "1") == "1"

# poids
WEIGHT_STRONG = float(os.getenv("WEIGHT_STRONG", "1.0"))
WEIGHT_POLICY = float(os.getenv("WEIGHT_POLICY", "0.7"))
WEIGHT_WEAK = float(os.getenv("WEIGHT_WEAK", "0.35"))
WEIGHT_VERY_WEAK = float(os.getenv("WEIGHT_VERY_WEAK", "0.1"))

# LLM
USE_LLM_KEYWORDS = os.getenv("USE_LLM_KEYWORDS", "1") == "1"
USE_LLM_DATES = os.getenv("USE_LLM_DATES", "1") == "1"

BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-haiku-20241022-v1:0")
BEDROCK_MODEL_ID_DATES = os.getenv("BEDROCK_MODEL_ID_DATES", BEDROCK_MODEL_ID)

# AWS clients
comprehend = boto3.client("comprehend", region_name=AWS_REGION)
translate = boto3.client("translate", region_name=AWS_REGION) if os.getenv("USE_TRANSLATE", "1") == "1" else None
bedrock_rt = boto3.client("bedrock-runtime", region_name=AWS_REGION)

# Pushshift endpoints
PUSHSHIFT_SUBMISSIONS = [
    "https://api.pushshift.io/reddit/search/submission",
    "https://api.pullpush.io/reddit/search/submission",
]
PUSHSHIFT_COMMENTS = [
    "https://api.pushshift.io/reddit/search/comment",
    "https://api.pullpush.io/reddit/search/comment",
]

# Mastodon
MASTODON_INSTANCES = [
    "mastodon.social",
    "mstdn.social",
]

# petits mois pour fallback regex
MONTHS_EN = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12
}
MONTHS_FR = {
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4,
    "mai": 5, "juin": 6, "juillet": 7, "août": 8, "aout": 8,
    "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12
}

# =========================================================
# UTILS GÉNÉRAUX
# =========================================================

def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    value = re.sub(r"[-\s]+", "-", value)
    return value


def list_directives(path: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(path, "*")))
    out = []
    for f in files:
        base = os.path.basename(f).lower()
        if base.startswith("readme") or base.endswith(".md"):
            continue
        out.append(f)
    return out


def read_directive_text(path: str) -> str:
    with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    if path.lower().endswith(".xml"):
        soup = BeautifulSoup(raw, "xml")
    else:
        soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "meta", "link"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)


def guess_title_from_filename(path: str) -> str:
    base = os.path.basename(path)
    base = re.sub(r"\.(html?|xml)$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"^\d+\.", "", base).strip()
    return base


def detect_language(text: str) -> str:
    try:
        resp = comprehend.detect_dominant_language(Text=text[:4000])
        langs = resp.get("Languages", [])
        if not langs:
            return "unknown"
        langs = sorted(langs, key=lambda x: x["Score"], reverse=True)
        return langs[0]["LanguageCode"]
    except Exception:
        return "unknown"


def translate_to_en(text: str, src: str) -> str:
    if not translate or src == "en":
        return text
    try:
        resp = translate.translate_text(Text=text[:4500], SourceLanguageCode=src, TargetLanguageCode="en")
        return resp.get("TranslatedText", text)
    except Exception:
        return text

# =========================================================
# LLM : EXTRACTION DES DATES
# =========================================================

LLM_DATE_PROMPT_BASE = (
    "You are an assistant that extracts DATES from an EU/US/CA regulation or directive.\n"
    "You MUST answer in STRICT JSON, no explanation, no markdown.\n\n"
    "You will receive:\n"
    "- the official title\n"
    "- the English version (or translation) of the directive\n"
    "Your task is to find the dates that matter for when the obligations start.\n\n"
    "Return EXACTLY this JSON:\n"
    "{\n"
    '  "application_date": "YYYY-MM-DD or null",\n'
    '  "entry_into_force_date": "YYYY-MM-DD or null",\n'
    '  "transposition_deadline": "YYYY-MM-DD or null",\n'
    '  "notes": "short note if needed"\n'
    "}\n\n"
    "Definitions:\n"
    "- application_date: when the rules start to apply to companies / market (e.g. \"shall apply from 28 May 2022\")\n"
    "- entry_into_force_date: when the act formally enters into force (often 20 days after publication)\n"
    "- transposition_deadline: when Member States must transpose the directive\n\n"
    "If you don't find a date for a field, put null.\n"
)

def _try_parse_iso_date(s: str) -> Optional[datetime.date]:
    if not s or s.lower() == "null":
        return None
    try:
        return datetime.date.fromisoformat(s.strip())
    except Exception:
        return None

def _extract_json_from_text(text: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

def call_bedrock_for_dates(title: str, directive_en: str) -> Optional[dict]:
    if not USE_LLM_DATES:
        return None

    prompt = (
        LLM_DATE_PROMPT_BASE
        + f"Title: {title}\n"
        + "Directive (english or translated) excerpt:\n"
        + directive_en[:5000]
    )

    try:
        resp = bedrock_rt.invoke_model(
            modelId=BEDROCK_MODEL_ID_DATES,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 600,
                "temperature": 0.0,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
        )
        raw = resp["body"].read().decode("utf-8")

        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "content" in obj:
                txt = obj["content"][0]["text"]
                extracted = _extract_json_from_text(txt)
                return extracted
            return obj
        except json.JSONDecodeError:
            extracted = _extract_json_from_text(raw)
            return extracted
    except Exception as e:
        print(f"[LLM-DATES] error calling Bedrock: {e}")
        return None

# =========================================================
# FALLBACK REGEX
# =========================================================

def _parse_date_en(day: str, month: str, year: str) -> Optional[datetime.date]:
    m = MONTHS_EN.get(month.lower())
    if not m:
        return None
    return datetime.date(int(year), m, int(day))

def _parse_date_fr(day: str, month: str, year: str) -> Optional[datetime.date]:
    m = MONTHS_FR.get(month.lower())
    if not m:
        return None
    return datetime.date(int(year), m, int(day))

def guess_implementation_date_regex(text: str) -> Optional[datetime.date]:
    m = re.search(r"(apply|applies|shall apply)\s+(from|as of)\s+(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", text, flags=re.IGNORECASE)
    if m:
        d, month, y = m.group(3), m.group(4), m.group(5)
        dt = _parse_date_en(d, month, y)
        if dt:
            return dt

    m = re.search(r"(s'applique|est applicable|sera applicable)\s+(?:à partir du|à compter du|à compter de|depuis le)\s+(\d{1,2})\s+([A-Za-zéûôîàù]+)\s+(\d{4})", text, flags=re.IGNORECASE)
    if m:
        d, month, y = m.group(2), m.group(3), m.group(4)
        dt = _parse_date_fr(d, month, y)
        if dt:
            return dt

    m = re.search(r"(enter into force|enters into force|entre en vigueur)\s+(on|le)\s+(\d{1,2})\s+([A-Za-zéûôîàù]+)\s+(\d{4})", text, flags=re.IGNORECASE)
    if m:
        d, month, y = m.group(3), m.group(4), m.group(5)
        dt_en = _parse_date_en(d, month, y)
        if dt_en:
            return dt_en
        dt_fr = _parse_date_fr(d, month, y)
        if dt_fr:
            return dt_fr

    return None

# =========================================================
# CONSTRUCTION FENÊTRE
# =========================================================

def compute_temporal_window_via_llm(directive_title: str,
                                    directive_text: str,
                                    directive_lang: str) -> Tuple[object, Optional[int], Optional[int], Optional[int], Optional[int], str]:
    if TEMPORAL_MODE != "law_date":
        return (PUSHSHIFT_RECENT_AFTER, None, None, None, "recent")

    directive_en = translate_to_en(directive_text, directive_lang)

    llm_dates = call_bedrock_for_dates(directive_title, directive_en)

    chosen_date: Optional[datetime.date] = None

    if llm_dates:
        app = _try_parse_iso_date(llm_dates.get("application_date"))
        eif = _try_parse_iso_date(llm_dates.get("entry_into_force_date"))
        trans = _try_parse_iso_date(llm_dates.get("transposition_deadline"))
        chosen_date = app or eif or trans

    if not chosen_date:
        chosen_date = guess_implementation_date_regex(directive_text)

    if not chosen_date:
        return (PUSHSHIFT_RECENT_AFTER, None, None, None, "recent")

    start_dt = chosen_date - datetime.timedelta(days=IMPLEMENTATION_BEFORE_DAYS)
    end_dt = chosen_date + datetime.timedelta(days=IMPLEMENTATION_AFTER_DAYS)

    start_ts = int(datetime.datetime.combine(start_dt, datetime.time.min).timestamp())
    end_ts = int(datetime.datetime.combine(end_dt, datetime.time.max).timestamp())

    return (start_ts, end_ts, start_ts, end_ts, "law_date")

# =========================================================
# LLM KEYWORDING
# =========================================================

LLM_PROMPT_TEMPLATE = """You are an assistant that extracts search keywords from an EU/US/CA regulation text.

You will receive:
1) the official title of the directive/regulation
2) the English translation of its body (possibly truncated)

You must output STRICT JSON, no explanation, no markdown.

Rules:
- detect official / close names (e.g. "Directive (EU) 2019/2161", "Omnibus Directive", "EU consumer modernization directive")
- detect policy / regulatory terms related to consumer protection, sanctions, unfair commercial practices, transparency
- detect how people might talk about this on Reddit/Hacker News/Mastodon (short phrases, lowercase ok)
- detect obvious noise terms for tech companies (GPU, driver, ARM bid) so that we can downweight them
- add short multilingual variants if the original language is French

Return JSON with exactly these keys:
{{
  "exact_mentions": [],
  "policy_terms": [],
  "social_phrases": [],
  "anti_noise": [],
  "lang_variants": []
}}

If some list is empty, return [].
Title: {title}
Language: {lang}
Directive_english_excerpt:
{body}
"""

def call_bedrock_for_keywords(title: str, directive_en: str, lang: str) -> Optional[Dict[str, List[str]]]:
    if not USE_LLM_KEYWORDS:
        return None

    prompt = LLM_PROMPT_TEMPLATE.format(
        title=title,
        body=directive_en[:5000],
        lang=lang
    )

    try:
        resp = bedrock_rt.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1200,
                "temperature": 0.2,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
        )
        raw = resp["body"].read().decode("utf-8")

        try:
            direct_json = json.loads(raw)
            if isinstance(direct_json, dict) and "content" in direct_json:
                txt = direct_json["content"][0]["text"]
                extracted = _extract_json_from_text(txt)
                return extracted
            return direct_json
        except json.JSONDecodeError:
            extracted = _extract_json_from_text(raw)
            return extracted
    except Exception as e:
        print(f"[LLM] error calling Bedrock for keywords: {e}")
        return None


def build_dynamic_lexicon(directive_title: str, directive_text: str, directive_lang: str) -> Dict[str, List[str]]:
    directive_en = translate_to_en(directive_text, directive_lang)

    if USE_LLM_KEYWORDS:
        llm_res = call_bedrock_for_keywords(directive_title, directive_en, directive_lang)
    else:
        llm_res = None

    if llm_res:
        return {
            "exact_mentions": [x.strip() for x in llm_res.get("exact_mentions", []) if x.strip()],
            "policy_terms": [x.strip() for x in llm_res.get("policy_terms", []) if x.strip()],
            "social_phrases": [x.strip() for x in llm_res.get("social_phrases", []) if x.strip()],
            "anti_noise": [x.strip() for x in llm_res.get("anti_noise", []) if x.strip()],
            "lang_variants": [x.strip() for x in llm_res.get("lang_variants", []) if x.strip()],
        }

    # fallback
    print("[LLM] fallback to static small lexicon")
    base = [
        directive_title,
        "EU consumer law",
        "EU consumer protection rules",
        "unfair commercial practices",
        "EU directive",
        "EU regulation",
        "European Commission",
        "EU sanctions",
    ]
    return {
        "exact_mentions": base,
        "policy_terms": [
            "consumer protection", "consumer rights", "price reduction", "transparency", "online marketplace"
        ],
        "social_phrases": [
            "eu fines", "new eu rules", "eu cracked down", "compliance with eu", "european consumer law"
        ],
        "anti_noise": [
            "gpu", "rtx", "driver", "arm bid", "nvidia driver", "4090"
        ],
        "lang_variants": [
            directive_title,
        ],
    }

# =========================================================
# COMPANIES
# =========================================================

def load_companies(path: str, limit: int = 0) -> List[Dict[str, str]]:
    df = pd.read_csv(path)
    name_cols = [c for c in df.columns if "name" in c.lower() or "company" in c.lower()]
    ticker_cols = [c for c in df.columns if "ticker" in c.lower() or "symbol" in c.lower()]
    companies = []
    for _, row in df.iterrows():
        companies.append({
            "name": str(row[name_cols[0]]) if name_cols else "",
            "ticker": str(row[ticker_cols[0]]) if ticker_cols else "",
        })
    if limit and limit > 0:
        companies = companies[:limit]
    return companies

# =========================================================
# QUERY BUILDER
# =========================================================

def build_queries_for_company(lex: Dict[str, List[str]], company_name: str, ticker: str) -> List[str]:
    q: List[str] = []
    for t in lex["exact_mentions"]:
        if " " in t:
            q.append(f'"{t}" {company_name}')
            if ticker:
                q.append(f'"{t}" {ticker}')
        else:
            q.append(f"{t} {company_name}")
            if ticker:
                q.append(f"{t} {ticker}")

    for p in lex["policy_terms"]:
        q.append(f'"{p}" {company_name}')
        if ticker:
            q.append(f'"{p}" {ticker}')

    for s in lex["social_phrases"]:
        q.append(f'"{s}" {company_name}')
        if ticker:
            q.append(f'"{s}" {ticker}')

    q.append(company_name)
    if ticker:
        q.append(ticker)

    final, seen = [], set()
    for x in q:
        x = x.strip()
        if not x or x in seen:
            continue
        seen.add(x)
        final.append(x)
    return final

# =========================================================
# SENTIMENT
# =========================================================

def detect_targeted_or_doc_sentiment(text: str, company_name: str) -> Dict[str, Any]:
    try:
        lang_resp = comprehend.detect_dominant_language(Text=text[:4000])
        langs = sorted(lang_resp.get("Languages", []), key=lambda x: x["Score"], reverse=True)
        lang = langs[0]["LanguageCode"] if langs else "en"
    except Exception:
        lang = "en"

    if lang != "en" and translate:
        text_en = translate_to_en(text, lang)
    else:
        text_en = text

    try:
        resp = comprehend.detect_targeted_sentiment(Text=text_en, LanguageCode="en")
        comp_low = company_name.lower()
        for ent in resp.get("Entities", []):
            if comp_low in ent.get("Text", "").lower():
                return {
                    "Sentiment": ent.get("Sentiment", "NEUTRAL"),
                    "SentimentScore": ent.get("SentimentScore", {})
                }
    except Exception:
        pass

    try:
        resp = comprehend.detect_sentiment(Text=text_en, LanguageCode="en")
        return resp
    except Exception:
        return {"Sentiment": "UNKNOWN", "SentimentScore": {}}

# =========================================================
# FETCHERS
# =========================================================

def reddit_pushshift_generic(endpoints: List[str], query: str, size: int,
                             after: Optional[Any] = None,
                             before: Optional[int] = None) -> List[Dict[str, Any]]:
    for ep in endpoints:
        try:
            params = {"q": query, "size": size, "sort": "desc"}
            if after is not None:
                params["after"] = after
            if before is not None:
                params["before"] = before
            r = requests.get(ep, params=params, timeout=20)
            if r.status_code == 200:
                data = r.json().get("data", [])
                if data:
                    return data
        except Exception as e:
            print(f"[REDDIT] endpoint {ep} error: {e}")
    return []

def reddit_submissions(query: str, size: int = 30, after=None, before=None) -> List[Dict[str, Any]]:
    return reddit_pushshift_generic(PUSHSHIFT_SUBMISSIONS, query, size, after, before)

def reddit_comments(query: str, size: int = 30, after=None, before=None) -> List[Dict[str, Any]]:
    return reddit_pushshift_generic(PUSHSHIFT_COMMENTS, query, size, after, before)

def hn_search(query: str, hits_per_page: int = 30) -> List[Dict[str, Any]]:
    url = "https://hn.algolia.com/api/v1/search"
    try:
        r = requests.get(
            url,
            params={"query": query, "tags": "story", "hitsPerPage": hits_per_page},
            timeout=15
        )
        if r.status_code == 200:
            return r.json().get("hits", [])
    except Exception as e:
        print(f"[HN] error: {e}")
    return []

def is_hn_in_window(hit: Dict[str, Any], start_ts: Optional[int], end_ts: Optional[int], days_back: int) -> bool:
    ts = hit.get("created_at_i")
    if not ts:
        return False
    if start_ts is not None and end_ts is not None:
        return start_ts <= ts <= end_ts
    dt = datetime.datetime.utcfromtimestamp(ts)
    age = (datetime.datetime.utcnow() - dt).days
    return age <= days_back

def fetch_from_mastodon_instance(instance: str, query: str, limit: int = 30) -> List[str]:
    url = f"https://{instance}/api/v2/search"
    try:
        r = requests.get(url, params={"q": query, "limit": limit, "resolve": "true"}, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        out = []
        for st in data.get("statuses", []):
            content_html = st.get("content", "")
            soup = BeautifulSoup(content_html, "html.parser")
            txt = soup.get_text(" ", strip=True)
            if txt:
                out.append(txt)
        return out
    except Exception as e:
        print(f"[MASTODON:{instance}] error: {e}")
        return []

def fetch_from_mastodon_all(query: str, limit: int = 30) -> List[str]:
    posts = []
    for inst in MASTODON_INSTANCES:
        posts.extend(fetch_from_mastodon_instance(inst, query, limit=limit))
    return posts[:limit]

# =========================================================
# MATCHING + SCORING
# =========================================================

def match_level(text: str, company_name: str, lex: Dict[str, List[str]]) -> str:
    low = text.lower()

    for t in lex["exact_mentions"] + lex["lang_variants"]:
        if t.lower() in low:
            return "strong"

    has_company = company_name.lower() in low
    has_policy = any(pk.lower() in low for pk in lex["policy_terms"])
    has_noise = any(no.lower() in low for no in lex["anti_noise"])

    if has_company and has_policy:
        return "policy"
    if has_company and not has_policy:
        return "very_weak" if has_noise else "weak"
    if has_policy:
        return "weak"
    return "very_weak"

def level_weight(level: str) -> float:
    if level == "strong":
        return WEIGHT_STRONG
    if level == "policy":
        return WEIGHT_POLICY
    if level == "weak":
        return WEIGHT_WEAK
    return WEIGHT_VERY_WEAK

def compute_relevance(level: str, source: str, temporal_mode: str) -> float:
    """
    Score simple pour trier les posts.
    - strong très haut
    - policy haut
    - + bonus si on est dans la fenêtre de la loi
    - + petit bonus si source "structurée"
    """
    base = level_weight(level)
    if level == "strong":
        base += 1.0
    elif level == "policy":
        base += 0.5

    if temporal_mode == "law_date":
        base += 0.3

    if source in ("reddit_submission", "hackernews"):
        base += 0.2

    return base

def hash_text(text: str) -> str:
    return md5(text.encode("utf-8", errors="ignore")).hexdigest()

# =========================================================
# CSV
# =========================================================

def write_social_csv(directive_slug: str, rows: List[Dict[str, Any]]):
    base_dir = os.path.join(OUTPUT_BASE, directive_slug)
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"social_{directive_slug}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[CSV] social -> {path}")

def write_analysis_csv(directive_slug: str, rows: List[Dict[str, Any]]):
    base_dir = os.path.join(OUTPUT_BASE, directive_slug)
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"analysis_{directive_slug}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[CSV] analysis -> {path}")

# =========================================================
# MAIN
# =========================================================

def main():
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] start social multi-source (temporal v3 LLM date + cap) ...")

    directives = list_directives(DIRECTIVES_DIR)
    if not directives:
        print("[!] no directive found")
        return

    directive_path = directives[3]
    directive_text = read_directive_text(directive_path)
    directive_lang = detect_language(directive_text)
    directive_title = guess_title_from_filename(directive_path)
    directive_slug = slugify(directive_title)

    # 1) lexique dynamique
    lexicon = build_dynamic_lexicon(directive_title, directive_text, directive_lang)
    print(f"[DIR] {directive_title} | lang={directive_lang}")
    print(f"[DIR] lexicon = {json.dumps(lexicon, indent=2)}")

    # 2) fenêtre temporelle via LLM DATES
    pushshift_after, pushshift_before, hn_start_ts, hn_end_ts, effective_mode = compute_temporal_window_via_llm(
        directive_title, directive_text, directive_lang
    )
    print(f"[TIME] mode={effective_mode} pushshift_after={pushshift_after} pushshift_before={pushshift_before} hn=({hn_start_ts},{hn_end_ts})")

    # 3) entreprises
    companies = load_companies(SP500_PATH, limit=MAX_COMPANIES)
    print(f"[SP500] loaded {len(companies)} companies")

    social_rows: List[Dict[str, Any]] = []
    analysis_rows: List[Dict[str, Any]] = []
    seen_hashes = set()

    for idx, company in enumerate(companies, start=1):
        name = company["name"]
        ticker = company["ticker"]
        print(f"[{idx}/{len(companies)}] company={name} ({ticker})")

        queries = build_queries_for_company(lexicon, name, ticker)

        # on collecte dans une liste locale
        company_posts: List[Dict[str, Any]] = []
        strategies_success: List[str] = []
        strategies_failed: List[str] = []

        # ============== PASS 1 : FENÊTRE LOI ==============
        for q in queries:
            # early stop souple
            if len(company_posts) >= int(MAX_POSTS_PER_COMPANY * EARLY_STOP_FACTOR):
                print(f"[{name}] early-stop (law_date) at {len(company_posts)} candidates")
                break

            # REDDIT
            if ENABLE_REDDIT:
                subms = reddit_submissions(q, size=POSTS_PER_QUERY,
                                           after=pushshift_after,
                                           before=pushshift_before)
                comms = reddit_comments(q, size=POSTS_PER_QUERY,
                                        after=pushshift_after,
                                        before=pushshift_before)
                if subms or comms:
                    strategies_success.append(f"reddit:{q} (law_date)")
                else:
                    strategies_failed.append(f"reddit:{q} (law_date)")

                for s in subms:
                    full = (s.get("title", "") + " " + s.get("selftext", "")).strip()
                    if not full:
                        continue
                    h = hash_text("REDDIT_SUBM|" + full)
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)

                    level = match_level(full, name, lexicon)
                    sent = detect_targeted_or_doc_sentiment(full, name)
                    label = sent.get("Sentiment", "UNKNOWN").upper()
                    scores = sent.get("SentimentScore", {})
                    rel = compute_relevance(level, "reddit_submission", "law_date")

                    company_posts.append({
                        "directive_slug": directive_slug,
                        "directive_title": directive_title,
                        "company": name,
                        "ticker": ticker,
                        "query": q,
                        "source": "reddit_submission",
                        "text": full,
                        "match_level": level,
                        "sentiment": label,
                        "sentiment_score": scores,
                        "temporal_mode": "law_date",
                        "relevance_score": rel,
                    })

                for c in comms:
                    body = c.get("body", "")
                    if not body or body == "[deleted]":
                        continue
                    h = hash_text("REDDIT_COMM|" + body)
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)

                    level = match_level(body, name, lexicon)
                    sent = detect_targeted_or_doc_sentiment(body, name)
                    label = sent.get("Sentiment", "UNKNOWN").upper()
                    scores = sent.get("SentimentScore", {})
                    rel = compute_relevance(level, "reddit_comment", "law_date")

                    company_posts.append({
                        "directive_slug": directive_slug,
                        "directive_title": directive_title,
                        "company": name,
                        "ticker": ticker,
                        "query": q,
                        "source": "reddit_comment",
                        "text": body,
                        "match_level": level,
                        "sentiment": label,
                        "sentiment_score": scores,
                        "temporal_mode": "law_date",
                        "relevance_score": rel,
                    })

            # HN
            if ENABLE_HN:
                hn_hits = hn_search(q, hits_per_page=POSTS_PER_QUERY)
                filtered_hn = []
                for hhit in hn_hits:
                    if is_hn_in_window(hhit, hn_start_ts, hn_end_ts, DAYS_BACK):
                        filtered_hn.append(hhit)

                if filtered_hn:
                    strategies_success.append(f"hn:{q} (law_date)")
                else:
                    strategies_failed.append(f"hn:{q} (law_date)")

                for hhit in filtered_hn:
                    title = hhit.get("title", "")
                    url = hhit.get("url", "") or ""
                    txt = title
                    h = hash_text("HN|" + txt + url)
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)

                    level = match_level(txt, name, lexicon)
                    sent = detect_targeted_or_doc_sentiment(txt, name)
                    label = sent.get("Sentiment", "UNKNOWN").upper()
                    scores = sent.get("SentimentScore", {})
                    rel = compute_relevance(level, "hackernews", "law_date")

                    company_posts.append({
                        "directive_slug": directive_slug,
                        "directive_title": directive_title,
                        "company": name,
                        "ticker": ticker,
                        "query": q,
                        "source": "hackernews",
                        "text": txt,
                        "match_level": level,
                        "sentiment": label,
                        "sentiment_score": scores,
                        "url": url,
                        "temporal_mode": "law_date",
                        "relevance_score": rel,
                    })

            # Mastodon
            if ENABLE_MASTODON:
                m_posts = fetch_from_mastodon_all(q, limit=POSTS_PER_QUERY)
                if m_posts:
                    strategies_success.append(f"mastodon:{q}")
                else:
                    strategies_failed.append(f"mastodon:{q}")

                for p in m_posts:
                    h = hash_text("MASTODON|" + p)
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)

                    level = match_level(p, name, lexicon)
                    sent = detect_targeted_or_doc_sentiment(p, name)
                    label = sent.get("Sentiment", "UNKNOWN").upper()
                    scores = sent.get("SentimentScore", {})
                    rel = compute_relevance(level, "mastodon", "law_date")

                    company_posts.append({
                        "directive_slug": directive_slug,
                        "directive_title": directive_title,
                        "company": name,
                        "ticker": ticker,
                        "query": q,
                        "source": "mastodon",
                        "text": p,
                        "match_level": level,
                        "sentiment": label,
                        "sentiment_score": scores,
                        "temporal_mode": "law_date",
                        "relevance_score": rel,
                    })

            time.sleep(0.12)

        # ============== PASS 2 : FALLBACK RECENT ==============
        if not company_posts:
            print(f"[{name}] no posts in law_date window -> recent fallback {DAYS_BACK}d")
            for q in queries:
                if len(company_posts) >= int(MAX_POSTS_PER_COMPANY * EARLY_STOP_FACTOR):
                    print(f"[{name}] early-stop (recent) at {len(company_posts)} candidates")
                    break

                if ENABLE_REDDIT:
                    subms = reddit_submissions(q, size=POSTS_PER_QUERY,
                                               after=PUSHSHIFT_RECENT_AFTER,
                                               before=None)
                    comms = reddit_comments(q, size=POSTS_PER_QUERY,
                                            after=PUSHSHIFT_RECENT_AFTER,
                                            before=None)
                    for s in subms:
                        full = (s.get("title", "") + " " + s.get("selftext", "")).strip()
                        if not full:
                            continue
                        h = hash_text("REDDIT_SUBM_RECENT|" + full)
                        if h in seen_hashes:
                            continue
                        seen_hashes.add(h)

                        level = match_level(full, name, lexicon)
                        sent = detect_targeted_or_doc_sentiment(full, name)
                        label = sent.get("Sentiment", "UNKNOWN").upper()
                        scores = sent.get("SentimentScore", {})
                        rel = compute_relevance(level, "reddit_submission", "recent_fallback")

                        company_posts.append({
                            "directive_slug": directive_slug,
                            "directive_title": directive_title,
                            "company": name,
                            "ticker": ticker,
                            "query": q,
                            "source": "reddit_submission",
                            "text": full,
                            "match_level": level,
                            "sentiment": label,
                            "sentiment_score": scores,
                            "temporal_mode": "recent_fallback",
                            "relevance_score": rel,
                        })

                    for c in comms:
                        body = c.get("body", "")
                        if not body or body == "[deleted]":
                            continue
                        h = hash_text("REDDIT_COMM_RECENT|" + body)
                        if h in seen_hashes:
                            continue
                        seen_hashes.add(h)

                        level = match_level(body, name, lexicon)
                        sent = detect_targeted_or_doc_sentiment(body, name)
                        label = sent.get("Sentiment", "UNKNOWN").upper()
                        scores = sent.get("SentimentScore", {})
                        rel = compute_relevance(level, "reddit_comment", "recent_fallback")

                        company_posts.append({
                            "directive_slug": directive_slug,
                            "directive_title": directive_title,
                            "company": name,
                            "ticker": ticker,
                            "query": q,
                            "source": "reddit_comment",
                            "text": body,
                            "match_level": level,
                            "sentiment": label,
                            "sentiment_score": scores,
                            "temporal_mode": "recent_fallback",
                            "relevance_score": rel,
                        })

                if ENABLE_HN:
                    hn_hits = hn_search(q, hits_per_page=POSTS_PER_QUERY)
                    for hhit in hn_hits:
                        if not is_hn_in_window(hhit, None, None, DAYS_BACK):
                            continue
                        title = hhit.get("title", "")
                        url = hhit.get("url", "") or ""
                        txt = title
                        h = hash_text("HN_RECENT|" + txt + url)
                        if h in seen_hashes:
                            continue
                        seen_hashes.add(h)

                        level = match_level(txt, name, lexicon)
                        sent = detect_targeted_or_doc_sentiment(txt, name)
                        label = sent.get("Sentiment", "UNKNOWN").upper()
                        scores = sent.get("SentimentScore", {})
                        rel = compute_relevance(level, "hackernews", "recent_fallback")

                        company_posts.append({
                            "directive_slug": directive_slug,
                            "directive_title": directive_title,
                            "company": name,
                            "ticker": ticker,
                            "query": q,
                            "source": "hackernews",
                            "text": txt,
                            "match_level": level,
                            "sentiment": label,
                            "sentiment_score": scores,
                            "url": url,
                            "temporal_mode": "recent_fallback",
                            "relevance_score": rel,
                        })

                time.sleep(0.12)

        # ======== SEULEMENT MAINTENANT : ON TRIE & ON COUPE ========
        # tri décroissant par pertinence
        company_posts_sorted = sorted(company_posts, key=lambda x: x["relevance_score"], reverse=True)
        company_posts_kept = company_posts_sorted[:MAX_POSTS_PER_COMPANY]

        # recalcul agrégats sur ce qu'on garde
        nb_posts_found = len(company_posts_kept)
        counts_by_level = {"strong": 0, "policy": 0, "weak": 0, "very_weak": 0}
        agg_pos = agg_neg = agg_neu = agg_mix = 0.0
        total_weight = 0.0

        for p in company_posts_kept:
            lvl = p["match_level"]
            counts_by_level[lvl] += 1
            scores = p["sentiment_score"] or {}
            w = level_weight(lvl)
            agg_pos += float(scores.get("Positive", 0.0)) * w
            agg_neg += float(scores.get("Negative", 0.0)) * w
            agg_neu += float(scores.get("Neutral", 0.0)) * w
            agg_mix += float(scores.get("Mixed", 0.0)) * w
            total_weight += w

        if total_weight > 0:
            avg_pos = agg_pos / total_weight
            avg_neg = agg_neg / total_weight
            avg_neu = agg_neu / total_weight
            avg_mix = agg_mix / total_weight
        else:
            avg_pos = avg_neg = avg_mix = 0.0
            avg_neu = 1.0

        # on pousse les posts gardés dans le CSV global
        for p in company_posts_kept:
            # on convertit le dict sentiment_score en json string pour csv
            p_out = dict(p)
            p_out["sentiment_score"] = json.dumps(p["sentiment_score"])
            social_rows.append(p_out)

        analysis_rows.append({
            "directive_slug": directive_slug,
            "directive_title": directive_title,
            "company": name,
            "ticker": ticker,
            "nb_posts_found": nb_posts_found,
            "nb_posts_strong": counts_by_level["strong"],
            "nb_posts_policy": counts_by_level["policy"],
            "nb_posts_weak": counts_by_level["weak"],
            "nb_posts_very_weak": counts_by_level["very_weak"],
            "avg_pos": round(avg_pos, 4),
            "avg_neg": round(avg_neg, 4),
            "avg_neu": round(avg_neu, 4),
            "avg_mix": round(avg_mix, 4),
            "strategies_success": " | ".join(strategies_success),
            "strategies_failed": " | ".join(strategies_failed),
            "temporal_mode": effective_mode,
        })

    write_social_csv(directive_slug, social_rows)
    write_analysis_csv(directive_slug, analysis_rows)
    print("[OK] done (temporal v3 LLM date + cap).")


if __name__ == "__main__":
    main()