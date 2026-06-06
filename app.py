"""
Fact-Check Agent — a "Truth Layer" for marketing content.

Upload a PDF; the app extracts factual claims (stats, dates, financial/technical
figures), verifies each against live web data using Gemini with Google Search
grounding, and flags them as VERIFIED, INACCURATE, or FALSE — with the correct
"real" fact and sources.
"""

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types

# ----------------------------------------------------------------------------- #
# Config
# ----------------------------------------------------------------------------- #
MODEL = "gemini-2.5-flash-lite"  # higher free-tier RPM than full flash
MAX_CLAIMS = 10   # cap to protect free-tier API credits
MAX_WORKERS = 3   # parallel claim verification (gated by the rate limiter below)
RPM = 14          # keep total requests/min under the free-tier quota
MAX_RETRIES = 5   # retry on 429 RESOURCE_EXHAUSTED with backoff

# global pacer: serializes + spaces calls across all worker threads so we never
# exceed the free-tier per-minute quota (the cause of "Unverified" 429 errors).
_pace_lock = threading.Lock()
_last_call = [0.0]


def paced_generate(client, contents, config):
    """Rate-limited generate_content with retry/backoff on 429."""
    interval = 60.0 / RPM
    for attempt in range(MAX_RETRIES):
        with _pace_lock:
            wait = _last_call[0] + interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _last_call[0] = time.monotonic()
        try:
            return client.models.generate_content(
                model=MODEL, contents=contents, config=config
            )
        except Exception as e:
            transient = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if transient and attempt < MAX_RETRIES - 1:
                time.sleep(min(20, 2 ** attempt + random.random()))
                continue
            raise

VERDICT_STYLE = {
    "VERIFIED":   {"emoji": "✅", "label": "Verified",   "color": "#16A34A", "bg": "#EAF7EF", "soft": "#D1F0DD"},
    "INACCURATE": {"emoji": "⚠️", "label": "Inaccurate", "color": "#D97706", "bg": "#FEF6E7", "soft": "#FBE6BE"},
    "FALSE":      {"emoji": "❌", "label": "False",      "color": "#DC2626", "bg": "#FDECEC", "soft": "#F8CFCF"},
    "UNVERIFIED": {"emoji": "❔", "label": "Unverified", "color": "#64748B", "bg": "#F1F4F9", "soft": "#DDE3EC"},
}

st.set_page_config(page_title="Fact-Check Agent", page_icon="🔎", layout="wide")

# ----------------------------------------------------------------------------- #
# Theme  (palette mirrors the GEO Citation Copilot deck)
# ----------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

.stApp { background: #F5F7FF; }
html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
.block-container { padding-top: 1.4rem; max-width: 1080px; }
#MainMenu, footer, header { visibility: hidden; }

/* hero */
.hero {
  background: linear-gradient(135deg, #0B1437 0%, #1B225E 55%, #2A2F7A 100%);
  border-radius: 20px; padding: 34px 38px; color: #fff;
  position: relative; overflow: hidden; box-shadow: 0 18px 40px rgba(11,20,55,.28);
}
.hero::after {
  content:""; position:absolute; right:-60px; top:-60px; width:240px; height:240px;
  background: radial-gradient(circle, rgba(124,92,252,.55), transparent 70%); border-radius:50%;
}
.hero::before {
  content:""; position:absolute; right:60px; bottom:-90px; width:200px; height:200px;
  background: radial-gradient(circle, rgba(34,211,238,.40), transparent 70%); border-radius:50%;
}
.hero .badge {
  display:inline-block; background: rgba(34,211,238,.16); color:#67E8F9;
  font-weight:700; font-size:.72rem; letter-spacing:.18em; padding:6px 12px;
  border-radius:999px; margin-bottom:14px;
}
.hero h1 { font-size:2.5rem; font-weight:800; margin:0 0 8px; line-height:1.05; }
.hero p { color:#C9D2F7; font-size:1.02rem; max-width:680px; margin:0; }

/* stat chips */
.stat-row { display:flex; gap:14px; margin-top:22px; flex-wrap:wrap; }
.stat { background: rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.12);
        border-radius:14px; padding:14px 16px; flex:1; min-width:170px; }
.stat b { display:block; font-size:1.55rem; color:#fff; font-weight:800; }
.stat span { color:#AEB9E8; font-size:.78rem; line-height:1.3; display:block; margin-top:2px; }

/* how-it-works step cards */
.steps { display:flex; gap:16px; margin:6px 0 4px; flex-wrap:wrap; }
.step { background:#fff; border:1px solid #E4E8F4; border-radius:16px; padding:20px;
        flex:1; min-width:200px; box-shadow:0 6px 18px rgba(11,20,55,.05); }
.step .n { width:34px; height:34px; border-radius:10px; display:flex; align-items:center;
           justify-content:center; font-weight:800; color:#fff; margin-bottom:12px; }
.step h4 { margin:0 0 4px; font-size:1.05rem; color:#0E1330; font-weight:700; }
.step p { margin:0; color:#5B6478; font-size:.88rem; line-height:1.4; }

/* trust score */
.dash { display:flex; gap:22px; align-items:center; background:#fff; border:1px solid #E4E8F4;
        border-radius:18px; padding:24px 26px; box-shadow:0 8px 22px rgba(11,20,55,.06); }
.donut { width:150px; height:150px; border-radius:50%; flex-shrink:0; position:relative; }
.donut .hole { position:absolute; inset:18px; background:#fff; border-radius:50%;
               display:flex; flex-direction:column; align-items:center; justify-content:center; }
.donut .hole .big { font-size:2.1rem; font-weight:800; color:#0E1330; line-height:1; }
.donut .hole .lbl { font-size:.66rem; color:#64748B; letter-spacing:.08em; font-weight:700; margin-top:4px;}
.legend { flex:1; }
.legend h3 { margin:0 0 10px; color:#0E1330; font-size:1.15rem; font-weight:800; }
.bar { display:flex; height:14px; border-radius:8px; overflow:hidden; margin:10px 0 14px; }
.legend-items { display:flex; gap:18px; flex-wrap:wrap; }
.li { display:flex; align-items:center; gap:7px; font-size:.86rem; color:#334155; font-weight:600; }
.dot { width:11px; height:11px; border-radius:3px; }

/* verdict card */
.vcard { background:#fff; border:1px solid #E4E8F4; border-left-width:6px; border-radius:14px;
         padding:16px 20px; margin-bottom:14px; box-shadow:0 4px 14px rgba(11,20,55,.05); }
.vhead { display:flex; align-items:center; justify-content:space-between; margin-bottom:10px; }
.vtag { font-weight:800; font-size:.8rem; letter-spacing:.04em; padding:4px 12px; border-radius:999px; }
.conf-wrap { display:flex; align-items:center; gap:8px; min-width:160px; }
.conf-track { flex:1; height:7px; background:#EDF0F7; border-radius:99px; overflow:hidden; }
.conf-fill { height:100%; border-radius:99px; }
.conf-num { font-size:.74rem; font-weight:700; color:#475467; white-space:nowrap; }
.vclaim { color:#0E1330; font-size:1rem; font-weight:600; margin-bottom:8px; }
.vfact { background:#F7F9FE; border-radius:10px; padding:10px 12px; color:#0E1330; font-size:.93rem; margin-bottom:6px; }
.vexp { color:#64748B; font-size:.84rem; }

div.stButton > button { background:#7C5CFC; color:#fff; border:none; border-radius:10px;
        font-weight:700; padding:.55rem 1.4rem; box-shadow:0 6px 16px rgba(124,92,252,.32); }
div.stButton > button:hover { background:#6B49F0; color:#fff; }
</style>
"""


# ----------------------------------------------------------------------------- #
# API key
# ----------------------------------------------------------------------------- #
def get_api_key() -> str | None:
    # Priority: Streamlit secrets (deployment) -> env var -> sidebar input
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass
    key = key or os.environ.get("GEMINI_API_KEY")
    if not key:
        key = st.session_state.get("api_key_input")
    return key


# ----------------------------------------------------------------------------- #
# PDF -> text
# ----------------------------------------------------------------------------- #
def extract_pdf_text(file) -> str:
    reader = PdfReader(file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


# ----------------------------------------------------------------------------- #
# Step 1 — extract claims
# ----------------------------------------------------------------------------- #
EXTRACT_PROMPT = """You are a fact-checking analyst. From the document text below,
extract the specific, checkable factual claims — statistics, dates, monetary
amounts, percentages, rankings, named records, and technical figures.

Rules:
- Only claims that can be objectively verified against external sources.
- Ignore opinions, slogans, and vague marketing language.
- Quote each claim concisely as a standalone sentence.
- Return at most {max_claims} of the most significant claims.

Return ONLY valid JSON in this exact shape:
{{"claims": ["claim 1", "claim 2", ...]}}

DOCUMENT TEXT:
\"\"\"
{doc}
\"\"\"
"""


def extract_claims(client: genai.Client, doc_text: str) -> list[str]:
    prompt = EXTRACT_PROMPT.format(max_claims=MAX_CLAIMS, doc=doc_text[:20000])
    resp = paced_generate(
        client,
        prompt,
        types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
        ),
    )
    try:
        data = json.loads(resp.text)
        claims = data.get("claims", [])
        return [c.strip() for c in claims if c and c.strip()][:MAX_CLAIMS]
    except (json.JSONDecodeError, AttributeError):
        return []


# ----------------------------------------------------------------------------- #
# Step 2 — verify a claim with Google Search grounding
# ----------------------------------------------------------------------------- #
VERIFY_PROMPT = """You are a rigorous fact-checker with live web access.

Verify this claim against current, authoritative web sources:
CLAIM: "{claim}"

Decide one verdict:
- VERIFIED: the claim matches reliable current data.
- INACCURATE: partially true but wrong/outdated (e.g., old stat, off number).
- FALSE: no credible evidence supports it, or evidence contradicts it.

Respond in EXACTLY this format, nothing else:
VERDICT: <VERIFIED|INACCURATE|FALSE>
CONFIDENCE: <integer 0-100, how sure you are given the sources>
CORRECT_FACT: <the accurate fact in one sentence; if VERIFIED, restate the confirmed fact>
EXPLANATION: <one sentence on why>
"""


def verify_claim(client: genai.Client, claim: str) -> dict:
    resp = paced_generate(
        client,
        VERIFY_PROMPT.format(claim=claim),
        types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0,
        ),
    )
    text = (resp.text or "").strip()

    verdict, correct_fact, explanation, confidence = "UNVERIFIED", "", "", None
    for line in text.splitlines():
        up = line.strip()
        if up.upper().startswith("VERDICT:"):
            v = up.split(":", 1)[1].strip().upper()
            verdict = v if v in VERDICT_STYLE else "UNVERIFIED"
        elif up.upper().startswith("CONFIDENCE:"):
            digits = "".join(ch for ch in up.split(":", 1)[1] if ch.isdigit())
            if digits:
                confidence = max(0, min(100, int(digits)))
        elif up.upper().startswith("CORRECT_FACT:"):
            correct_fact = up.split(":", 1)[1].strip()
        elif up.upper().startswith("EXPLANATION:"):
            explanation = up.split(":", 1)[1].strip()

    # pull grounding source URLs
    sources = []
    try:
        gm = resp.candidates[0].grounding_metadata
        for chunk in (gm.grounding_chunks or []):
            if chunk.web and chunk.web.uri:
                sources.append({"title": chunk.web.title or chunk.web.uri,
                                "uri": chunk.web.uri})
    except (AttributeError, IndexError, TypeError):
        pass

    return {
        "claim": claim,
        "verdict": verdict,
        "confidence": confidence,
        "correct_fact": correct_fact,
        "explanation": explanation,
        "sources": sources[:3],
    }


# ----------------------------------------------------------------------------- #
# UI components
# ----------------------------------------------------------------------------- #
def hero():
    st.markdown(
        """
        <div class="hero">
          <span class="badge">🔎 THE TRUTH LAYER</span>
          <h1>Fact-Check Agent</h1>
          <p>Marketing decks are full of outdated and hallucinated stats. Upload a PDF —
          this tool extracts every claim, verifies it against the live web, and flags
          what's wrong with the real number and its source.</p>
          <div class="stat-row">
            <div class="stat"><b>43%</b><span>of Google searches now end with zero clicks — AI answers decide what's "true"</span></div>
            <div class="stat"><b>14.2%</b><span>conversion on AI-search traffic vs 2.8% — accuracy in answers is high-stakes</span></div>
            <div class="stat"><b>&lt;60s</b><span>to audit a document's claims against authoritative sources</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def how_it_works():
    st.markdown(
        """
        <div class="steps">
          <div class="step"><div class="n" style="background:#7C5CFC">1</div>
            <h4>Extract</h4><p>Pulls specific claims — stats, dates, money, technical figures — from the PDF.</p></div>
          <div class="step"><div class="n" style="background:#22B6D6">2</div>
            <h4>Verify</h4><p>Checks each claim against live web data via Gemini + Google Search grounding.</p></div>
          <div class="step"><div class="n" style="background:#16A34A">3</div>
            <h4>Report</h4><p>Flags Verified / Inaccurate / False with a confidence score and the real fact.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def trust_dashboard(results: list[dict], counts: dict):
    total = len(results) or 1
    checked = counts["VERIFIED"] + counts["INACCURATE"] + counts["FALSE"]
    score = round(100 * counts["VERIFIED"] / checked) if checked else 0

    # donut degrees (verified / inaccurate / false / unverified)
    segs = [("VERIFIED", counts["VERIFIED"]), ("INACCURATE", counts["INACCURATE"]),
            ("FALSE", counts["FALSE"]), ("UNVERIFIED", counts["UNVERIFIED"])]
    stops, acc = [], 0.0
    for k, n in segs:
        if n == 0:
            continue
        start = acc / total * 360
        acc += n
        end = acc / total * 360
        stops.append(f"{VERDICT_STYLE[k]['color']} {start:.1f}deg {end:.1f}deg")
    gradient = ", ".join(stops) if stops else "#DDE3EC 0deg 360deg"

    # stacked bar segments
    bar = "".join(
        f'<div style="width:{n/total*100:.1f}%;background:{VERDICT_STYLE[k]["color"]}"></div>'
        for k, n in segs if n > 0
    )
    legend = "".join(
        f'<div class="li"><span class="dot" style="background:{VERDICT_STYLE[k]["color"]}"></span>'
        f'{VERDICT_STYLE[k]["label"]} · {n}</div>'
        for k, n in segs if n > 0
    )
    flagged = counts["INACCURATE"] + counts["FALSE"]
    head = (f"{flagged} issue{'s' if flagged != 1 else ''} flagged across {len(results)} claims"
            if flagged else f"All {len(results)} claims checked")

    st.markdown(
        f"""
        <div class="dash">
          <div class="donut" style="background: conic-gradient({gradient});">
            <div class="hole"><div class="big">{score}%</div><div class="lbl">TRUST SCORE</div></div>
          </div>
          <div class="legend">
            <h3>{head}</h3>
            <div class="bar">{bar}</div>
            <div class="legend-items">{legend}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(r: dict):
    s = VERDICT_STYLE[r["verdict"]]
    conf = r.get("confidence")
    if conf is not None:
        conf_html = (
            f'<div class="conf-wrap"><div class="conf-track">'
            f'<div class="conf-fill" style="width:{conf}%;background:{s["color"]}"></div></div>'
            f'<span class="conf-num">{conf}%</span></div>'
        )
    else:
        conf_html = ""

    fact = r["correct_fact"] or "—"
    st.markdown(
        f"""
        <div class="vcard" style="border-left-color:{s['color']}">
          <div class="vhead">
            <span class="vtag" style="background:{s['bg']};color:{s['color']}">{s['emoji']} {s['label'].upper()}</span>
            {conf_html}
          </div>
          <div class="vclaim">{r['claim']}</div>
          <div class="vfact"><b>Real fact:</b> {fact}</div>
          <div class="vexp">{r['explanation']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if r["sources"]:
        with st.expander("🔗 Sources"):
            for src in r["sources"]:
                st.markdown(f"- [{src['title']}]({src['uri']})")


# ----------------------------------------------------------------------------- #
# App
# ----------------------------------------------------------------------------- #
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    hero()
    st.write("")

    with st.sidebar:
        st.markdown("### ⚙️ Setup")
        if not (st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None) \
                and not os.environ.get("GEMINI_API_KEY"):
            st.text_input(
                "Gemini API key",
                type="password",
                key="api_key_input",
                help="Free key at aistudio.google.com/apikey",
            )
        st.markdown(
            f"""
            **Engine**
            Gemini 2.5 Flash-Lite + Google Search grounding.

            **Limits**
            Up to **{MAX_CLAIMS}** claims/run · throttled to **{RPM}** req/min to
            respect the free tier.

            **Verdicts**
            ✅ Verified  ⚠️ Inaccurate  ❌ False  ❔ Unverified
            """
        )

    api_key = get_api_key()
    if not api_key:
        how_it_works()
        st.info("👈 Add your Gemini API key in the sidebar to begin.")
        st.stop()

    client = genai.Client(api_key=api_key)

    how_it_works()
    uploaded = st.file_uploader("Upload a PDF to fact-check", type=["pdf"])
    if not uploaded:
        return

    with st.spinner("Reading PDF…"):
        doc_text = extract_pdf_text(uploaded)
    if not doc_text:
        st.error("Could not extract text from this PDF (it may be scanned/image-only).")
        return

    if not st.button("🚀 Run fact-check", type="primary"):
        return

    with st.spinner("Extracting claims…"):
        claims = extract_claims(client, doc_text)

    if not claims:
        st.warning("No checkable factual claims found in this document.")
        return

    progress = st.progress(0.0, text=f"Verifying {len(claims)} claims against live web…")
    results, done = [], 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(verify_claim, client, c): c for c in claims}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:  # keep going if one claim fails
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    note = "Gemini free-tier rate limit reached — wait a minute and re-run."
                else:
                    note = f"Verification error: {msg[:160]}"
                results.append({"claim": futures[fut], "verdict": "UNVERIFIED",
                                "confidence": None, "correct_fact": "",
                                "explanation": note, "sources": []})
            done += 1
            progress.progress(done / len(claims),
                              text=f"Verified {done}/{len(claims)} claims…")
    progress.empty()

    counts = {k: sum(1 for r in results if r["verdict"] == k) for k in VERDICT_STYLE}

    trust_dashboard(results, counts)
    st.write("")

    # show flagged first
    order = {"FALSE": 0, "INACCURATE": 1, "UNVERIFIED": 2, "VERIFIED": 3}
    for r in sorted(results, key=lambda x: order.get(x["verdict"], 9)):
        render_result(r)

    st.download_button(
        "⬇️ Download report (JSON)",
        data=json.dumps(results, indent=2),
        file_name="factcheck_report.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()

def paced_generate(client, contents, config):
    """Rate-limited generate_content with retry/backoff on 429."""
    interval = 60.0 / RPM
    for attempt in range(MAX_RETRIES):
        with _pace_lock:
            wait = _last_call[0] + interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _last_call[0] = time.monotonic()
        try:
            return client.models.generate_content(
                model=MODEL, contents=contents, config=config
            )
        except Exception as e:
            transient = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if transient and attempt < MAX_RETRIES - 1:
                time.sleep(min(20, 2 ** attempt + random.random()))
                continue
            raise

VERDICT_STYLE = {
    "VERIFIED":   {"emoji": "✅", "color": "#1a7f37", "bg": "#e6f4ea"},
    "INACCURATE": {"emoji": "⚠️", "color": "#b54708", "bg": "#fdf2e3"},
    "FALSE":      {"emoji": "❌", "color": "#b42318", "bg": "#fce8e6"},
    "UNVERIFIED": {"emoji": "❔", "color": "#475467", "bg": "#f2f4f7"},
}

st.set_page_config(page_title="Fact-Check Agent", page_icon="🔎", layout="wide")


# ----------------------------------------------------------------------------- #
# API key
# ----------------------------------------------------------------------------- #
def get_api_key() -> str | None:
    # Priority: Streamlit secrets (deployment) -> env var -> sidebar input
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass
    key = key or os.environ.get("GEMINI_API_KEY")
    if not key:
        key = st.session_state.get("api_key_input")
    return key


# ----------------------------------------------------------------------------- #
# PDF -> text
# ----------------------------------------------------------------------------- #
def extract_pdf_text(file) -> str:
    reader = PdfReader(file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


# ----------------------------------------------------------------------------- #
# Step 1 — extract claims
# ----------------------------------------------------------------------------- #
EXTRACT_PROMPT = """You are a fact-checking analyst. From the document text below,
extract the specific, checkable factual claims — statistics, dates, monetary
amounts, percentages, rankings, named records, and technical figures.

Rules:
- Only claims that can be objectively verified against external sources.
- Ignore opinions, slogans, and vague marketing language.
- Quote each claim concisely as a standalone sentence.
- Return at most {max_claims} of the most significant claims.

Return ONLY valid JSON in this exact shape:
{{"claims": ["claim 1", "claim 2", ...]}}

DOCUMENT TEXT:
\"\"\"
{doc}
\"\"\"
"""


def extract_claims(client: genai.Client, doc_text: str) -> list[str]:
    prompt = EXTRACT_PROMPT.format(max_claims=MAX_CLAIMS, doc=doc_text[:20000])
    resp = paced_generate(
        client,
        prompt,
        types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
        ),
    )
    try:
        data = json.loads(resp.text)
        claims = data.get("claims", [])
        return [c.strip() for c in claims if c and c.strip()][:MAX_CLAIMS]
    except (json.JSONDecodeError, AttributeError):
        return []


# ----------------------------------------------------------------------------- #
# Step 2 — verify a claim with Google Search grounding
# ----------------------------------------------------------------------------- #
VERIFY_PROMPT = """You are a rigorous fact-checker with live web access.

Verify this claim against current, authoritative web sources:
CLAIM: "{claim}"

Decide one verdict:
- VERIFIED: the claim matches reliable current data.
- INACCURATE: partially true but wrong/outdated (e.g., old stat, off number).
- FALSE: no credible evidence supports it, or evidence contradicts it.

Respond in EXACTLY this format, nothing else:
VERDICT: <VERIFIED|INACCURATE|FALSE>
CONFIDENCE: <integer 0-100, how sure you are given the sources>
CORRECT_FACT: <the accurate fact in one sentence; if VERIFIED, restate the confirmed fact>
EXPLANATION: <one sentence on why>
"""


def verify_claim(client: genai.Client, claim: str) -> dict:
    resp = paced_generate(
        client,
        VERIFY_PROMPT.format(claim=claim),
        types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0,
        ),
    )
    text = (resp.text or "").strip()

    verdict, correct_fact, explanation, confidence = "UNVERIFIED", "", "", None
    for line in text.splitlines():
        up = line.strip()
        if up.upper().startswith("VERDICT:"):
            v = up.split(":", 1)[1].strip().upper()
            verdict = v if v in VERDICT_STYLE else "UNVERIFIED"
        elif up.upper().startswith("CONFIDENCE:"):
            digits = "".join(ch for ch in up.split(":", 1)[1] if ch.isdigit())
            if digits:
                confidence = max(0, min(100, int(digits)))
        elif up.upper().startswith("CORRECT_FACT:"):
            correct_fact = up.split(":", 1)[1].strip()
        elif up.upper().startswith("EXPLANATION:"):
            explanation = up.split(":", 1)[1].strip()

    # pull grounding source URLs
    sources = []
    try:
        gm = resp.candidates[0].grounding_metadata
        for chunk in (gm.grounding_chunks or []):
            if chunk.web and chunk.web.uri:
                sources.append({"title": chunk.web.title or chunk.web.uri,
                                "uri": chunk.web.uri})
    except (AttributeError, IndexError, TypeError):
        pass

    return {
        "claim": claim,
        "verdict": verdict,
        "confidence": confidence,
        "correct_fact": correct_fact,
        "explanation": explanation,
        "sources": sources[:3],
    }


# ----------------------------------------------------------------------------- #
# UI
# ----------------------------------------------------------------------------- #
def render_result(r: dict):
    style = VERDICT_STYLE[r["verdict"]]
    conf = f" · {r['confidence']}% confidence" if r.get("confidence") is not None else ""
    st.markdown(
        f"""
        <div style="border-left:6px solid {style['color']};
                    background:{style['bg']};
                    padding:14px 18px;border-radius:8px;margin-bottom:14px;">
            <div style="font-weight:600;color:{style['color']};
                        font-size:0.95rem;margin-bottom:6px;">
                {style['emoji']} {r['verdict']}{conf}
            </div>
            <div style="color:#101828;margin-bottom:8px;">
                <b>Claim:</b> {r['claim']}
            </div>
            <div style="color:#101828;margin-bottom:4px;">
                <b>Real fact:</b> {r['correct_fact'] or '—'}
            </div>
            <div style="color:#475467;font-size:0.88rem;">
                {r['explanation']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if r["sources"]:
        with st.expander("Sources"):
            for s in r["sources"]:
                st.markdown(f"- [{s['title']}]({s['uri']})")


def main():
    st.title("🔎 Fact-Check Agent")
    st.caption(
        "A Truth Layer for marketing content. Upload a PDF — the app extracts "
        "claims, checks them against live web data, and flags inaccuracies."
    )

    with st.sidebar:
        st.header("Setup")
        if not (st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None) \
                and not os.environ.get("GEMINI_API_KEY"):
            st.text_input(
                "Gemini API key",
                type="password",
                key="api_key_input",
                help="Get a free key at aistudio.google.com/apikey",
            )
        st.markdown(
            "Powered by **Gemini 2.5 Flash** + Google Search grounding.\n\n"
            f"Checks up to **{MAX_CLAIMS}** claims per document."
        )

    api_key = get_api_key()
    if not api_key:
        st.info("Add your Gemini API key in the sidebar to begin.")
        st.stop()

    client = genai.Client(api_key=api_key)

    uploaded = st.file_uploader("Upload a PDF to fact-check", type=["pdf"])
    if not uploaded:
        return

    with st.spinner("Reading PDF…"):
        doc_text = extract_pdf_text(uploaded)
    if not doc_text:
        st.error("Could not extract text from this PDF (it may be scanned/image-only).")
        return

    if not st.button("Run fact-check", type="primary"):
        return

    with st.spinner("Extracting claims…"):
        claims = extract_claims(client, doc_text)

    if not claims:
        st.warning("No checkable factual claims found in this document.")
        return

    st.subheader(f"Found {len(claims)} claims — verifying against live web…")
    progress = st.progress(0.0)
    results, done = [], 0
    # verify claims in parallel for speed on larger PDFs
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(verify_claim, client, c): c for c in claims}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:  # keep going if one claim fails
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    note = "Gemini free-tier rate limit reached — wait a minute and re-run."
                else:
                    note = f"Verification error: {msg[:160]}"
                results.append({"claim": futures[fut], "verdict": "UNVERIFIED",
                                "confidence": None, "correct_fact": "",
                                "explanation": note, "sources": []})
            done += 1
            progress.progress(done / len(claims))
    progress.empty()

    # summary metrics
    counts = {k: sum(1 for r in results if r["verdict"] == k) for k in VERDICT_STYLE}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✅ Verified", counts["VERIFIED"])
    c2.metric("⚠️ Inaccurate", counts["INACCURATE"])
    c3.metric("❌ False", counts["FALSE"])
    c4.metric("❔ Unverified", counts["UNVERIFIED"])

    # summary table
    summary_df = pd.DataFrame(
        [{"Verdict": f"{VERDICT_STYLE[k]['emoji']} {k.title()}", "Count": counts[k]}
         for k in VERDICT_STYLE if counts[k] > 0]
    )
    st.table(summary_df.set_index("Verdict"))
    st.divider()

    # show flagged first
    order = {"FALSE": 0, "INACCURATE": 1, "UNVERIFIED": 2, "VERIFIED": 3}
    for r in sorted(results, key=lambda x: order.get(x["verdict"], 9)):
        render_result(r)

    st.download_button(
        "Download report (JSON)",
        data=json.dumps(results, indent=2),
        file_name="factcheck_report.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
