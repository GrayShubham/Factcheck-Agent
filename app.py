"""
Fact-Check Agent — a "Truth Layer" for marketing content.

Upload a PDF; the app extracts factual claims (stats, dates, financial/technical
figures), verifies each against live web data using Gemini with Google Search
grounding, and flags them as VERIFIED, INACCURATE, or FALSE — with the correct
"real" fact and sources.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types

# ----------------------------------------------------------------------------- #
# Config
# ----------------------------------------------------------------------------- #
MODEL = "gemini-2.5-flash"
MAX_CLAIMS = 15   # cap to protect free-tier API credits
MAX_WORKERS = 4  # parallel claim verification

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
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
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
    resp = client.models.generate_content(
        model=MODEL,
        contents=VERIFY_PROMPT.format(claim=claim),
        config=types.GenerateContentConfig(
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
                results.append({"claim": futures[fut], "verdict": "UNVERIFIED",
                                "confidence": None, "correct_fact": "",
                                "explanation": f"Verification error: {e}", "sources": []})
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
