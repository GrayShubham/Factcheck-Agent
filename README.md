# 🔎 Fact-Check Agent

A **"Truth Layer" for marketing content.** Upload a PDF and the app extracts the
factual claims (stats, dates, financial/technical figures), cross-references each
against **live web data**, and flags them as **Verified**, **Inaccurate**, or
**False** — along with the correct "real" fact and supporting sources.

Built for the scenario where marketing decks contain outdated or hallucinated
statistics. Drop in a "trap document" full of intentional lies and the app will
catch them.

**Live app:** _add your Streamlit Cloud URL here after deploying_

---

## How it works

```
PDF → text extraction (pypdf)
    → claim extraction      (Gemini 2.5 Flash, JSON output)
    → claim verification    (Gemini 2.5 Flash + Google Search grounding)
    → verdict + real fact + sources
```

1. **Extract** — `pypdf` pulls raw text from the uploaded PDF.
2. **Identify claims** — Gemini isolates the specific, checkable factual claims.
3. **Verify** — each claim is checked against the live web via Gemini's Google
   Search grounding tool.
4. **Report** — claims are flagged with a verdict, a **confidence score**, the
   correct fact, a one-line explanation, and source links. A summary table and
   metric counts sit on top; flagged items are surfaced first.

Claims are verified **in parallel** (thread pool) so larger PDFs stay fast.

## Verdicts

| Verdict | Meaning |
|---|---|
| ✅ **Verified** | Matches reliable current data. |
| ⚠️ **Inaccurate** | Partially true but wrong or outdated. |
| ❌ **False** | No credible evidence, or evidence contradicts it. |

## Tech stack

- **Frontend / app:** Streamlit
- **LLM + web search:** Google Gemini 2.5 Flash with Google Search grounding (one API, free tier)
- **PDF parsing:** pypdf

## Run locally

```bash
git clone <your-repo-url>
cd factcheck-app
pip install -r requirements.txt

# provide your Gemini API key (free: https://aistudio.google.com/apikey)
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml and paste your key

streamlit run app.py
```

You can also paste the key directly in the app sidebar instead of using secrets.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select the repo, branch `main`, main file `app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```
   GEMINI_API_KEY = "your-key"
   ```
5. Deploy. You get a public `https://<name>.streamlit.app` URL.

## Project structure

```
factcheck-app/
├── app.py                          # Streamlit app (extract → verify → report)
├── requirements.txt                # dependencies
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets.toml.example        # template for your API key
```

## Notes

- Checks up to **15 claims** per document to stay within free-tier API limits;
  adjust `MAX_CLAIMS` (and `MAX_WORKERS` for parallelism) in `app.py` if needed.
- Works best on text PDFs. Scanned/image-only PDFs would need an OCR step
  (e.g. Tesseract / Google Vision) — a planned enhancement, not in this build.

## Roadmap

- OCR fallback for scanned PDFs (Tesseract / Cloud Vision).
- Per-claim caching to cut repeat API calls.
- Export report as PDF/CSV in addition to JSON.
