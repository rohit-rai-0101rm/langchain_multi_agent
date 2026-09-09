# Multi-Agent Research Assistant

A small LangChain / LangGraph pipeline where four specialised agents cooperate to
produce a written research report on any topic:

| # | Agent / Chain | Job | Tool |
|---|---------------|-----|------|
| 1 | **Search Agent** | Finds recent, reliable sources for the topic | `web_search` (Tavily) |
| 2 | **Reader Agent** | Picks the best URL and scrapes its main content | `scrape_url` (trafilatura → readability → BeautifulSoup) |
| 3 | **Writer Chain** | Drafts a structured report from the gathered research | – |
| 4 | **Critic Chain** | Scores the report and lists strengths / weaknesses | – |

Steps 1–2 are tool-calling agents built with `langchain.agents.create_agent`.
Steps 3–4 are plain LCEL chains (`prompt | llm | StrOutputParser`).

---

## Architecture

```
topic
  │
  ▼
┌─────────────┐   web_search    ┌──────────────┐
│ Search Agent│ ──────────────▶ │  Tavily API  │
└─────┬───────┘                 └──────────────┘
      │ search_results
      ▼
┌─────────────┐   scrape_url    ┌──────────────┐
│ Reader Agent│ ──────────────▶ │  target page │
└─────┬───────┘                 └──────────────┘
      │ scraped_content
      ▼
┌─────────────┐
│ Writer Chain│ ──▶ report
└─────┬───────┘
      ▼
┌─────────────┐
│ Critic Chain│ ──▶ feedback
└─────────────┘
```

The LLM for every step is a single `ChatOpenAI` instance pointed at Groq's
OpenAI-compatible endpoint (`qwen/qwen3.8-27b`).

---

## Project structure

```
main.py                 CLI entry point (runs the pipeline once for a hard-coded topic)
app.py                  Streamlit UI
src/
  agents/agents.py      llm config, agent builders, writer/critic chains + prompts
  tools/tools.py        web_search (Tavily) and scrape_url tools
  pipeline/pipeline.py  run_research_pipeline(topic) -> dict  (the 4 steps, orchestrated)
  utils.py              approx_tokens() + invoke_with_retry() rate-limit helper
requirements.txt
```

`run_research_pipeline(topic)` returns a dict with keys:
`search_results`, `scraped_content`, `report`, `feedback`.

---

## Setup

Requires **Python 3.12+** (developed on 3.14) and a LangChain 1.x install.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root:

```dotenv
GROK_API_KEY=your_groq_api_key      # from https://console.groq.com/keys
TAVILY_API_KEY=your_tavily_api_key  # from https://app.tavily.com
```

> The variable is spelled `GROK_API_KEY` but it is a **Groq** key
> (`api.groq.com`), not xAI's Grok. Keep the name consistent between `.env` and
> `src/agents/agents.py`.

`.env` is gitignored and must never be committed.

---

## Running

### CLI

Edit the `topic` in `main.py`, then:

```bash
python main.py
```

Progress and the four outputs are printed to the terminal, including a
`[size] step N input ~X tokens` line before each model call.

### Streamlit UI

```bash
streamlit run app.py
```

Enter a topic, click **Run Research Pipeline**, and watch the four steps run with
raw outputs and the final report shown below.

> Restart the Streamlit server after changing anything in `src/agents/agents.py` –
> the `llm` object is built once at import and a hot-reload won't rebuild it.

---

## Groq free-tier rate limits

The free `on_demand` tier is tight and the pipeline is built around it:

| Limit | Value | How the code handles it |
|-------|-------|-------------------------|
| Input tokens / minute (ITPM)  | ~7,000 | Every blob fed to a model is truncated (`MAX_SEARCH_CHARS`, `MAX_SCRAPED_CHARS`, `MAX_REPORT_CHARS`); `web_search` output is capped; agents get `recursion_limit: 6` and system prompts telling them to call their tool once or twice and stop |
| Output tokens / minute (OTPM) | ~1,000 | `max_tokens=900` on the `llm` so no single response is rejected up front |
| Rolling-window 429 / 413      | –      | `invoke_with_retry()` catches the error, waits ~65 s for the window to clear, and retries (up to 2×) |

**Consequence:** on the free tier the writer's report is capped at ~900 output
tokens (~700 words), so long multi-section reports will be cut short. To get
full-length reports, use Groq's paid Dev tier, switch providers, or generate the
report section by section.

---

## Known limitations

- **Model hallucination.** `qwen/qwen3.8-27b` will invent plausible-sounding
  specifics (fake report titles, made-up statistics) when the scraped source
  material is thin. The Critic step exists partly to flag this.
- **URL hand-off.** The Reader Agent depends on the Search Agent's summary
  containing clean URLs; if it doesn't, the scrape step degrades gracefully but
  returns less.
- `app.py` currently duplicates the four-step flow instead of calling
  `run_research_pipeline`; both paths share the same rate-limit protections.
