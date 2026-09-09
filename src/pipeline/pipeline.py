from src.agents.agents import build_search_agent, build_reader_agent, writer_chain, critic_chain
from src.utils import invoke_with_retry, invoke_agent_text, approx_tokens

# Groq free tier allows ~7000 input tokens per rolling minute (ITPM).
# Keep every blob that feeds a model well under that so 4 calls fit the window.
MAX_SEARCH_CHARS = 2000
MAX_SCRAPED_CHARS = 3000
MAX_REPORT_CHARS = 6000


def run_research_pipeline(topic: str) -> dict:

    state = {}

    # step 1 - search agent
    print("\n" + " =" * 50)
    print("step 1 - search agent is working ...")
    print("=" * 50)

    search_agent = build_search_agent()
    search_msg = f"Find recent, reliable and detailed information about: {topic}"
    print(f"[size] step 1 input  ~{approx_tokens(search_msg)} tokens")
    search_text = invoke_agent_text(
        search_agent,
        {"messages": [("user", search_msg)]},
        config={"recursion_limit": 8},  # backstop: agent should search once or twice
        label="step 1 search agent",
        fallback=f"(search failed) topic: {topic}",
    )
    state["search_results"] = search_text[:MAX_SEARCH_CHARS]

    print("\n search result ", state["search_results"])

    # step 2 - reader agent
    print("\n" + " =" * 50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("=" * 50)

    reader_agent = build_reader_agent()
    reader_msg = (
        f"Based on the following search results about '{topic}', "
        f"pick the most relevant URL and scrape it for deeper content.\n\n"
        f"Search Results:\n{state['search_results'][:800]}"
    )
    print(f"[size] step 2 input  ~{approx_tokens(reader_msg)} tokens")
    reader_text = invoke_agent_text(
        reader_agent,
        {"messages": [("user", reader_msg)]},
        config={"recursion_limit": 12},  # backstop against a scrape-retry loop
        label="step 2 reader agent",
        # if the reader gets stuck looping, fall back to the search summary
        fallback=state["search_results"],
    )
    state["scraped_content"] = reader_text[:MAX_SCRAPED_CHARS]

    print("\nscraped content: \n", state["scraped_content"])

    # step 3 - writer chain
    print("\n" + " =" * 50)
    print("step 3 - Writer is drafting the report ...")
    print("=" * 50)

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )
    print(f"[size] step 3 input  ~{approx_tokens(topic + research_combined)} tokens")
    state["report"] = invoke_with_retry(
        writer_chain,
        {"topic": topic, "research": research_combined},
        label="step 3 writer",
    )

    print("\n Final Report\n", state["report"])

    # step 4 - critic chain
    print("\n" + " =" * 50)
    print("step 4 - critic is reviewing the report ")
    print("=" * 50)

    report_for_review = state["report"][:MAX_REPORT_CHARS]
    print(f"[size] step 4 input  ~{approx_tokens(report_for_review)} tokens")
    state["feedback"] = invoke_with_retry(
        critic_chain,
        {"report": report_for_review},
        label="step 4 critic",
    )

    print("\n critic report \n", state["feedback"])

    return state
