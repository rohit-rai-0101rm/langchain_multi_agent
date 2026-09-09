import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()

GROK_API_KEY = os.getenv("GROK_API_KEY")

llm = ChatOpenAI(
    model="qwen/qwen3.8-27b",
    temperature=0,
    api_key=GROK_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    # Groq free tier caps OUTPUT at 1000 tokens/minute (OTPM). Keep every
    # single response under that so a call isn't rejected before it runs.
    max_tokens=900,
)

#agent 1:Search Agent

SEARCH_SYSTEM_PROMPT = (
    "You are a web research assistant. Call the `web_search` tool AT MOST TWICE "
    "with focused queries. Do NOT keep searching for more sources. As soon as you "
    "have search results, STOP calling tools and reply with a short plain-text "
    "summary of the key findings and the URLs you found."
)

READER_SYSTEM_PROMPT = (
    "You are a web reader. Pick the single most relevant URL from what the user "
    "gives you and call the `scrape_url` tool ONCE on it. You get ONE attempt. "
    "If the scrape fails, returns an error, or returns little content, do NOT "
    "try another URL and do NOT call the tool again - just reply with whatever "
    "you got (or a one-line note that scraping failed) and STOP."
)


def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=SEARCH_SYSTEM_PROMPT,
    )


#agent 2:Reader Agent

def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt=READER_SYSTEM_PROMPT,
    )

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()




#critic_chain 

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()





