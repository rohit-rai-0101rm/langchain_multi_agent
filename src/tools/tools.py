from langchain.tools import tool
import requests

import os
from dotenv import load_dotenv
from tavily import TavilyClient

from rich import print
load_dotenv()

tavily=TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search(query:str)->str:
    """
    Search the web for recent and relaible information ona topic.Return Titles,URL ,Snippets and contents
    """
    results=tavily.search(query=query,max_results=5)

    # print(results)

    out=[]
    for r in results["results"]:
        out.append(
        f"Title:{r['title']}\nURL:{r['url']}\nSnippet:{r['content'][:300]}\n"
        )

    return "\n-----\n".join(out)