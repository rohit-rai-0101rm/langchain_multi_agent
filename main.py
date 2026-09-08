from src.tools.tools import web_search

from src.tools.tools import scrape_url

output=web_search.invoke('latest news on USA and IRAN war')
# output=web_search("Latest news on AI research")
print(output)
