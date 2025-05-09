from dotenv import load_dotenv
load_dotenv()

import os
import httpx
from fastapi import FastAPI, HTTPException
from crewai import LLM, Agent

# Wrap HTTP call into a CrewAI-compatible tool
define_search_tool = '''
class HTTPSearchTool:
    name = "http_search"
    def __init__(self, base_url):
        self.base_url = base_url

    def run(self, query: str) -> str:
        resp = httpx.get(f"{self.base_url}/search", params={"q": query})
        resp.raise_for_status()
        return resp.json()["results"]
'''

# Instantiate tool and agent
SEARCH_URL = os.getenv("SEARCH_SERVICE_URL", "http://localhost:8001")
exec(define_search_tool)
search_tool = HTTPSearchTool(SEARCH_URL)

default_llm = LLM(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "azure/gpt40"),
    api_key=os.getenv("AZURE_API_KEY"),
    api_base=os.getenv("AZURE_API_BASE"),
    api_version=os.getenv("AZURE_API_VERSION"),
)

agent = Agent(
    llm=default_llm,
    role="Generative AI Researcher",
    goal="Summarize the latest generative AI news",
    backstory=(
        "Fetch raw search results via http_search then synthesize them into a concise summary."
    ),
    tools=[search_tool],
    verbose=True,
)

app = FastAPI(title="Agent Service")

@app.get("/agent")
async def run_agent(q: str):
    try:
        summary = agent.run(q)
        return {"query": q, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
