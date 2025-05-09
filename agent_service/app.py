# agent_service/app.py
from dotenv import load_dotenv
load_dotenv()

import os, httpx
from fastapi import FastAPI, HTTPException
from crewai import LLM, Agent
from crewai_tools import BaseTool  # ← import BaseTool from crewai_tools

# 1. Create a proper BaseTool subclass:
class HTTPSearchTool(BaseTool):
    name: str = "http_search"
    description: str = "Fetch raw search results via HTTP from the local Search Service"

    def __init__(self, base_url: str):
        super().__init__()            # ensure BaseTool is initialized
        self.base_url = base_url

    def _run(self, q: str) -> str:
        resp = httpx.get(f"{self.base_url}/search", params={"q": q})
        resp.raise_for_status()
        return resp.json()["results"]

# 2. Instantiate it
SEARCH_URL = os.getenv("SEARCH_SERVICE_URL", "http://localhost:8001")
search_tool = HTTPSearchTool(SEARCH_URL)

# 3. Configure LLM as before
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
    backstory="Fetch raw search results via http_search then synthesize them.",
    tools=[search_tool],  # now a true BaseTool instance
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
