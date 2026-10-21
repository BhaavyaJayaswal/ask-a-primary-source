import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "primary_sources"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
GEMINI_MODEL_NAME = "gemini-flash-lite-latest"
TOP_K = 5
MCP_SERVER_PATH = Path(__file__).parent.parent / "mcp_server" / "server.py"


def _mcp_server_params() -> StdioServerParameters:
    return StdioServerParameters(command=sys.executable, args=[str(MCP_SERVER_PATH)])


async def _list_mcp_tool_declarations() -> list[types.FunctionDeclaration]:
    async with stdio_client(_mcp_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return [
                types.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters_json_schema=t.input_schema,
                )
                for t in tools.tools
            ]


async def _call_mcp_tool(name: str, args: dict) -> str:
    async with stdio_client(_mcp_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, args)
            return result.content[0].text


# Loaded once when the server starts, not on every request — these are the
# slow parts (loading a model, opening the database connection).
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_collection(COLLECTION_NAME)
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Filled in by the lifespan startup hook below, once, when the server starts.
gemini_tools: types.Tool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global gemini_tools
    declarations = await _list_mcp_tool_declarations()
    gemini_tools = types.Tool(function_declarations=declarations)
    yield


app = FastAPI(lifespan=lifespan)


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    source_title: str
    author: str
    era: str
    chapter: str
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


def build_prompt(question: str, sources: list[Source]) -> str:
    passages = "\n\n".join(
        f"[Source {i + 1}: {s.source_title} by {s.author}, "
        f"{s.chapter or 'no chapter listed'}]\n{s.text}"
        for i, s in enumerate(sources)
    )
    return f"""You are a careful historical research assistant. For any \
factual claim about history, rely ONLY on the passages below, and mention \
which Source number supports it, e.g. "(Source 2)".

You also have a lookup_year tool with a small fixed list of event dates. \
If the question asks what year or when something happened, and the \
passages below do not clearly state that year — whether because they're \
vague OR because they don't mention the event at all — you MUST call \
lookup_year before answering. Only say you don't know the year if \
lookup_year also fails to find it.

Passages:
{passages}

Question: {question}

Answer:"""


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    query_embedding = embedding_model.encode([request.question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=TOP_K)

    sources = [
        Source(
            source_title=meta["source_title"],
            author=meta["author"],
            era=meta["era"],
            chapter=meta["chapter"],
            text=doc,
        )
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]

    prompt = build_prompt(request.question, sources)
    config = types.GenerateContentConfig(tools=[gemini_tools])
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=[prompt],
        config=config,
    )

    if response.function_calls:
        # The model decided it needs the tool instead of answering straight
        # from the retrieved passages. Actually run it via MCP, then give
        # Gemini the result so it can write the real final answer.
        call = response.function_calls[0]
        tool_result = asyncio.run(_call_mcp_tool(call.name, call.args))
        print(f"[TOOL CALL] {call.name}({call.args}) -> {tool_result}")

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=[
                prompt,
                # Reuse Gemini's own original turn as-is (rather than
                # rebuilding it by hand) so internal tracking metadata it
                # attached — a "thought_signature" — survives into this
                # follow-up call, which Gemini requires.
                response.candidates[0].content,
                types.Part.from_function_response(
                    name=call.name, response={"result": tool_result}
                ),
            ],
            config=config,
        )

    return AskResponse(answer=response.text, sources=sources)
