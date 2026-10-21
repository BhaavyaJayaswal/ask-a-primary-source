"""Launch our timeline MCP server and call its lookup_year tool directly,
to confirm the server works before wiring it into the main app.

Run with:  python scripts/test_mcp_server.py
"""

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PATH = Path(__file__).parent.parent / "mcp_server" / "server.py"


async def main():
    # Describes how to launch the server, same as if you typed this
    # command yourself: python mcp_server/server.py
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Tools the server offers:", [t.name for t in tools.tools])

            for event in ["Battle of Hastings", "The Prince", "something made up"]:
                result = await session.call_tool("lookup_year", {"event": event})
                print(f"\nlookup_year({event!r}) ->")
                print(result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
