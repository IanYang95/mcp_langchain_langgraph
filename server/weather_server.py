from typing import List
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather", host="0.0.0.0", port=8000)

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    return "It's always sunny in New York"

if __name__ == "__main__":
    mcp.run(transport="sse")