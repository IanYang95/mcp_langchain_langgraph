from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import asyncio
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

async def main():
    async with MultiServerMCPClient(
        {
        "math": {
            "command": "python",
            "args": ["./server/math_server.py"],
            "transport": "stdio",
        },
        "weather": {
            "command": "python",
            "args": ["./server/weather_server.py"],
            "transport": "stdio",
        },			

        }
    ) as client:
        agent = create_react_agent(model, client.get_tools())
        math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
        weather_response = await agent.ainvoke({"messages": "what is the weather in nyc?"})
        print(math_response)
        print(weather_response)
# asyncio 이벤트 루프를 통해 main() 실행
asyncio.run(main())