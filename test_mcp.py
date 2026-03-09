import asyncio
from mcp.client.sse import sse_client
from mcp import ClientSession
import traceback

URL = "https://mcp.tavily.com/mcp/?tavilyApiKey=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"

async def main():
    try:
        async with sse_client(url=URL) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                print("连接成功，工具列表：")
                for t in tools.tools:
                    print("-", t.name, ":", t.description)
    except Exception as e:
        print("外层异常类型:", type(e))
        print("外层异常信息:", e)
        print("\n=== traceback ===")
        traceback.print_exception(type(e), e, e.__traceback__)

        # 如果是 ExceptionGroup，把子异常也打印出来
        if hasattr(e, "exceptions"):
            for i, sub in enumerate(e.exceptions, 1):
                print(f"\n--- 子异常 {i} ---")
                print("类型:", type(sub))
                print("信息:", sub)
                traceback.print_exception(type(sub), sub, sub.__traceback__)

if __name__ == "__main__":
    asyncio.run(main())