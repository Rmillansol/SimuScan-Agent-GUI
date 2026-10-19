from fastmcp import FastMCP

mcp = FastMCP(
    name="Math MCP Server",
    instructions="MCP server with simple math tools."
)

def sumar(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b

mcp.tool()(sumar)

if __name__ == "__main__":
    # Adjust this to your transport/configuration.
    # Many FastMCP examples expose it over HTTP so it can be used as a remote server.
    mcp.run()