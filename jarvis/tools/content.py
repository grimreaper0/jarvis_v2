"""Content tools for LangGraph agents."""
from langchain_core.tools import tool


@tool
def score_content_quality(content: dict) -> float:
    """Score content quality on a 0-1 scale deterministically.

    Args:
        content: Dict with title, body, hashtags, media_url fields

    Returns:
        Quality score 0.0-1.0
    """
    score = 0.0
    if content.get("title") and len(content["title"]) > 10:
        score += 0.2
    if content.get("body") and len(content["body"]) > 100:
        score += 0.3
    if content.get("hashtags") and len(content["hashtags"]) >= 3:
        score += 0.2
    if content.get("media_url"):
        score += 0.3
    return round(score, 4)


@tool
def get_trending_topics(platform: str = "instagram") -> list[str]:
    """Get currently trending topics for a social media platform.

    Args:
        platform: 'instagram' | 'tiktok' | 'twitter' | 'youtube'

    Returns:
        List of trending topic strings
    """
    defaults = {
        "instagram": ["#AITools", "#MachineLearning", "#ChatGPT", "#TechNews"],
        "tiktok": ["#FYP", "#AITok", "#TechTok", "#AItools"],
        "twitter": ["#AI", "#MachineLearning", "#OpenAI", "#LLM"],
        "youtube": ["AI tutorial", "automation tools", "make money with AI"],
    }
    return defaults.get(platform, ["#AI", "#Tech"])


@tool
def check_affiliate_link(tool_name: str) -> dict:
    """Look up affiliate link for an AI tool.

    Args:
        tool_name: Name of the AI tool

    Returns:
        Dict with affiliate_url, commission_rate, program_name
    """
    return {
        "tool_name": tool_name,
        "affiliate_url": f"https://example.com/ref/{tool_name.lower().replace(' ', '-')}",
        "commission_rate": 0.30,
        "program_name": "Implement with affiliate network API",
    }
