"""NewsletterAgent — Weekly AI tools roundup (ConvertKit/Beehiiv)."""
import structlog
from langgraph.graph import StateGraph, END
from jarvis.core.state import ContentState

log = structlog.get_logger()

MIN_WORD_COUNT = 1500
MAX_WORD_COUNT = 2000


def curate_tools(state: ContentState) -> ContentState:
    content = state["content"]
    log.info("newsletter.curate_tools", week=content.get("week", "current"))
    tools = content.get("tools", [{"name": "Example AI Tool", "description": "Does amazing things", "affiliate_link": "#"}])
    return {**state, "content": {**content, "tools": tools}}


def write_roundup(state: ContentState) -> ContentState:
    content = state["content"]
    tools = content.get("tools", [])
    body = f"# This Week in AI Tools\n\n"
    for tool in tools:
        body += f"## {tool['name']}\n{tool['description']}\n[Check it out]({tool.get('affiliate_link', '#')})\n\n"
    body += "\n---\n*FiestyGoat AI Newsletter — AI Education for Everyone*"
    return {**state, "content": {**content, "body": body}}


def validate_compliance(state: ContentState) -> ContentState:
    """CAN-SPAM compliance: unsubscribe link, physical address."""
    content = state["content"]
    body = content.get("body", "")
    if "unsubscribe" not in body.lower():
        body += "\n\n[Unsubscribe]({{unsubscribe_url}}) | FiestyGoat AI LLC, California"
    return {**state, "content": {**content, "body": body}}


def queue_for_send(state: ContentState) -> ContentState:
    log.info("newsletter.queued", platform="newsletter", score=state.get("quality_score", 0))
    return state


def build_newsletter_agent() -> StateGraph:
    graph = StateGraph(ContentState)

    graph.add_node("curate_tools", curate_tools)
    graph.add_node("write_roundup", write_roundup)
    graph.add_node("validate_compliance", validate_compliance)
    graph.add_node("queue_for_send", queue_for_send)

    graph.set_entry_point("curate_tools")
    graph.add_edge("curate_tools", "write_roundup")
    graph.add_edge("write_roundup", "validate_compliance")
    graph.add_edge("validate_compliance", "queue_for_send")
    graph.add_edge("queue_for_send", END)

    return graph.compile()


newsletter_agent = build_newsletter_agent()
