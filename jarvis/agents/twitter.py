"""TwitterAgent — Thread creation, authority building, traffic generation."""
import structlog
from langgraph.graph import StateGraph, END
from jarvis.core.state import ContentState

log = structlog.get_logger()

MAX_TWEET_CHARS = 280
THREAD_TWEET_MIN = 5
THREAD_TWEET_MAX = 10


def generate_thread(state: ContentState) -> ContentState:
    content = state["content"]
    topic = content.get("topic", "AI")
    tweets = [
        f"🧵 {THREAD_TWEET_MIN} things about {topic} that will change how you work:\n\n(A thread)",
        f"1/ The basics: {topic} is transforming everything from content creation to trading...",
        f"2/ The real opportunity: Most people don't know you can automate [X] with {topic}...",
        f"3/ The tool you need: [Affiliate link with context]...",
        f"4/ How to get started in 10 minutes: [Step-by-step]...",
        f"5/ The bottom line: If you're not using {topic} yet, you're leaving money on the table.\n\nFollow @FiestyGoatAI for daily AI tools.",
    ]
    tweets = [t[:MAX_TWEET_CHARS] for t in tweets]
    return {**state, "content": {**content, "tweets": tweets, "body": "\n\n".join(tweets)}}


def validate_character_counts(state: ContentState) -> ContentState:
    tweets = state["content"].get("tweets", [])
    violations = [i for i, t in enumerate(tweets) if len(t) > MAX_TWEET_CHARS]
    if violations:
        log.warning("twitter.char_limit_violations", tweet_indices=violations)
    return state


def add_first_mover_timing(state: ContentState) -> ContentState:
    """Mark thread for first-mover publish window (2-hour viral window)."""
    content = state["content"]
    return {**state, "content": {**content, "publish_window_hours": 2, "first_mover": True}}


def queue_for_publish(state: ContentState) -> ContentState:
    log.info("twitter.queued", platform="twitter", tweets=len(state["content"].get("tweets", [])))
    return state


def build_twitter_agent() -> StateGraph:
    graph = StateGraph(ContentState)

    graph.add_node("generate_thread", generate_thread)
    graph.add_node("validate_character_counts", validate_character_counts)
    graph.add_node("add_first_mover_timing", add_first_mover_timing)
    graph.add_node("queue_for_publish", queue_for_publish)

    graph.set_entry_point("generate_thread")
    graph.add_edge("generate_thread", "validate_character_counts")
    graph.add_edge("validate_character_counts", "add_first_mover_timing")
    graph.add_edge("add_first_mover_timing", "queue_for_publish")
    graph.add_edge("queue_for_publish", END)

    return graph.compile()


twitter_agent = build_twitter_agent()
