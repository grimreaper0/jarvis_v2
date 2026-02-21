"""LangGraph Admin UI — read-only system overview at /admin.

Returns a self-contained HTML page with embedded CSS/JS.
All data is server-rendered; no external CDN dependencies.
"""
from __future__ import annotations

from config.settings import get_settings


# ---------------------------------------------------------------------------
# Data catalogue (source of truth for the admin UI)
# ---------------------------------------------------------------------------

PRIME_DIRECTIVES = [
    {
        "number": 1,
        "name": "HELP CODY",
        "description": "Every system exists to help the operator (you) succeed. "
                       "When confidence is low, pause and ask rather than guess.",
        "langgraph_location": "interrupt() in clarify_node — confidence.py",
        "tier": "Cross-cutting",
        "color": "#6366f1",
        "details": [
            "Implemented via interrupt() in the Confidence Graph's clarify_node",
            "When confidence < 0.60, graph pauses and waits for operator input",
            "Resume via POST /confidence/resume with your decision",
            "All decisions audited to confidence_audit_log in PostgreSQL",
        ],
    },
    {
        "number": 2,
        "name": "GENERATE REVENUE",
        "description": "Every opportunity must be evaluated and executed autonomously. "
                       "Revenue bots run 24/7, never idle, never miss an opportunity.",
        "langgraph_location": "Revenue Graph (Tier 2) + Trading Graph (Tier 2) + all Tier 3 agents",
        "tier": "Tier 2 + Tier 3",
        "color": "#22c55e",
        "details": [
            "Revenue Graph evaluates opportunities, routes to Tier 3 agents via Redis",
            "Trading Graph validates signals, sizes positions, submits to Alpaca",
            "Tier 3 agents: Instagram, YouTube, Newsletter, TikTok, Twitter, SEO Blog",
            "All opportunities gated by 4-layer Confidence Gate (≥0.90 to execute)",
        ],
    },
    {
        "number": 4,
        "name": "OPERATE WITH CONFIDENCE",
        "description": "Every action requires ≥90% confidence. Use all tools, critical thinking, "
                       "and sequential reasoning. Quality over speed. Fix problems or escalate — never skip.",
        "langgraph_location": "LLMRouter._openai_compat_complete — injected as system prompt prefix on EVERY LLM call",
        "tier": "Cross-cutting (all tiers)",
        "color": "#ec4899",
        "details": [
            "Injected automatically in LLMRouter._openai_compat_complete() via jarvis/core/mandate.py",
            "Every LLM call in BILLY (Tier 1, 2, 3) carries the mandate — it cannot be overridden",
            "Task-specific system prompts are APPENDED after the mandate, not replacing it",
            "6 principles: CONFIDENCE ≥90%, QUALITY over speed, THINK HARD, PARALLELIZE, FIX OR ASK, CONTEXT",
            "Full verbatim mandate: 'as you continue to work, these are complicated tasks...'",
        ],
    },
    {
        "number": 3,
        "name": "CONTINUOUSLY LEARN",
        "description": "Never be idle. Always be learning from cutting-edge sources. "
                       "Research Bot must ALWAYS be running.",
        "langgraph_location": "Tier 1 ContinuousWorkers — explicitly NOT LangGraph",
        "tier": "Tier 1 (No LangGraph)",
        "color": "#f59e0b",
        "details": [
            "LearningWorker: 28 RSS feeds (arXiv, Reddit, research labs, revenue blogs)",
            "ResearchWorker: 19 dark-hole sources (MoltBook, Medium, ProductHunt, HN)",
            "Explicitly uses raw Python + Redis — no LangGraph to avoid idle token waste",
            "supervisord: autostart=true, autorestart=true — NEVER offline",
            "Idle behavior: scrapes all sources every hour even with empty queue",
        ],
    },
]

COMMANDMENTS = [
    {
        "name": "Research Bot MUST Always Run",
        "rule": "supervisord autorestart=true — auto-restart on any crash",
        "langgraph_equivalent": "N/A — Tier 1 worker, no LangGraph",
    },
    {
        "name": "Never Act When Uncertain",
        "rule": "Confidence < 0.60 → clarify (human-in-the-loop interrupt)",
        "langgraph_equivalent": "interrupt() in clarify_node of confidence.py",
    },
    {
        "name": "Local-First: Everything on Mac Studio",
        "rule": "LLM Router tries vLLM local (mlx_lm.server) first ($0), then free cloud APIs, then paid dev-only",
        "langgraph_equivalent": "LLMRouter TASK_ROUTING — vllm_local first in every task type",
    },
    {
        "name": "Operating Mandate Always Active",
        "rule": "Prime Directive 4 injected as system prompt on every LLM call — 90%+ confidence, quality over speed, fix or escalate",
        "langgraph_equivalent": "LLMRouter._openai_compat_complete — MANDATE_SYSTEM_PROMPT prepended to all messages",
    },
    {
        "name": "Fail-Safe: Safety Over Automation",
        "rule": "Guardrails block trades > $100, daily loss > $300, content below 0.70",
        "langgraph_equivalent": "check_guardrails node in trading graph, check_guidelines in content graph",
    },
    {
        "name": "Pattern-Driven: Learn What Works",
        "rule": "AutoMem golden rules feed into confidence historical layer (+0.15 each)",
        "langgraph_equivalent": "score_historical node in confidence.py queries AutoMem",
    },
    {
        "name": "Research Without Implementation = FAILURE",
        "rule": "Every research session must produce running code, not just docs",
        "langgraph_equivalent": "Research → graph node → code, not just logs",
    },
]

GRAPHS = [
    {
        "name": "Revenue Graph",
        "file": "jarvis/graphs/revenue.py",
        "queue": "revenue_opportunity",
        "description": "Evaluates revenue opportunities end-to-end. Enriches with AutoMem history, "
                       "scores confidence, and routes to the correct Tier 3 agent queue.",
        "state": "RevenueState",
        "checkpointer": "MemorySaver",
        "nodes": [
            {"name": "load_opportunity", "type": "start", "desc": "Validates required fields (task_type, description). Sets defaults."},
            {"name": "enrich_opportunity", "type": "normal", "desc": "AutoMem pattern search adds historical context to opportunity."},
            {"name": "evaluate_confidence", "type": "normal", "desc": "4-layer confidence scoring (base/validation/historical/reflexive)."},
            {"name": "route_decision", "type": "normal", "desc": "Sets action=execute/delegate/skip based on score vs thresholds."},
            {"name": "execute_opportunity", "type": "terminal", "desc": "≥0.90: routes to Tier 3 queue (instagram_task, trading_signal, etc.)"},
            {"name": "delegate_opportunity", "type": "terminal", "desc": "0.60-0.89: pushes to human_review queue with full context."},
            {"name": "skip_opportunity", "type": "terminal", "desc": "<0.60: logs skip reason, discards opportunity."},
        ],
        "flow": "load → enrich → evaluate_confidence → route_decision → [execute | delegate | skip] → END",
        "queue_map": {
            "content_post / content_story / content_reel": "instagram_task",
            "content_video / seo_optimization": "youtube_task",
            "trade_stock / trade_crypto / backtest": "trading_signal",
            "newsletter / newsletter_issue": "newsletter_task",
            "tiktok_video / tiktok_trend": "tiktok_task",
        },
    },
    {
        "name": "Trading Graph",
        "file": "jarvis/graphs/trading.py",
        "queue": "trading_signal",
        "description": "Trading signal decision pipeline. Validates signals, applies financial guardrails, "
                       "sizes positions via Kelly Criterion, submits to Alpaca via Tier 1 TradingWorker.",
        "state": "TradingState",
        "checkpointer": "MemorySaver",
        "nodes": [
            {"name": "validate_signal", "type": "start", "desc": "Checks required fields (symbol, action, price, confidence). Normalises action to uppercase."},
            {"name": "check_guardrails", "type": "normal", "desc": "Financial guardrails: max $100/trade, $500 daily loss, 3 consecutive losses = circuit breaker."},
            {"name": "size_position", "type": "normal", "desc": "Kelly Criterion: f* = (b·p - q)/b, clamped to 1-10% of bankroll, hard cap $100."},
            {"name": "risk_approve", "type": "normal", "desc": "Final approval gate — all checks passed, logs approval."},
            {"name": "submit_order", "type": "terminal", "desc": "Pushes approved order to trading_execution queue for Tier 1 TradingWorker."},
            {"name": "reject_order", "type": "terminal", "desc": "Logs rejection reason to audit log, returns order_id=None."},
        ],
        "flow": "validate_signal → check_guardrails → [size_position → risk_approve → submit_order → END] or [reject_order → END]",
        "queue_map": {},
    },
    {
        "name": "Content Gate",
        "file": "jarvis/graphs/content.py",
        "queue": "content_gate (also: instagram_content_check, youtube_content_check, etc.)",
        "description": "Platform-specific content quality gate. Fully deterministic scoring (no LLM) "
                       "for speed. Checks rate limits, quality floors, and blocked keywords.",
        "state": "ContentState",
        "checkpointer": "MemorySaver",
        "nodes": [
            {"name": "score_content", "type": "start", "desc": "Platform-specific scoring (Instagram, YouTube, Newsletter, TikTok, generic). Returns quality_score 0.0-1.0."},
            {"name": "check_guidelines", "type": "normal", "desc": "Rate limit check (platform posts/hour), quality floor (≥0.70), blocked keywords sweep."},
            {"name": "approve_content", "type": "terminal", "desc": "Pushes to {platform}_publish queue for scheduled posting."},
            {"name": "reject_content", "type": "terminal", "desc": "Pushes to {platform}_regenerate queue with rejection reason."},
        ],
        "flow": "score_content → check_guidelines → [approve_content → END] or [reject_content → END]",
        "scoring": {
            "Instagram": "Caption 50-300 chars +0.25, Hashtags 5-30 +0.25, CTA +0.25, media_url +0.25",
            "YouTube": "Title 30-70 chars +0.25, Description 100+ chars +0.25, 3+ keywords +0.25, thumbnail +0.25",
            "Newsletter": "Word count 1500-2000 +0.30, affiliate links +0.25, CTA +0.25, subject line +0.20",
            "TikTok": "Hook present +0.30, trending sounds +0.25, caption 10-150 chars +0.25, video/script +0.20",
        },
        "queue_map": {},
    },
    {
        "name": "Confidence Graph",
        "file": "jarvis/graphs/confidence.py",
        "queue": "confidence_eval",
        "description": "Standalone 4-layer confidence subgraph. Each scoring layer is a separate async node "
                       "with its own checkpoint. When confidence < 0.60, interrupt() pauses the graph "
                       "and surfaces a decision to the operator.",
        "state": "ConfidenceState",
        "checkpointer": "MemorySaver",
        "has_interrupt": True,
        "nodes": [
            {"name": "score_base", "type": "start", "desc": "Task clarity: recognised type +0.35, description ≥20 chars +0.35, non-empty context +0.30. Max: 1.0"},
            {"name": "score_validation", "type": "normal", "desc": "Domain heuristics: api_available +0.20, data_fresh +0.15, within_limits violation -0.30. Baseline 0.50."},
            {"name": "score_historical", "type": "normal", "desc": "AutoMem pattern match: +0.05/pattern (cap +0.20), golden rules +0.15 each (cap +0.30). Baseline 0.30."},
            {"name": "score_reflexive", "type": "normal", "desc": "Coherence check: spread ≤0.15 → avg+0.10, spread ≤0.30 → avg, spread >0.30 → avg-0.15."},
            {"name": "finalize", "type": "normal", "desc": "Weighted combination → final score → decision. Audits to confidence_audit_log."},
            {"name": "clarify_node", "type": "interrupt", "desc": "⚡ INTERRUPT: Calls interrupt() — freezes graph, surfaces question+options to operator. Resumes on Command(resume=choice)."},
        ],
        "flow": "score_base → score_validation → score_historical → score_reflexive → finalize → [execute→END | delegate→END | clarify→clarify_node→END]",
        "queue_map": {},
    },
]

WORKERS = [
    {
        "name": "LearningWorker",
        "file": "jarvis/workers/learning.py",
        "queue": "learning_scrape",
        "worker_name": "learning_bot",
        "description": "24/7 RSS + arXiv scraping. Stores articles to PostgreSQL with pgvector embeddings. "
                       "Implements Prime Directive #3: Continuously Learn.",
        "idle_behavior": "Scrapes all 28 RSS feeds every hour",
        "sources": "28 feeds: arXiv (cs.AI/cs.LG/cs.MA), Reddit (LocalLLaMA/ML/AI/Singularity), "
                   "OpenAI/DeepMind/Anthropic/Meta AI blogs, revenue feeds (niche_pursuits, backlinko, etc.)",
        "dispatches_to": "None (stores to AutoMem directly)",
        "langgraph": "NO — explicitly avoids LangGraph to prevent idle token waste",
    },
    {
        "name": "ResearchWorker",
        "file": "jarvis/workers/research.py",
        "queue": "research_request",
        "worker_name": "research_bot",
        "description": "19-source deep research bot. Monitors dark-hole sources (MoltBook, Medium, ProductHunt, HN). "
                       "Implements Prime Directive #3: Continuously Learn.",
        "idle_behavior": "Scrapes all 19 sources every hour",
        "sources": "MoltBook AI network, Reddit AI communities, arXiv papers, ProductHunt trends, "
                   "HackerNews viral projects, Medium first-mover detection",
        "dispatches_to": "Tier 2 Revenue Graph (via revenue_opportunity queue) for high-value discoveries",
        "langgraph": "NO — ContinuousWorker base class, Redis poll loop only",
    },
    {
        "name": "TradingWorker",
        "file": "jarvis/workers/trading.py",
        "queue": "trading_execution",
        "worker_name": "trading_bot",
        "description": "Alpaca paper trading execution. Receives approved orders from Tier 2 Trading Graph. "
                       "Runs 3 strategies: VWAP Mean Reversion, Opening Range Breakout, Bollinger+RSI.",
        "idle_behavior": "Performance analysis, strategy optimization, next-day preparation",
        "sources": "Alpaca paper-api.alpaca.markets",
        "dispatches_to": "trading_signal queue → Tier 2 Trading Graph for validation",
        "langgraph": "NO — ContinuousWorker base class. Execution only, no decision logic here.",
    },
    {
        "name": "HealthWorker",
        "file": "jarvis/workers/health.py",
        "queue": "health_check",
        "worker_name": "health_monitor",
        "description": "24/7 infrastructure oversight: Redis, PostgreSQL, disk space, memory, "
                       "supervisor processes, cron jobs. Auto-remediation on failures.",
        "idle_behavior": "Full system health check every 5 minutes",
        "sources": "localhost services (Redis, PostgreSQL, Ollama, supervisord)",
        "dispatches_to": "notifications queue for critical alerts",
        "langgraph": "NO — ContinuousWorker base class. Monitoring only.",
    },
]

AGENTS = [
    {
        "name": "Instagram Agent",
        "file": "jarvis/agents/instagram.py",
        "queue": "instagram_task",
        "platform": "Instagram",
        "description": "End-to-end Instagram content creation. Research → caption generation "
                       "(self-consistency: 3 variants, pick best) → hashtag selection → affiliate link → quality gate.",
        "nodes": ["research_topic", "generate_caption", "select_hashtags", "add_affiliate_link", "quality_gate"],
        "min_quality": "0.70",
        "llm_calls": "3 async caption variants (self-consistency)",
        "output_queue": "content_gate",
    },
    {
        "name": "YouTube Agent",
        "file": "jarvis/agents/youtube.py",
        "queue": "youtube_task",
        "platform": "YouTube",
        "description": "Faceless video content planning. SEO-optimized titles, descriptions, keyword research, "
                       "thumbnail concepts. Self-consistency for title generation.",
        "nodes": ["research_topic", "generate_title", "generate_description", "select_keywords", "thumbnail_concept", "quality_gate"],
        "min_quality": "0.70",
        "llm_calls": "3 async title variants (self-consistency)",
        "output_queue": "youtube_content_check",
    },
    {
        "name": "Newsletter Agent",
        "file": "jarvis/agents/newsletter.py",
        "queue": "newsletter_task",
        "platform": "Email (ConvertKit/Beehiiv)",
        "description": "Weekly AI tools roundup. 1,500-2,000 words, affiliate links, premium tier, CAN-SPAM compliant.",
        "nodes": ["research_topics", "generate_sections", "add_affiliate_links", "format_newsletter", "quality_gate"],
        "min_quality": "0.70",
        "llm_calls": "Per-section generation",
        "output_queue": "newsletter_content_check",
    },
    {
        "name": "SEO Blog Agent",
        "file": "jarvis/agents/seo_blog.py",
        "queue": "seo_blog_task",
        "platform": "WordPress/Blog",
        "description": "Long-form AI tool content 1,500-2,500 words. Keyword research, on-page SEO, "
                       "internal linking, schema markup. Evergreen Google traffic.",
        "nodes": ["keyword_research", "generate_outline", "write_sections", "add_seo_meta", "quality_gate"],
        "min_quality": "0.70",
        "llm_calls": "Per-section generation + SEO optimization",
        "output_queue": "blog_content_check",
    },
    {
        "name": "TikTok Agent",
        "file": "jarvis/agents/tiktok.py",
        "queue": "tiktok_task",
        "platform": "TikTok",
        "description": "Viral short-form content. Trending sounds, FYP optimization, hook generation, "
                       "viral scoring. Creator Fund + TikTok Shop + affiliate integration.",
        "nodes": ["research_trends", "generate_hook", "write_script", "select_sounds", "quality_gate"],
        "min_quality": "0.70",
        "llm_calls": "Hook + script generation",
        "output_queue": "tiktok_content_check",
    },
    {
        "name": "Twitter Agent",
        "file": "jarvis/agents/twitter.py",
        "queue": "twitter_task",
        "platform": "Twitter/X",
        "description": "Thread creation, character validation, rate limits, first-mover timing. "
                       "Authority building + traffic generation.",
        "nodes": ["research_topic", "generate_thread", "validate_length", "optimize_timing", "quality_gate"],
        "min_quality": "0.70",
        "llm_calls": "Thread generation",
        "output_queue": "twitter_content_check",
    },
    {
        "name": "Trading Agent",
        "file": "jarvis/agents/trading_agent.py",
        "queue": "trading_analysis",
        "platform": "Alpaca (stocks)",
        "description": "Strategy analysis and signal generation. Runs VWAP, ORB, Bollinger+RSI strategies. "
                       "Outputs validated signals to Tier 2 Trading Graph.",
        "nodes": ["screen_symbols", "analyze_signals", "score_confidence", "submit_signal"],
        "min_quality": "0.90 (confidence gate)",
        "llm_calls": "Market analysis with deepseek-r1:7b reasoning",
        "output_queue": "trading_signal → Trading Graph",
    },
]

GUARDRAILS = [
    {"name": "max_trade_usd", "type": "FINANCIAL", "value": "$100.00", "description": "Maximum size per trade. Single-share price check before order submission."},
    {"name": "max_daily_loss_usd", "type": "FINANCIAL", "value": "$300.00", "description": "Daily loss circuit breaker. Halt all trading if daily losses exceed this."},
    {"name": "circuit_breaker_losses", "type": "FINANCIAL", "value": "3 consecutive", "description": "After 3 consecutive losing trades, halt trading for the session."},
    {"name": "min_content_quality", "type": "QUALITY", "value": "0.70", "description": "Minimum quality score for all content platforms. Below this → regenerate."},
    {"name": "max_instagram_posts_per_hour", "type": "RATE_LIMIT", "value": "3/hour", "description": "Instagram rate limit. Exceeding risks account flags/shadowban."},
    {"name": "max_twitter_posts_per_hour", "type": "RATE_LIMIT", "value": "5/hour", "description": "Twitter rate limit. Respects API tier limits."},
    {"name": "blocked_keywords_content", "type": "CONTENT", "value": "guaranteed returns, get rich quick, 100% profit", "description": "Blocks financial fraud language from all content outputs."},
    {"name": "confidence_execute_threshold", "type": "QUALITY", "value": "0.90", "description": "Minimum confidence to execute autonomously. Below → delegate or clarify."},
    {"name": "confidence_delegate_threshold", "type": "QUALITY", "value": "0.60", "description": "Minimum confidence to delegate. Below → interrupt() and ask operator."},
]

LLMS = [
    # ── Tier 1: vLLM Local — Mac Studio M2 Max (always-on, $0) ─────────────
    {
        "name": "Qwen/Qwen3-4B",
        "provider": "vLLM Local — Mac Studio M2 Max",
        "cost": "$0.00",
        "size": "~2.5GB",
        "speed": "~5-8s",
        "best_for": "Fast general tasks, content drafts, simple routing decisions",
        "not_for": "Heavy reasoning",
        "task_types": "simple, content",
        "config": "vllm serve Qwen/Qwen3-4B --device mps --port 8000",
        "priority": 1,
    },
    {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "provider": "vLLM Local — Mac Studio M2 Max",
        "cost": "$0.00",
        "size": "~0.5GB",
        "speed": "~2-3s",
        "best_for": "Ultra-fast simple tasks, high-frequency calls",
        "not_for": "Quality-sensitive tasks",
        "task_types": "simple (ultra-fast path)",
        "config": "vllm serve Qwen/Qwen2.5-0.5B-Instruct --device mps --port 8000",
        "priority": 2,
    },
    {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "provider": "vLLM Local — Mac Studio M2 Max",
        "cost": "$0.00",
        "size": "~4.7GB",
        "speed": "~20-40s",
        "best_for": "Trading analysis, chain-of-thought reasoning, complex decisions",
        "not_for": "Real-time responses, simple tasks",
        "task_types": "reasoning, trading",
        "config": "vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --device mps --port 8000 — LLMRequest(reasoning=True)",
        "priority": 3,
    },
    {
        "name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "provider": "vLLM Local — Mac Studio M2 Max",
        "cost": "$0.00",
        "size": "~1.0GB",
        "speed": "~5s",
        "best_for": "Code generation, debugging, implementation specs",
        "not_for": "Creative writing, reasoning",
        "task_types": "coding",
        "config": "vllm serve Qwen/Qwen2.5-Coder-1.5B-Instruct --device mps --port 8000",
        "priority": 4,
    },
    {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "provider": "vLLM Local — Mac Studio M2 Max",
        "cost": "$0.00",
        "size": "~4.1GB",
        "speed": "~15s",
        "best_for": "Creative writing, marketing copy, social media captions",
        "not_for": "Technical reasoning",
        "task_types": "creative",
        "config": "vllm serve mistralai/Mistral-7B-Instruct-v0.3 --device mps --port 8000",
        "priority": 5,
    },
    {
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "provider": "vLLM Local — Mac Studio M2 Max",
        "cost": "$0.00",
        "size": "~2.0GB",
        "speed": "~8s",
        "best_for": "General tasks fallback, summarization",
        "not_for": "Reasoning, long documents",
        "task_types": "simple fallback",
        "config": "vllm serve meta-llama/Llama-3.2-3B-Instruct --device mps --port 8000",
        "priority": 6,
    },
    # ── Tier 2: vLLM Remote — AWS g5.xlarge / New Rig (on-demand) ───────────
    {
        "name": "Qwen/Qwen3-30B-A3B (MoE)",
        "provider": "vLLM Remote — GPU Server",
        "cost": "$0.73/hr on g5.xlarge — stop when idle | $0 on new rig in 2 weeks",
        "size": "18GB VRAM (3B active params via MoE)",
        "speed": "~25 tok/s",
        "best_for": "Primary heavy model — analysis, long-context 128K, general quality reasoning",
        "not_for": "Always-on (instance cost until new rig)",
        "task_types": "analysis, reasoning, creative",
        "config": "keyring.set_password('jarvis_v2', 'vllm_base_url', 'http://<ip>:8000/v1')",
        "priority": 7,
    },
    {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "provider": "vLLM Remote — GPU Server",
        "cost": "$0.73/hr shared | $0 on new rig",
        "size": "~8.5GB VRAM",
        "speed": "Better reasoning than local 7B",
        "best_for": "Heavy trading analysis, deep reasoning when local R1-7B isn\'t enough",
        "not_for": "Always-on before new rig",
        "task_types": "reasoning, trading (GPU tier)",
        "config": "vllm_base_url in Keychain — auto-selected when GPU online",
        "priority": 8,
    },
    {
        "name": "mistralai/Devstral-Small-2505 (Devstral-24B)",
        "provider": "vLLM Remote — GPU Server",
        "cost": "$0.73/hr shared | $0 on new rig",
        "size": "~14GB VRAM",
        "speed": "~15 tok/s",
        "best_for": "Complex code generation, architecture specs, agentic coding tasks",
        "not_for": "Always-on before new rig",
        "task_types": "coding (GPU tier)",
        "config": "vllm_base_url in Keychain — auto-selected for coding when GPU online",
        "priority": 9,
    },
    # ── NEW RIG (2 weeks): Intel Core Ultra 9 285K + 2x RTX 5060 Ti 16GB ────
    {
        "name": "Qwen3.5-397B-A17B (future — new rig)",
        "provider": "vLLM Local — New Rig (Intel Core Ultra 9 + 2x RTX 5060 Ti, 32GB VRAM)",
        "cost": "$0.00 — local inference",
        "size": "17B active (397B MoE, FP8), ~8-10GB active VRAM",
        "speed": "8.6x faster than Qwen3-30B, 25+ tok/s estimated",
        "best_for": "All heavy tasks — analysis, coding, reasoning, long-context 256K",
        "not_for": "N/A — will be the primary model on new rig",
        "task_types": "ALL task types — primary model on new rig",
        "config": "Available in 2 weeks. Tensor parallel across 2x RTX 5060 Ti.",
        "priority": 10,
    },
    # ── Tier 3: Free Cloud APIs ──────────────────────────────────────────────
    {
        "name": "Groq: llama-3.3-70b-versatile",
        "provider": "Groq Cloud (free tier)",
        "cost": "$0.00 (rate-limited)",
        "size": "70B cloud",
        "speed": "~1-3s (fastest cloud inference)",
        "best_for": "70B quality when GPU offline, content, analysis fallback",
        "not_for": "High-frequency calls (rate limits)",
        "task_types": "simple, analysis, content fallback",
        "config": "keyring.set_password('jarvis_v2', 'groq_api_key', 'gsk_...') — console.groq.com",
        "priority": 11,
    },
    {
        "name": "Groq: deepseek-r1-distill-llama-70b",
        "provider": "Groq Cloud (free tier)",
        "cost": "$0.00 (rate-limited)",
        "size": "70B cloud",
        "speed": "~2-5s",
        "best_for": "Reasoning fallback when local/GPU offline",
        "not_for": "High-frequency reasoning",
        "task_types": "reasoning, trading fallback",
        "config": "groq_api_key in Keychain — auto-selected",
        "priority": 12,
    },
    {
        "name": "DeepSeek: deepseek-reasoner (R1 full)",
        "provider": "DeepSeek API",
        "cost": "$0.55/M input, $2.19/M output",
        "size": "Full R1 cloud",
        "speed": "~5-15s",
        "best_for": "Highest-quality reasoning when all local options fail",
        "not_for": "High-frequency calls",
        "task_types": "reasoning final fallback",
        "config": "keyring.set_password('jarvis_v2', 'deepseek_api_key', 'sk-...')",
        "priority": 13,
    },
    {
        "name": "OpenRouter: Qwen3.5-397B-A17B (free API)",
        "provider": "OpenRouter (free pool)",
        "cost": "$0.00 (free model tier)",
        "size": "397B MoE cloud, 17B active",
        "speed": "Variable (shared free tier)",
        "best_for": "Long-context tasks 256K, highest quality before new rig arrives",
        "not_for": "Latency-sensitive tasks",
        "task_types": "long_ctx, analysis (cloud fallback)",
        "config": "keyring.set_password('jarvis_v2', 'openrouter_api_key', 'sk-or-...')",
        "priority": 14,
    },
    # ── Tier 4: Dev-Only Paid ────────────────────────────────────────────────
    {
        "name": "claude-sonnet-4-6",
        "provider": "Anthropic API — DEV ONLY",
        "cost": "$3/M input, $15/M output",
        "size": "Cloud",
        "speed": "<5s",
        "best_for": "Development, architecture decisions — NEVER call from production BILLY bots",
        "not_for": "Production use — use vLLM or free cloud instead",
        "task_types": "dev only — LLMRequest(backend=LLMBackend.CLAUDE)",
        "config": "anthropic_api_key in Keychain",
        "priority": 15,
    },
    {
        "name": "grok-4-fast-reasoning",
        "provider": "xAI API — configured",
        "cost": "Per-token (see xAI pricing)",
        "size": "Cloud",
        "speed": "<10s",
        "best_for": "Validation, research second-opinion, real-time web awareness",
        "not_for": "High-frequency calls",
        "task_types": "creative validation — LLMRequest(backend=LLMBackend.GROK)",
        "config": "xai_api_key in Keychain — configured \u2705",
        "priority": 16,
    },
]

CONFIDENCE_LAYERS = [
    {
        "layer": "Base",
        "weight": 0.25,
        "description": "Task clarity — how well-defined is the request?",
        "scoring": [
            "+0.35 if task_type is in KNOWN_TASK_TYPES (20 recognised types)",
            "+0.35 if description is ≥20 characters",
            "+0.30 if context dict is non-empty",
        ],
        "max": "1.00",
    },
    {
        "layer": "Validation",
        "weight": 0.25,
        "description": "Domain checks — are preconditions met?",
        "scoring": [
            "Baseline: 0.50",
            "+0.20 if context.api_available = True",
            "+0.15 if context.data_fresh = True",
            "+0.15 if context.within_limits = True",
            "-0.30 if context.within_limits = False (guardrail violation)",
        ],
        "max": "1.00",
    },
    {
        "layer": "Historical",
        "weight": 0.30,
        "description": "AutoMem pattern match — what does past experience say?",
        "scoring": [
            "Baseline: 0.30 (novel task, no history)",
            "+0.05 per matching pattern, capped at +0.20",
            "+0.15 per golden rule applied, capped at +0.30",
        ],
        "max": "1.00",
    },
    {
        "layer": "Reflexive",
        "weight": 0.20,
        "description": "Self-assessment — do the first 3 layers agree?",
        "scoring": [
            "Compute spread = max(layers) - min(layers)",
            "spread ≤0.15 → avg + 0.10 (high agreement bonus)",
            "spread ≤0.30 → avg (neutral)",
            "spread  >0.30 → avg - 0.15 (high divergence penalty)",
        ],
        "max": "1.00",
    },
]

STATES = [
    {
        "name": "BaseState",
        "fields": [
            ("messages", "Annotated[list, add_messages]", "Conversation history, append-only via add_messages reducer"),
            ("confidence", "float", "Current confidence score (0.0-1.0)"),
            ("error", "str | None", "Error message from any node, None if clean"),
        ],
    },
    {
        "name": "RevenueState",
        "extends": "BaseState",
        "fields": [
            ("opportunity", "dict[str, Any]", "Full opportunity payload (task_type, description, context, source)"),
            ("evaluation_score", "float", "Final confidence score from evaluate_confidence node"),
            ("action", "str", "execute | delegate | skip — set by route_decision node"),
            ("delegated_to", "str | None", "Queue name if delegated, None otherwise"),
        ],
    },
    {
        "name": "TradingState",
        "extends": "BaseState",
        "fields": [
            ("symbol", "str", "Stock ticker (e.g. NVDA, SPY)"),
            ("signal", "dict[str, Any]", "Full signal payload (action, price, confidence, strategy)"),
            ("risk_approved", "bool", "True if guardrails passed and risk is approved"),
            ("order_id", "str | None", "queued if submitted, None if rejected"),
            ("position_size", "float", "Kelly Criterion position size in USD"),
        ],
    },
    {
        "name": "ContentState",
        "extends": "BaseState",
        "fields": [
            ("content", "dict[str, Any]", "Platform-specific content payload (caption, hashtags, media_url, etc.)"),
            ("platform", "str", "instagram | youtube | newsletter | tiktok | generic"),
            ("quality_score", "float", "Deterministic quality score (0.0-1.0)"),
            ("approved", "bool", "True if quality ≥ 0.70 and guidelines passed"),
            ("rejection_reason", "str | None", "Reason string if rejected, None if approved"),
        ],
    },
    {
        "name": "ConfidenceState",
        "extends": "BaseState",
        "fields": [
            ("base_score", "float", "Layer 1 score (task clarity)"),
            ("validation_score", "float", "Layer 2 score (domain checks)"),
            ("historical_score", "float", "Layer 3 score (AutoMem patterns)"),
            ("reflexive_score", "float", "Layer 4 score (coherence self-assessment)"),
            ("final_score", "float", "Weighted combination of all 4 layers"),
            ("decision", "str", "execute | delegate | clarify — or operator's override after resume"),
            ("user_decision", "str | None (NotRequired)", "Set after interrupt() resume — operator's choice"),
        ],
    },
]

GLOSSARY = [
    {
        "term_familiar": "Guardrail",
        "langgraph_equivalent": "Guard node + conditional edge",
        "how_we_use": "check_guardrails node in trading.py; check_guidelines node in content.py. Explicit Python logic blocks the graph path before unsafe actions.",
    },
    {
        "term_familiar": "Skill / Command",
        "langgraph_equivalent": "Tool (function the LLM can call within a node)",
        "how_we_use": "jarvis/tools/ directory: search.py, trading.py, content.py. Registered as @tool decorated functions, callable by LLM nodes.",
    },
    {
        "term_familiar": "Role / Persona",
        "langgraph_equivalent": "System prompt in a node + node-level state",
        "how_we_use": "Each Tier 3 agent has a specific purpose (instagram, trading, etc.). The 'role' is encoded in the LLM prompt within generate_* nodes.",
    },
    {
        "term_familiar": "Memory / Context",
        "langgraph_equivalent": "Checkpointer (short-term) + AutoMem/Neo4j (long-term)",
        "how_we_use": "MemorySaver checkpointer persists state within a thread_id. AutoMem (PostgreSQL + pgvector) stores patterns across all runs.",
    },
    {
        "term_familiar": "Confidence Gate",
        "langgraph_equivalent": "Standalone StateGraph subgraph with 4 scoring nodes",
        "how_we_use": "jarvis/graphs/confidence.py — reusable subgraph. Each scoring layer is a separate async node for independent checkpointing and audit.",
    },
    {
        "term_familiar": "Interrupt / Ask for input",
        "langgraph_equivalent": "interrupt() from langgraph.types",
        "how_we_use": "Called in clarify_node when confidence < 0.60. Freezes graph state. Resume via POST /confidence/resume with Command(resume=choice).",
    },
    {
        "term_familiar": "Worker / Bot",
        "langgraph_equivalent": "Tier 1: NOT LangGraph (ContinuousWorker). Tier 3: LangGraph subgraph (StateGraph)",
        "how_we_use": "Tier 1 workers use raw Python + Redis to avoid idle token cost. Tier 3 agents use LangGraph subgraphs for branching content pipelines.",
    },
    {
        "term_familiar": "Session / Conversation",
        "langgraph_equivalent": "thread_id in configurable config dict",
        "how_we_use": "Every graph invocation uses a thread_id. Same thread_id = same conversation/run. MemorySaver persists state per thread.",
    },
    {
        "term_familiar": "Pipeline / Workflow",
        "langgraph_equivalent": "StateGraph with nodes and edges",
        "how_we_use": "Each Tier 2 graph is a StateGraph. Nodes are async functions. Edges define execution order. Conditional edges implement branching logic.",
    },
    {
        "term_familiar": "Queue / Task",
        "langgraph_equivalent": "Redis list (RPUSH/BLPOP) feeding into ainvoke()",
        "how_we_use": "Each graph has a named Redis queue. Workers push tasks; graph runners BLPOP and call graph.ainvoke() for each item.",
    },
    {
        "term_familiar": "Golden Rule",
        "langgraph_equivalent": "AutoMem pattern with is_golden_rule=True, boosts historical layer",
        "how_we_use": "Patterns promoted to golden rules after 5+ uses with ≥90% success rate. Each golden rule adds +0.15 to historical score in confidence graph.",
    },
    {
        "term_familiar": "AutoMem / Memory Store",
        "langgraph_equivalent": "External store (complementary to checkpointer)",
        "how_we_use": "PostgreSQL + pgvector. Stores conversations, patterns, golden rules, audit log. Queried by score_historical node for pattern boosts.",
    },
]


# ---------------------------------------------------------------------------
# HTML generator
# ---------------------------------------------------------------------------

def get_admin_html(settings=None) -> str:
    """Generate the complete admin UI HTML string."""
    if settings is None:
        settings = get_settings()

    conf_weights_html = ""
    for layer in CONFIDENCE_LAYERS:
        pct = int(layer["weight"] * 100)
        scoring_html = "".join(f"<li>{s}</li>" for s in layer["scoring"])
        conf_weights_html += f"""
        <div class="layer-card">
          <div class="layer-header">
            <span class="layer-name">{layer["layer"]}</span>
            <span class="layer-weight badge-info">{pct}% weight</span>
          </div>
          <p class="layer-desc">{layer["description"]}</p>
          <ul class="layer-scoring">{scoring_html}</ul>
        </div>"""

    graphs_html = ""
    for g in GRAPHS:
        interrupt_badge = '<span class="badge badge-interrupt">⚡ interrupt()</span>' if g.get("has_interrupt") else ""
        nodes_html = ""
        for node in g["nodes"]:
            node_class = {
                "start": "node-start",
                "normal": "node-normal",
                "terminal": "node-terminal",
                "interrupt": "node-interrupt",
            }.get(node["type"], "node-normal")
            nodes_html += f"""<div class="graph-node {node_class}">
              <span class="node-name">{node["name"]}</span>
              <span class="node-desc">{node["desc"]}</span>
            </div>"""

        queue_map_html = ""
        if g.get("queue_map"):
            rows = "".join(
                f"<tr><td class='mono'>{k}</td><td class='mono badge-success-text'>{v}</td></tr>"
                for k, v in g["queue_map"].items()
            )
            queue_map_html = f"""<div class="subsection">
              <h4>Queue Routing Map</h4>
              <table class="data-table"><thead><tr><th>task_type</th><th>Redis Queue</th></tr></thead>
              <tbody>{rows}</tbody></table></div>"""

        scoring_html = ""
        if g.get("scoring"):
            rows = "".join(
                f"<tr><td>{k}</td><td class='mono small-text'>{v}</td></tr>"
                for k, v in g["scoring"].items()
            )
            scoring_html = f"""<div class="subsection">
              <h4>Platform Scoring Rules</h4>
              <table class="data-table"><thead><tr><th>Platform</th><th>Scoring Formula</th></tr></thead>
              <tbody>{rows}</tbody></table></div>"""

        graphs_html += f"""
        <div class="card" id="graph-{g['name'].lower().replace(' ','-')}">
          <div class="card-header">
            <h3>{g["name"]} {interrupt_badge}</h3>
            <span class="badge badge-info">queue: {g["queue"]}</span>
            <span class="badge badge-muted">state: {g["state"]}</span>
          </div>
          <p class="card-desc">{g["description"]}</p>
          <div class="flow-box"><code>{g["flow"]}</code></div>
          <div class="nodes-grid">{nodes_html}</div>
          {queue_map_html}
          {scoring_html}
          <div class="file-ref">📄 {g["file"]}</div>
        </div>"""

    workers_html = ""
    for w in WORKERS:
        workers_html += f"""
        <div class="card">
          <div class="card-header">
            <h3>{w["name"]}</h3>
            <span class="badge badge-warning">queue: {w["queue"]}</span>
            <span class="badge badge-error">{w["langgraph"]}</span>
          </div>
          <p class="card-desc">{w["description"]}</p>
          <table class="data-table">
            <tr><td>Idle behavior</td><td>{w["idle_behavior"]}</td></tr>
            <tr><td>Sources</td><td>{w["sources"]}</td></tr>
            <tr><td>Dispatches to</td><td>{w["dispatches_to"]}</td></tr>
          </table>
          <div class="file-ref">📄 {w["file"]}</div>
        </div>"""

    agents_html = ""
    for a in AGENTS:
        nodes_str = " → ".join(a["nodes"])
        agents_html += f"""
        <div class="card">
          <div class="card-header">
            <h3>{a["name"]}</h3>
            <span class="badge badge-success">{a["platform"]}</span>
            <span class="badge badge-info">queue: {a["queue"]}</span>
          </div>
          <p class="card-desc">{a["description"]}</p>
          <div class="flow-box"><code>{nodes_str} → END</code></div>
          <table class="data-table">
            <tr><td>Min quality</td><td>{a["min_quality"]}</td></tr>
            <tr><td>LLM calls</td><td>{a["llm_calls"]}</td></tr>
            <tr><td>Output queue</td><td class="mono">{a["output_queue"]}</td></tr>
          </table>
          <div class="file-ref">📄 {a["file"]}</div>
        </div>"""

    guardrails_html = ""
    type_colors = {"FINANCIAL": "badge-error", "QUALITY": "badge-info", "RATE_LIMIT": "badge-warning", "CONTENT": "badge-muted"}
    for gr in GUARDRAILS:
        color = type_colors.get(gr["type"], "badge-muted")
        guardrails_html += f"""<tr>
          <td><strong>{gr["name"]}</strong></td>
          <td><span class="badge {color}">{gr["type"]}</span></td>
          <td class="mono">{gr["value"]}</td>
          <td>{gr["description"]}</td>
        </tr>"""

    llms_html = ""
    for llm in LLMS:
        llms_html += f"""
        <div class="llm-card">
          <div class="llm-header">
            <span class="llm-name">{llm["name"]}</span>
            <span class="badge badge-muted">{llm["provider"]}</span>
            <span class="badge badge-success-text">💰 {llm["cost"]}</span>
            <span class="badge badge-info">⚡ {llm["speed"]}</span>
          </div>
          <div class="llm-grid">
            <div><strong>✅ Best for:</strong> {llm["best_for"]}</div>
            <div><strong>❌ Not for:</strong> {llm["not_for"]}</div>
            <div><strong>🎯 Task types:</strong> <code>{llm.get("task_types", "general")}</code></div>
            <div><strong>⚙️ Config:</strong> <code>{llm["config"]}</code></div>
          </div>
        </div>"""

    pd_html = ""
    for pd in PRIME_DIRECTIVES:
        details_html = "".join(f"<li>{d}</li>" for d in pd["details"])
        pd_html += f"""
        <div class="pd-card" style="border-left: 4px solid {pd['color']}">
          <div class="pd-header">
            <span class="pd-number" style="color:{pd['color']}">#{pd['number']}</span>
            <span class="pd-name">{pd['name']}</span>
            <span class="badge badge-muted">{pd['tier']}</span>
          </div>
          <p class="pd-desc">{pd['description']}</p>
          <div class="pd-location">
            <strong>LangGraph location:</strong> <code>{pd['langgraph_location']}</code>
          </div>
          <ul class="pd-details">{details_html}</ul>
        </div>"""

    commandments_html = ""
    for cmd in COMMANDMENTS:
        commandments_html += f"""<tr>
          <td><strong>{cmd["name"]}</strong></td>
          <td>{cmd["rule"]}</td>
          <td class="mono small-text">{cmd["langgraph_equivalent"]}</td>
        </tr>"""

    glossary_html = ""
    for g in GLOSSARY:
        glossary_html += f"""<tr>
          <td><strong>{g["term_familiar"]}</strong></td>
          <td><code>{g["langgraph_equivalent"]}</code></td>
          <td>{g["how_we_use"]}</td>
        </tr>"""

    states_html = ""
    for s in STATES:
        extends = f' <span class="badge badge-muted">extends {s.get("extends","")}</span>' if s.get("extends") else ""
        fields_html = ""
        for field_name, field_type, field_desc in s["fields"]:
            fields_html += f"""<tr>
              <td class="mono">{field_name}</td>
              <td class="mono small-text">{field_type}</td>
              <td>{field_desc}</td>
            </tr>"""
        states_html += f"""
        <div class="card">
          <div class="card-header">
            <h3>{s["name"]}{extends}</h3>
          </div>
          <table class="data-table">
            <thead><tr><th>Field</th><th>Type</th><th>Description</th></tr></thead>
            <tbody>{fields_html}</tbody>
          </table>
        </div>"""

    settings_html = f"""
        <table class="data-table">
          <thead><tr><th>Setting</th><th>Value</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td>confidence_execute</td><td class="mono badge-success-text">{settings.confidence_execute}</td><td>Minimum confidence to execute autonomously</td></tr>
            <tr><td>confidence_delegate</td><td class="mono badge-warning-text">{settings.confidence_delegate}</td><td>Minimum confidence to delegate. Below → interrupt()</td></tr>
            <tr><td>vllm_local_base_url</td><td class="mono">{settings.vllm_local_base_url}</td><td>vLLM local inference server (Mac Studio, primary)</td></tr>
            <tr><td>vllm_remote</td><td class="mono">{"configured ✅" if True else "not set"}</td><td>vLLM GPU server URL (g5.xlarge / new rig) — set via Keychain</td></tr>
            <tr><td>groq_base_url</td><td class="mono">{settings.groq_base_url}</td><td>Groq free cloud API endpoint</td></tr>
            <tr><td>postgres_url</td><td class="mono">{settings.postgres_url}</td><td>PostgreSQL connection string</td></tr>
            <tr><td>redis_url</td><td class="mono">{settings.redis_url}</td><td>Redis connection URL</td></tr>
            <tr><td>neo4j_uri</td><td class="mono">{settings.neo4j_uri}</td><td>Neo4j knowledge graph URI</td></tr>
            <tr><td>alpaca_base_url</td><td class="mono">{settings.alpaca_base_url}</td><td>Alpaca trading API endpoint</td></tr>
            <tr><td>api_port</td><td class="mono">{settings.api_port}</td><td>FastAPI server port</td></tr>
            <tr><td>app_version</td><td class="mono">{settings.app_version}</td><td>Application version</td></tr>
          </tbody>
        </table>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BILLY Admin — FiestyGoat AI</title>
<style>
  :root {{
    --bg: #080e12;
    --sidebar-bg: #0a1219;
    --card-bg: #0f1c24;
    --card-border: #1a3040;
    --text: #e2e8f0;
    --text-muted: #7a9aaa;
    --accent: #00d4c8;
    --success: #22c55e;
    --warning: #f59e0b;
    --error: #ef4444;
    --info: #00b8d4;
    --interrupt: #ec4899;
    --code-bg: #060d11;
    --node-start: #0a2535;
    --node-normal: #0a2015;
    --node-terminal: #251010;
    --node-interrupt: #1a0a22;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: var(--bg); color: var(--text); display: flex; height: 100vh; overflow: hidden; }}

  /* Sidebar */
  .sidebar {{ width: 240px; min-width: 240px; background: var(--sidebar-bg);
              border-right: 1px solid var(--card-border); display: flex;
              flex-direction: column; overflow-y: auto; }}
  .sidebar-logo {{ padding: 16px 16px 12px; border-bottom: 1px solid var(--card-border); text-align: center; }}
  .sidebar-logo img {{ width: 120px; height: auto; display: block; margin: 0 auto 6px; }}
  .sidebar-logo p {{ font-size: 10px; color: var(--text-muted); margin-top: 2px; letter-spacing: 0.05em; }}
  .nav-section {{ padding: 12px 0 4px; }}
  .nav-section-label {{ padding: 0 16px 6px; font-size: 10px; font-weight: 600;
                         color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; }}
  .nav-item {{ display: block; padding: 7px 16px; font-size: 13px; color: var(--text-muted);
               cursor: pointer; border: none; background: none; width: 100%; text-align: left;
               transition: all 0.15s; border-left: 3px solid transparent; }}
  .nav-item:hover {{ color: var(--text); background: rgba(99,102,241,0.08); }}
  .nav-item.active {{ color: var(--accent); border-left-color: var(--accent);
                      background: rgba(99,102,241,0.12); }}

  /* Main */
  .main {{ flex: 1; overflow-y: auto; padding: 32px; }}
  .section {{ display: none; }}
  .section.active {{ display: block; }}
  .section-title {{ font-size: 24px; font-weight: 700; margin-bottom: 6px; }}
  .section-subtitle {{ color: var(--text-muted); margin-bottom: 24px; font-size: 14px; }}

  /* Cards */
  .card {{ background: var(--card-bg); border: 1px solid var(--card-border);
           border-radius: 10px; padding: 20px; margin-bottom: 16px; }}
  .card-header {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }}
  .card-header h3 {{ font-size: 16px; font-weight: 600; }}
  .card-desc {{ color: var(--text-muted); font-size: 13px; margin-bottom: 14px; line-height: 1.6; }}
  .file-ref {{ font-size: 11px; color: var(--text-muted); margin-top: 12px; font-family: monospace; }}

  /* Badges */
  .badge {{ font-size: 11px; padding: 2px 8px; border-radius: 4px; font-weight: 500; white-space: nowrap; }}
  .badge-info {{ background: rgba(59,130,246,0.15); color: #60a5fa; }}
  .badge-success {{ background: rgba(34,197,94,0.15); color: #4ade80; }}
  .badge-warning {{ background: rgba(245,158,11,0.15); color: #fbbf24; }}
  .badge-error {{ background: rgba(239,68,68,0.15); color: #f87171; }}
  .badge-muted {{ background: rgba(100,116,139,0.2); color: #94a3b8; }}
  .badge-interrupt {{ background: rgba(236,72,153,0.15); color: #f472b6; }}
  .badge-success-text {{ color: #4ade80; }}
  .badge-warning-text {{ color: #fbbf24; }}

  /* Graph nodes */
  .nodes-grid {{ display: grid; gap: 8px; margin: 14px 0; }}
  .graph-node {{ padding: 10px 14px; border-radius: 6px; border: 1px solid var(--card-border); }}
  .node-start {{ background: var(--node-start); border-color: #1e4d8c; }}
  .node-normal {{ background: var(--node-normal); border-color: #1a3d1a; }}
  .node-terminal {{ background: var(--node-terminal); border-color: #3d1a1a; }}
  .node-interrupt {{ background: var(--node-interrupt); border-color: #5b1a42; }}
  .node-name {{ font-family: monospace; font-size: 13px; font-weight: 600; display: block; margin-bottom: 2px; }}
  .node-desc {{ font-size: 12px; color: var(--text-muted); }}
  .flow-box {{ background: var(--code-bg); border: 1px solid var(--card-border); border-radius: 6px;
               padding: 10px 14px; margin: 10px 0; font-size: 12px; overflow-x: auto; }}
  .flow-box code {{ color: #a5b4fc; white-space: nowrap; }}

  /* Tables */
  .data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .data-table th {{ text-align: left; padding: 8px 12px; color: var(--text-muted);
                    font-size: 11px; font-weight: 600; text-transform: uppercase;
                    border-bottom: 1px solid var(--card-border); }}
  .data-table td {{ padding: 8px 12px; border-bottom: 1px solid rgba(42,46,69,0.5); vertical-align: top; }}
  .data-table tr:last-child td {{ border-bottom: none; }}
  .data-table tr:hover td {{ background: rgba(99,102,241,0.04); }}
  .mono {{ font-family: monospace; font-size: 12px; }}
  .small-text {{ font-size: 11px; }}

  /* Prime Directives */
  .pd-card {{ background: var(--card-bg); border: 1px solid var(--card-border);
              border-radius: 10px; padding: 20px; margin-bottom: 16px; }}
  .pd-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }}
  .pd-number {{ font-size: 20px; font-weight: 700; }}
  .pd-name {{ font-size: 17px; font-weight: 700; }}
  .pd-desc {{ color: var(--text-muted); font-size: 13px; margin-bottom: 10px; line-height: 1.6; }}
  .pd-location {{ background: var(--code-bg); border-radius: 6px; padding: 8px 12px;
                  font-size: 12px; margin-bottom: 10px; }}
  .pd-details {{ padding-left: 20px; }}
  .pd-details li {{ font-size: 12px; color: var(--text-muted); margin: 3px 0; line-height: 1.5; }}

  /* Confidence layers */
  .layers-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; margin: 16px 0; }}
  .layer-card {{ background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; padding: 16px; }}
  .layer-header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }}
  .layer-name {{ font-weight: 600; font-size: 15px; }}
  .layer-desc {{ font-size: 12px; color: var(--text-muted); margin-bottom: 8px; }}
  .layer-scoring {{ padding-left: 16px; }}
  .layer-scoring li {{ font-size: 11px; color: var(--text-muted); margin: 2px 0; font-family: monospace; }}

  /* Decision thresholds viz */
  .threshold-bar {{ background: var(--card-bg); border: 1px solid var(--card-border);
                    border-radius: 8px; padding: 16px; margin: 16px 0; }}
  .bar-track {{ height: 28px; border-radius: 6px; overflow: hidden; display: flex;
                position: relative; margin: 12px 0; border: 1px solid var(--card-border); }}
  .bar-clarify {{ background: linear-gradient(90deg, #ef444430, #ef4444); width: 60%; display: flex;
                  align-items: center; justify-content: center; font-size: 11px; font-weight: 600;
                  color: white; }}
  .bar-delegate {{ background: linear-gradient(90deg, #f59e0b50, #f59e0b); width: 30%; display: flex;
                   align-items: center; justify-content: center; font-size: 11px; font-weight: 600;
                   color: white; }}
  .bar-execute {{ background: linear-gradient(90deg, #22c55e50, #22c55e); width: 10%; display: flex;
                  align-items: center; justify-content: center; font-size: 11px; font-weight: 600;
                  color: white; }}

  /* LLM cards */
  .llm-card {{ background: var(--card-bg); border: 1px solid var(--card-border);
               border-radius: 10px; padding: 16px; margin-bottom: 12px; }}
  .llm-header {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }}
  .llm-name {{ font-size: 16px; font-weight: 700; font-family: monospace; color: var(--accent); }}
  .llm-grid {{ display: grid; grid-template-columns: 1fr; gap: 6px; font-size: 12px; }}

  /* Glossary */
  .glossary-table td {{ font-size: 12px; }}

  /* Architecture diagram */
  .arch-diagram {{ background: var(--code-bg); border: 1px solid var(--card-border);
                   border-radius: 10px; padding: 20px; font-family: monospace; font-size: 12px;
                   line-height: 1.8; color: var(--text-muted); overflow-x: auto; }}
  .arch-t0 {{ color: #f472b6; font-weight: 600; }}
  .arch-t1 {{ color: #f59e0b; font-weight: 600; }}
  .arch-t2 {{ color: #6366f1; font-weight: 600; }}
  .arch-t3 {{ color: #22c55e; font-weight: 600; }}

  /* Subsection */
  .subsection {{ margin-top: 14px; }}
  .subsection h4 {{ font-size: 12px; font-weight: 600; color: var(--text-muted);
                    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; }}

  /* API reference */
  .api-endpoint {{ background: var(--code-bg); border: 1px solid var(--card-border);
                   border-radius: 6px; padding: 14px; margin: 10px 0; }}
  .api-method {{ font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 3px; margin-right: 8px; }}
  .method-post {{ background: #1a3d2d; color: #4ade80; }}
  .method-get {{ background: #1a2d4d; color: #60a5fa; }}
  .api-path {{ font-family: monospace; font-size: 13px; }}
  .api-desc {{ font-size: 12px; color: var(--text-muted); margin-top: 6px; }}
  pre {{ background: var(--code-bg); border: 1px solid var(--card-border); border-radius: 6px;
         padding: 12px; font-size: 11px; overflow-x: auto; margin: 8px 0; color: #a5b4fc; }}
</style>
</head>
<body>

<nav class="sidebar">
  <div class="sidebar-logo">
    <img src="/static/Billy FG AI logo.png" alt="BILLY — FiestyGoat AI" />
    <p>Autonomous Revenue System — v{settings.app_version}</p>
  </div>
  <div class="nav-section">
    <div class="nav-section-label">Get Started</div>
    <button class="nav-item active" onclick="showSection('help')">📖 Help &amp; Concepts</button>
    <button class="nav-item" onclick="showSection('about')">🏠 About BILLY</button>
    <button class="nav-item" onclick="showSection('services')">🟢 Services &amp; Status</button>
    <button class="nav-item" onclick="showSection('directives')">🎯 Prime Directives</button>
    <button class="nav-item" onclick="showSection('architecture')">🏗️ Architecture</button>
  </div>
  <div class="nav-section">
    <div class="nav-section-label">LangGraph Components</div>
    <button class="nav-item" onclick="showSection('workers')">⚡ Tier 1 Workers</button>
    <button class="nav-item" onclick="showSection('graphs')">🔷 Tier 2 Graphs</button>
    <button class="nav-item" onclick="showSection('agents')">🤖 Tier 3 Agents</button>
  </div>
  <div class="nav-section">
    <div class="nav-section-label">Rules &amp; Intelligence</div>
    <button class="nav-item" onclick="showSection('guardrails')">🛡️ Guardrails</button>
    <button class="nav-item" onclick="showSection('confidence')">📊 Confidence Gate</button>
    <button class="nav-item" onclick="showSection('llms')">🧠 LLM Roster</button>
  </div>
  <div class="nav-section">
    <div class="nav-section-label">Reference</div>
    <button class="nav-item" onclick="showSection('states')">📋 State Schemas</button>
    <button class="nav-item" onclick="showSection('api')">🔌 API Reference</button>
    <button class="nav-item" onclick="showSection('settings')">⚙️ Settings</button>
  </div>
</nav>

<main class="main">

  <!-- ========== ABOUT BILLY ========== -->
  <section id="section-about" class="section">
    <h2 class="section-title">🏠 About BILLY</h2>
    <p class="section-subtitle">System context, architecture overview, and the Operating Mandate — reference this every session.</p>

    <div class="card" style="border-left: 4px solid #f59e0b; margin-bottom: 20px;">
      <h3 style="color: #f59e0b; margin-bottom: 12px;">⚡ Operating Mandate</h3>
      <p style="font-style: italic; font-size: 14px; line-height: 1.7; color: var(--text);">
        "As you continue to work, these are complicated tasks. Make sure you're using all tools, reasoning and logic available to you and save as much context as necessary to continue with <strong>90%+ confidence</strong> in all actions. If you can't get to 90%+ confidence ask questions and do research. Use <strong>Critical Thinking, Task Tracking and Sequential Thinking</strong> as needed. <strong>Accuracy and Quality over speed and time.</strong> Do as much work <strong>in parallel</strong> as you can. If something is broken always <strong>fix it or prompt for help, never skip.</strong>"
      </p>
      <p style="font-size: 12px; color: var(--text-muted); margin-top: 8px;">
        This mandate is always active. Applied at every session start, after every context compaction, and embedded in BILLY's Prime Directives.
      </p>
    </div>

    <div class="card" style="margin-bottom: 20px;">
      <h3 style="margin-bottom: 16px;">System Overview</h3>
      <table class="data-table">
        <thead><tr><th>System</th><th>What It Is</th><th>Location</th></tr></thead>
        <tbody>
          <tr>
            <td><strong>BILLY AI</strong></td>
            <td>Autonomous revenue system — LangGraph-native replacement for all of jarvis_v1. Named after Billy Goat (FiestyGoat AI LLC).</td>
            <td class="mono">/Users/TehFiestyGoat/Development/jarvis_v2/<br>github.com/grimreaper0/jarvis_v2</td>
          </tr>
          <tr>
            <td><strong>The Codex</strong></td>
            <td>Project management system — quests (QUE-XXX), ideas, inklings. Originally a work project, ported to personal use. Docker container.</td>
            <td class="mono">/Users/TehFiestyGoat/Downloads/solutions-catalog-main/<br>github.com/grimreaper0/the_codex</td>
          </tr>
          <tr>
            <td><strong>FiestyGoat AI LLC</strong></td>
            <td>CA Multi-Member LLC (EIN: 41-4302206). Natalie Kaiser 51% CEO, Ernest Kaiser 49% CTO. Purpose: AI Education for Underserved Communities.</td>
            <td class="mono">Entity No: B20260063405<br>Bank: Mercury (approved 02/17/2026)</td>
          </tr>
          <tr>
            <td><strong>jarvis_v1</strong></td>
            <td>Legacy system — still running via cron/supervisor. Being replaced by BILLY. ChromaDB + Ollama + Streamlit dashboard at port 8502.</td>
            <td class="mono">/Users/TehFiestyGoat/Development/personal-agent-hub/<br>(OpenClaw dashboard being deleted)</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="card" style="margin-bottom: 20px;">
      <h3 style="margin-bottom: 16px;">Port Map</h3>
      <table class="data-table">
        <thead><tr><th>Service</th><th>Port</th><th>Notes</th></tr></thead>
        <tbody>
          <tr><td><strong>BILLY API (jarvis_v2)</strong></td><td class="mono badge-success-text">8506</td><td>FastAPI — FIXED, never change</td></tr>
          <tr><td>mlx_lm.server (local LLM)</td><td class="mono">8001</td><td>Apple Metal inference. mlx-community/Qwen3-4B-4bit</td></tr>
          <tr><td>The Codex — Backend</td><td class="mono">8000</td><td>FastAPI (Docker), trailing slash required on API calls</td></tr>
          <tr><td>The Codex — Frontend</td><td class="mono">3001</td><td>nginx/React — <a href="http://localhost:3001" style="color:var(--accent)">localhost:3001</a></td></tr>
          <tr><td>jarvis_v1 Dashboard</td><td class="mono">8502</td><td>Streamlit</td></tr>
          <tr><td>Neo4j</td><td class="mono">7687</td><td>bolt:// — FiestyGoatNeo4j2026!</td></tr>
          <tr><td>Redis</td><td class="mono">6379</td><td>Coordination layer</td></tr>
          <tr><td>PostgreSQL</td><td class="mono">5432</td><td>personal_agent_hub (shared v1+v2)</td></tr>
        </tbody>
      </table>
    </div>

    <div class="card" style="margin-bottom: 20px;">
      <h3 style="margin-bottom: 16px;">LLM Inference — Current Setup (Mac Studio M2 Max)</h3>
      <table class="data-table">
        <thead><tr><th>Backend</th><th>Command</th><th>Notes</th></tr></thead>
        <tbody>
          <tr>
            <td><strong>mlx_lm.server</strong> (local primary)</td>
            <td class="mono small-text">venv/bin/python3.13 -m mlx_lm server --model mlx-community/Qwen3-4B-4bit --port 8001 --host 0.0.0.0</td>
            <td>Apple Metal native. Port 8001. ~1.9s latency. vLLM has NO MPS backend — mlx_lm is the correct path on Mac.</td>
          </tr>
          <tr>
            <td><strong>New Rig</strong> (arriving ~2 weeks)</td>
            <td class="mono small-text">VLLM_USE_V1=0 vllm serve Qwen/Qwen3-4B --port 8001 --tensor-parallel-size 2</td>
            <td>Intel Core Ultra 9 + 2x RTX 5060 Ti 16GB = 32GB VRAM. Real vLLM on CUDA.</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="card">
      <h3 style="margin-bottom: 16px;">The Codex — Active Quests</h3>
      <table class="data-table">
        <thead><tr><th>ID</th><th>Quest</th><th>Status</th></tr></thead>
        <tbody>
          <tr><td class="mono">QUE-001</td><td>DOL VETS Grant Application</td><td>Active</td></tr>
          <tr><td class="mono">QUE-002</td><td>AARP Community Challenge</td><td>Active</td></tr>
          <tr><td class="mono">QUE-003</td><td>Digital Promise K-12 AI Infrastructure ($50K-$250K, Mar 8 deadline)</td><td>Active — FULL CONTEXT SAVED</td></tr>
          <tr><td class="mono">QUE-004</td><td>WOSB Certification (free, Natalie qualifies)</td><td>Active</td></tr>
          <tr><td class="mono">QUE-005</td><td>MBE/NMSDC Certification</td><td>Active</td></tr>
          <tr><td class="mono">QUE-006</td><td>Mercury Bank Sub-Account Setup</td><td>Active</td></tr>
          <tr><td class="mono">QUE-007</td><td>FiestyGoat AI Architecture Strategy</td><td>Active</td></tr>
        </tbody>
      </table>
      <p style="font-size:12px;color:var(--text-muted);margin-top:8px;">
        API: <span class="mono">GET http://localhost:8000/api/v1/solutions/</span> (requires Codex Docker running — port 8000, trailing slash required)
      </p>
    </div>
  </section>

  <!-- ========== SERVICES & STATUS ========== -->
  <section id="section-services" class="section">
    <h2 class="section-title">🟢 Services &amp; Status</h2>
    <p class="section-subtitle">All dashboards, inference servers, and infrastructure services. Click any link to open.</p>

    <div class="card" style="margin-bottom: 20px;">
      <h3 style="margin-bottom: 16px;">Dashboards &amp; UIs</h3>
      <table class="data-table">
        <thead><tr><th>Service</th><th>URL</th><th>Port</th><th>Type</th><th id="dash-status-header">Status</th></tr></thead>
        <tbody>
          <tr>
            <td><strong>BILLY Admin</strong></td>
            <td><a href="http://localhost:8506/admin" style="color:var(--accent)" target="_blank">localhost:8506/admin</a></td>
            <td class="mono">8506</td>
            <td>FastAPI</td>
            <td class="svc-status" data-url="http://localhost:8506/health">...</td>
          </tr>
          <tr>
            <td><strong>The Codex</strong></td>
            <td><a href="http://localhost:3001" style="color:var(--accent)" target="_blank">localhost:3001</a></td>
            <td class="mono">3001</td>
            <td>React + nginx (Docker)</td>
            <td class="svc-status" data-url="http://localhost:3001">...</td>
          </tr>
          <tr>
            <td><strong>jarvis_v1 Dashboard</strong></td>
            <td><a href="http://localhost:8502" style="color:var(--accent)" target="_blank">localhost:8502</a></td>
            <td class="mono">8502</td>
            <td>Streamlit</td>
            <td class="svc-status" data-url="http://localhost:8502">...</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="card" style="margin-bottom: 20px;">
      <h3 style="margin-bottom: 16px;">LLM Inference</h3>
      <table class="data-table">
        <thead><tr><th>Backend</th><th>URL</th><th>Port</th><th>Model</th><th>Status</th></tr></thead>
        <tbody>
          <tr>
            <td><strong>mlx_lm.server</strong> (local primary)</td>
            <td class="mono">localhost:8001/v1</td>
            <td class="mono">8001</td>
            <td>mlx-community/Qwen3-4B-4bit</td>
            <td class="svc-status" data-url="http://localhost:8001/v1/models">...</td>
          </tr>
          <tr>
            <td><strong>Groq</strong> (cloud free)</td>
            <td class="mono">api.groq.com</td>
            <td class="mono">443</td>
            <td>qwen/qwen3-32b</td>
            <td id="groq-key-status">...</td>
          </tr>
          <tr>
            <td><strong>OpenRouter</strong> (cloud free)</td>
            <td class="mono">openrouter.ai</td>
            <td class="mono">443</td>
            <td>qwen/qwen3-coder-480b:free</td>
            <td id="openrouter-key-status">...</td>
          </tr>
          <tr>
            <td><strong>Grok</strong> (xAI)</td>
            <td class="mono">api.x.ai</td>
            <td class="mono">443</td>
            <td>grok-4-fast-reasoning</td>
            <td id="grok-key-status">...</td>
          </tr>
          <tr>
            <td><strong>DeepSeek</strong> (cloud cheap)</td>
            <td class="mono">api.deepseek.com</td>
            <td class="mono">443</td>
            <td>deepseek-reasoner / deepseek-chat</td>
            <td id="deepseek-key-status">...</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="card" style="margin-bottom: 20px;">
      <h3 style="margin-bottom: 16px;">Infrastructure</h3>
      <table class="data-table">
        <thead><tr><th>Service</th><th>URL</th><th>Port</th><th>Notes</th><th>Status</th></tr></thead>
        <tbody>
          <tr>
            <td><strong>PostgreSQL</strong></td>
            <td class="mono">localhost:5432</td>
            <td class="mono">5432</td>
            <td>personal_agent_hub (shared v1 + v2)</td>
            <td class="svc-status" data-url="http://localhost:8506/health">...</td>
          </tr>
          <tr>
            <td><strong>Redis</strong></td>
            <td class="mono">localhost:6379</td>
            <td class="mono">6379</td>
            <td>Coordination layer + task queues</td>
            <td class="svc-status" data-url="http://localhost:8506/health">...</td>
          </tr>
          <tr>
            <td><strong>Neo4j</strong></td>
            <td class="mono">bolt://localhost:7687</td>
            <td class="mono">7687</td>
            <td>Knowledge graph</td>
            <td class="svc-status" data-url="http://localhost:7474">...</td>
          </tr>
          <tr>
            <td><strong>The Codex Backend</strong></td>
            <td class="mono">localhost:8000</td>
            <td class="mono">8000</td>
            <td>FastAPI (Docker) — solutions-catalog</td>
            <td class="svc-status" data-url="http://localhost:8000/health">...</td>
          </tr>
        </tbody>
      </table>
    </div>

    <p style="font-size:12px;color:var(--text-muted);">Status checks run client-side on page load. Green = reachable, Red = down, Yellow = API key configured but not health-checked.</p>
  </section>

  <!-- ========== HELP ========== -->
  <section id="section-help" class="section active">
    <h2 class="section-title">📖 Help &amp; Concepts</h2>
    <p class="section-subtitle">How LangGraph terminology maps to concepts you already know.</p>

    <div class="card">
      <div class="card-header"><h3>What is LangGraph?</h3></div>
      <p class="card-desc">
        LangGraph is a framework for building stateful, multi-step AI workflows as directed graphs.
        You define <strong>nodes</strong> (Python functions that do work) and <strong>edges</strong> (connections that define execution order).
        State flows through nodes, gets updated at each step, and is checkpointed so runs can be paused and resumed.
        BILLY uses LangGraph for all <em>decision workflows</em> (Tier 2) and <em>specialist content agents</em> (Tier 3),
        but deliberately does NOT use it for always-on background workers (Tier 1) to avoid idle token costs.
      </p>
    </div>

    <div class="card">
      <div class="card-header"><h3>Glossary: Terms You Know → LangGraph Equivalents</h3></div>
      <table class="data-table glossary-table">
        <thead><tr><th>Term You Know</th><th>LangGraph Equivalent</th><th>How We Use It</th></tr></thead>
        <tbody>{glossary_html}</tbody>
      </table>
    </div>

    <div class="card">
      <div class="card-header"><h3>The Three Tiers</h3></div>
      <table class="data-table">
        <thead><tr><th>Tier</th><th>What it is</th><th>Uses LangGraph?</th><th>Examples</th></tr></thead>
        <tbody>
          <tr><td><strong>Tier 0</strong></td><td>JARVIS Controller + orchestration</td><td>Via interrupt()</td><td>JARVIS Controller, this Admin UI</td></tr>
          <tr><td><strong>Tier 1</strong></td><td>Always-on ContinuousWorkers (Redis poll loops)</td><td class="badge-error" style="color:#f87171">NO — by design</td><td>LearningWorker, ResearchWorker, TradingWorker, HealthWorker</td></tr>
          <tr><td><strong>Tier 2</strong></td><td>Decision workflows (StateGraph)</td><td class="badge-success" style="color:#4ade80">YES — core use case</td><td>Revenue Graph, Trading Graph, Content Gate, Confidence Graph</td></tr>
          <tr><td><strong>Tier 3</strong></td><td>Specialist content subgraphs</td><td class="badge-success" style="color:#4ade80">YES — subgraphs</td><td>Instagram Agent, YouTube Agent, Trading Agent, etc.</td></tr>
        </tbody>
      </table>
    </div>

    <div class="card">
      <div class="card-header"><h3>Key LangGraph Concepts</h3></div>
      <table class="data-table">
        <thead><tr><th>Concept</th><th>What it does</th></tr></thead>
        <tbody>
          <tr><td><code>StateGraph</code></td><td>The main graph class. You add nodes and edges, then compile() it.</td></tr>
          <tr><td><code>TypedDict state</code></td><td>The data structure that flows through all nodes. Each node reads it and returns a dict of updates to merge back in.</td></tr>
          <tr><td><code>add_messages reducer</code></td><td>Special reducer for the messages field — appends instead of overwrites. Enables conversation history.</td></tr>
          <tr><td><code>MemorySaver</code></td><td>In-memory checkpointer. Saves state per thread_id. Allows pause/resume. Resets on server restart.</td></tr>
          <tr><td><code>thread_id</code></td><td>Unique run identifier. Same thread_id = same conversation. UUID for each new run.</td></tr>
          <tr><td><code>interrupt(value)</code></td><td>Pauses graph at that exact point, saves state to checkpointer, returns value to caller. Resume with Command(resume=...).</td></tr>
          <tr><td><code>Command(resume=value)</code></td><td>Resumes a frozen graph. The value becomes the return value of the interrupted interrupt() call.</td></tr>
          <tr><td><code>conditional_edges</code></td><td>Routing function reads state and returns the name of the next node. Enables branching (execute vs delegate vs skip).</td></tr>
          <tr><td><code>END</code></td><td>Special terminal node. Graph execution stops when a node routes here.</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <!-- ========== PRIME DIRECTIVES ========== -->
  <section id="section-directives" class="section">
    <h2 class="section-title">🎯 Prime Directives &amp; Commandments</h2>
    <p class="section-subtitle">The operating principles of FiestyGoat AI — and exactly where each lives in LangGraph.</p>
    {pd_html}
    <div class="card">
      <div class="card-header"><h3>Commandments</h3></div>
      <table class="data-table">
        <thead><tr><th>Commandment</th><th>Rule</th><th>LangGraph Equivalent</th></tr></thead>
        <tbody>{commandments_html}</tbody>
      </table>
    </div>
  </section>

  <!-- ========== ARCHITECTURE ========== -->
  <section id="section-architecture" class="section">
    <h2 class="section-title">🏗️ System Architecture</h2>
    <p class="section-subtitle">Three-tier architecture: Redis queues connect the tiers. LangGraph lives in Tier 2 and 3.</p>
    <div class="arch-diagram">
<span class="arch-t0">TIER 0 — Orchestration</span>
  JARVIS Controller → interrupt() for human-in-the-loop
  Admin UI (this page) → read-only system overview at /admin
  API Server (FastAPI @ port 8506) → /confidence/evaluate, /confidence/resume, /revenue/evaluate...
         │
         │  Redis Queues
         ▼
<span class="arch-t1">TIER 1 — ContinuousWorkers (NO LangGraph — Redis poll loops)</span>
  LearningWorker    queue: learning_scrape      idle: scrape 28 RSS feeds every hour
  ResearchWorker    queue: research_request     idle: scrape 19 dark-hole sources every hour
  TradingWorker     queue: trading_execution    idle: performance analysis + strategy optimization
  HealthWorker      queue: health_check         idle: system health check every 5 minutes
         │
         │  push_task_to_graph() → Redis queues
         ▼
<span class="arch-t2">TIER 2 — LangGraph StateGraphs (decision workflows)</span>
  Revenue Graph     queue: revenue_opportunity  routes to Tier 3 agents
  Trading Graph     queue: trading_signal       validates + sizes + submits to Tier 1
  Content Gate      queue: content_gate         scores quality, routes approve/reject
  Confidence Graph  queue: confidence_eval      4-layer scoring + interrupt() for &lt;0.60
         │
         │  Redis queues (instagram_task, youtube_task, etc.)
         ▼
<span class="arch-t3">TIER 3 — LangGraph Specialist Agents (content subgraphs)</span>
  Instagram Agent   queue: instagram_task       research → caption → hashtags → quality_gate
  YouTube Agent     queue: youtube_task         research → title → description → quality_gate
  Newsletter Agent  queue: newsletter_task      research → sections → affiliate → quality_gate
  SEO Blog Agent    queue: seo_blog_task        keywords → outline → write → SEO → quality_gate
  TikTok Agent      queue: tiktok_task          trends → hook → script → sounds → quality_gate
  Twitter Agent     queue: twitter_task         research → thread → validate → quality_gate
  Trading Agent     queue: trading_analysis     screen → analyze → score → submit to Tier 2
    </div>

    <div class="card" style="margin-bottom: 20px;">
      <div class="card-header"><h3>Visual Architecture</h3></div>
      <div id="mermaid-arch"></div>
      <script>
        document.addEventListener('DOMContentLoaded', function() {{
          const diagram = `graph TB
    subgraph T0["TIER 0 - Orchestration"]
        JARVIS["JARVIS Controller"]
        ADMIN["Admin UI :8506"]
        API["FastAPI API :8506"]
    end

    subgraph T1["TIER 1 - ContinuousWorkers"]
        LW["LearningWorker"]
        RW["ResearchWorker"]
        TW["TradingWorker"]
        HW["HealthWorker"]
    end

    subgraph T2["TIER 2 - LangGraph StateGraphs"]
        RG["Revenue Graph"]
        TG["Trading Graph"]
        CG["Content Gate"]
        CONF["Confidence Graph"]
    end

    subgraph T3["TIER 3 - Specialist Agents"]
        IG["Instagram"]
        YT["YouTube"]
        NL["Newsletter"]
        SEO["SEO Blog"]
        TK["TikTok"]
        TW2["Twitter"]
        TA["Trading Analyst"]
    end

    subgraph INFRA["Infrastructure"]
        PG["PostgreSQL :5432"]
        REDIS["Redis :6379"]
        NEO["Neo4j :7687"]
        MLX["mlx_lm :8001"]
        CLOUD["Cloud LLMs"]
    end

    T0 -->|"Redis Queues"| T1
    T1 -->|"push_task_to_graph"| T2
    T2 -->|"Redis task queues"| T3

    T1 --> MLX
    T2 --> MLX
    T3 --> MLX
    T1 -.->|"fallback"| CLOUD
    T2 -.->|"fallback"| CLOUD
    T3 -.->|"fallback"| CLOUD
    T1 --> PG
    T2 --> PG
    T3 --> PG
    T0 --> REDIS
    T1 --> REDIS
    T2 --> REDIS
    T2 --> NEO
    T3 --> NEO

    RG --> IG
    RG --> YT
    RG --> NL
    RG --> SEO
    RG --> TK
    RG --> TW2
    TG --> TA
    CG --> T3

    classDef tier0 fill:#831843,stroke:#f472b6,color:#fce7f3
    classDef tier1 fill:#78350f,stroke:#f59e0b,color:#fef3c7
    classDef tier2 fill:#312e81,stroke:#6366f1,color:#e0e7ff
    classDef tier3 fill:#14532d,stroke:#22c55e,color:#dcfce7
    classDef infra fill:#1e293b,stroke:#64748b,color:#e2e8f0

    class JARVIS,ADMIN,API tier0
    class LW,RW,TW,HW tier1
    class RG,TG,CG,CONF tier2
    class IG,YT,NL,SEO,TK,TW2,TA tier3
    class PG,REDIS,NEO,MLX,CLOUD infra`;

          mermaid.render('mermaid-svg', diagram).then(function(result) {{
            document.getElementById('mermaid-arch').innerHTML = result.svg;
          }}).catch(function(err) {{
            document.getElementById('mermaid-arch').innerHTML = '<pre style="color:#ef4444;">Mermaid error: ' + err.message + '</pre>';
          }});
        }});
      </script>
    </div>

    <div class="card">
      <div class="card-header"><h3>Why Tier 1 is NOT LangGraph</h3></div>
      <p class="card-desc">
        LangGraph supervisor loops call the LLM on every iteration to decide "what next?" — even when there's
        nothing to do. Our default LLM (mlx_lm / Qwen3-4B-4bit) is free, so this isn't a dollar cost problem.
        It's a <strong>GPU contention</strong> problem: 4 idle workers constantly hitting the LLM steals GPU
        bandwidth from Tier 2/3 agents doing actual productive work (generating content, analyzing trades).
        Each supervisor cycle burns ~2-3 seconds of GPU time for zero value. Instead, Tier 1 uses raw Python +
        Redis: <code>BLPOP</code> blocks with zero CPU/GPU until work arrives. When no work arrives for 1 hour,
        workers run their idle_loop() (scraping sources) then block again. Individual workers can still use any
        LLM backend for their actual work — the architecture just doesn't waste compute on orchestration overhead.
      </p>
    </div>
  </section>

  <!-- ========== GRAPHS ========== -->
  <section id="section-graphs" class="section">
    <h2 class="section-title">🔷 Tier 2 Graphs</h2>
    <p class="section-subtitle">LangGraph StateGraphs that handle decision workflows. Each runs as a supervisord process consuming from Redis.</p>
    {graphs_html}
  </section>

  <!-- ========== WORKERS ========== -->
  <section id="section-workers" class="section">
    <h2 class="section-title">⚡ Tier 1 Workers</h2>
    <p class="section-subtitle">ContinuousWorkers — always-on Python processes. Deliberately NOT LangGraph. Redis poll loops with supervisord auto-restart.</p>
    {workers_html}
  </section>

  <!-- ========== AGENTS ========== -->
  <section id="section-agents" class="section">
    <h2 class="section-title">🤖 Tier 3 Agents</h2>
    <p class="section-subtitle">Specialist LangGraph subgraphs for content creation. Each is a compiled StateGraph consumed by a Redis queue.</p>
    {agents_html}
  </section>

  <!-- ========== GUARDRAILS ========== -->
  <section id="section-guardrails" class="section">
    <h2 class="section-title">🛡️ Guardrails</h2>
    <p class="section-subtitle">Safety limits enforced in graph nodes before any financial or content action. Hard limits — not suggestions.</p>
    <div class="card">
      <table class="data-table">
        <thead><tr><th>Rule Name</th><th>Type</th><th>Value</th><th>Description</th></tr></thead>
        <tbody>{guardrails_html}</tbody>
      </table>
    </div>
    <div class="card">
      <div class="card-header"><h3>Where Guardrails Live in LangGraph</h3></div>
      <table class="data-table">
        <thead><tr><th>Guardrail</th><th>Graph</th><th>Node</th><th>Effect if Triggered</th></tr></thead>
        <tbody>
          <tr><td>max_trade_usd, circuit_breaker</td><td>Trading Graph</td><td>check_guardrails</td><td>risk_approved=False → reject_order node → END</td></tr>
          <tr><td>min_content_quality, rate_limits</td><td>Content Gate</td><td>check_guidelines</td><td>approved=False → reject_content node → {"{platform}_regenerate"} queue</td></tr>
          <tr><td>confidence thresholds</td><td>Confidence Graph</td><td>finalize → routing</td><td>&lt;0.60 → clarify_node → interrupt() → operator decision</td></tr>
          <tr><td>blocked_keywords</td><td>Content Gate</td><td>check_guidelines</td><td>approved=False with rejection_reason</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <!-- ========== CONFIDENCE ========== -->
  <section id="section-confidence" class="section">
    <h2 class="section-title">📊 Confidence Gate</h2>
    <p class="section-subtitle">4-layer weighted scoring system. Each layer is a separate async LangGraph node with its own checkpoint.</p>

    <div class="threshold-bar card">
      <h3 style="margin-bottom:12px">Decision Thresholds</h3>
      <div class="bar-track">
        <div class="bar-clarify">0.0 – 0.59 → CLARIFY (interrupt)</div>
        <div class="bar-delegate">0.60 – 0.89 → DELEGATE</div>
        <div class="bar-execute">≥0.90 → EXECUTE</div>
      </div>
      <p style="font-size:12px; color: var(--text-muted); margin-top:8px">
        <strong style="color:#f87171">CLARIFY</strong> triggers interrupt() — graph pauses, operator is asked what to do via /confidence/resume.<br>
        <strong style="color:#fbbf24">DELEGATE</strong> pushes to human_review queue with full context for async review.<br>
        <strong style="color:#4ade80">EXECUTE</strong> routes directly to the appropriate Tier 3 agent queue, no human needed.
      </p>
    </div>

    <div class="layers-grid">{conf_weights_html}</div>

    <div class="card">
      <div class="card-header"><h3>Formula</h3></div>
      <div class="flow-box">
        <code>final_score = (base × 0.25) + (validation × 0.25) + (historical × 0.30) + (reflexive × 0.20)</code>
      </div>
      <p style="font-size:12px; color: var(--text-muted); margin-top:8px">
        Historical has the highest weight (0.30) because past performance is the strongest predictor of future success.
        Reflexive layer provides a coherence check — if the first 3 layers wildly disagree, the final score is penalised.
      </p>
    </div>

    <div class="card">
      <div class="card-header"><h3>Human-in-the-Loop: interrupt() Flow</h3></div>
      <pre>
# Step 1: Evaluate confidence
POST /confidence/evaluate
{{"task_type": "trade_stock", "description": "NVDA short squeeze signal", "context": {{}}}}

# Response if confidence &lt; 0.60:
{{"status": "waiting_for_input", "thread_id": "abc-123",
  "interrupt": {{"question": "...", "options": ["execute","delegate","skip","abort"],
                 "context": {{"confidence": 0.45, "layers": {{...}}}}}}}}

# Step 2: Resume with your decision
POST /confidence/resume
{{"thread_id": "abc-123", "user_choice": "delegate"}}

# Response:
{{"status": "complete", "decision": "delegate", "user_decision": "delegate", "final_score": 0.45}}</pre>
    </div>
  </section>

  <!-- ========== LLMS ========== -->
  <section id="section-llms" class="section">
    <h2 class="section-title">🧠 LLM Roster</h2>
    <p class="section-subtitle">All available language models, their strengths, costs, and how to invoke them. Free-first principle: Ollama always tried before API keys.</p>
    {llms_html}
    <div class="card">
      <div class="card-header"><h3>Routing Priority (LLMRouter)</h3></div>
      <p class="card-desc">
        Default calls: <code>qwen2.5:0.5b</code> → <code>grok-4-fast-reasoning</code> → <code>claude-sonnet-4-6</code><br>
        Reasoning calls (<code>LLMRequest(reasoning=True)</code>): <code>deepseek-r1:7b</code> → <code>grok-4-fast-reasoning</code> → <code>claude-sonnet-4-6</code><br>
        Router caches backend availability for 5 minutes (Ollama probe, keychain key presence check).
      </p>
      <pre>
# Standard call (qwen2.5:0.5b, $0)
router = LLMRouter()
resp = await router.complete(LLMRequest(prompt="Write a caption for..."))

# Reasoning call (deepseek-r1:7b, $0)
resp = await router.reason("Should we trade NVDA right now?", context={{"price": 900, "volume": 1.2e7}})
print(resp["reasoning"])   # chain-of-thought
print(resp["conclusion"])  # final answer

# Force Claude (cost: $)
resp = await router.complete(LLMRequest(prompt="...", backend=LLMBackend.CLAUDE))

# LangChain compatible (for LangGraph nodes)
llm = router.as_langchain_llm()   # returns OllamaLLM</pre>
    </div>
  </section>

  <!-- ========== STATES ========== -->
  <section id="section-states" class="section">
    <h2 class="section-title">📋 State Schemas</h2>
    <p class="section-subtitle">TypedDict definitions for all LangGraph graphs. State is the data contract flowing through every node.</p>
    {states_html}
  </section>

  <!-- ========== API REFERENCE ========== -->
  <section id="section-api" class="section">
    <h2 class="section-title">🔌 API Reference</h2>
    <p class="section-subtitle">FastAPI endpoints at port {settings.api_port}. All graphs accessible via REST.</p>

    <div class="card">
      <div class="card-header"><h3>Core Endpoints</h3></div>
      <div class="api-endpoint">
        <span class="api-method method-get">GET</span><span class="api-path">/health</span>
        <div class="api-desc">Health check. Returns service name and version.</div>
      </div>
      <div class="api-endpoint">
        <span class="api-method method-get">GET</span><span class="api-path">/admin</span>
        <div class="api-desc">This page.</div>
      </div>
      <div class="api-endpoint">
        <span class="api-method method-get">GET</span><span class="api-path">/admin/data</span>
        <div class="api-desc">All config as JSON (thresholds, LLM models, guardrail values).</div>
      </div>
      <div class="api-endpoint">
        <span class="api-method method-get">GET</span><span class="api-path">/workers/status</span>
        <div class="api-desc">Redis queue depths for all worker queues.</div>
      </div>
    </div>

    <div class="card">
      <div class="card-header"><h3>Confidence Graph (Human-in-the-Loop)</h3></div>
      <div class="api-endpoint">
        <span class="api-method method-post">POST</span><span class="api-path">/confidence/evaluate</span>
        <div class="api-desc">Run 4-layer confidence scoring. If confidence &lt; 0.60, returns waiting_for_input status with interrupt value.</div>
        <pre>Body: {{"task_type": "trade_stock", "description": "...", "context": {{}}, "thread_id": null}}</pre>
      </div>
      <div class="api-endpoint">
        <span class="api-method method-post">POST</span><span class="api-path">/confidence/resume</span>
        <div class="api-desc">Resume an interrupted confidence evaluation. Pass same thread_id from /evaluate response.</div>
        <pre>Body: {{"thread_id": "uuid", "user_choice": "execute | delegate | skip | abort"}}</pre>
      </div>
    </div>

    <div class="card">
      <div class="card-header"><h3>Revenue &amp; Trading Graphs</h3></div>
      <div class="api-endpoint">
        <span class="api-method method-post">POST</span><span class="api-path">/revenue/evaluate</span>
        <div class="api-desc">Evaluate a revenue opportunity through the full pipeline.</div>
        <pre>Body: {{"name": "AI Tools Reel", "estimated_value": 50.0, "platform": "instagram", "description": "..."}}</pre>
      </div>
      <div class="api-endpoint">
        <span class="api-method method-post">POST</span><span class="api-path">/content/gate</span>
        <div class="api-desc">Run content through the quality gate. Returns approved=true/false and quality_score.</div>
        <pre>Body: {{"topic": "...", "platform": "instagram", "hashtags": [], "media_url": ""}}</pre>
      </div>
      <div class="api-endpoint">
        <span class="api-method method-post">POST</span><span class="api-path">/trading/signal</span>
        <div class="api-desc">Submit a trading signal through validation + guardrails + Kelly sizing.</div>
        <pre>Body: {{"symbol": "NVDA", "side": "buy", "price": 900.0, "qty": 1, "strategy": "vwap_mean_reversion"}}</pre>
      </div>
    </div>
  </section>

  <!-- ========== SETTINGS ========== -->
  <section id="section-settings" class="section">
    <h2 class="section-title">⚙️ Settings</h2>
    <p class="section-subtitle">Current configuration from config/settings.py (loaded from .env at startup).</p>
    <div class="card">{settings_html}</div>
  </section>

</main>

<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<script>
  mermaid.initialize({{ startOnLoad: true, theme: 'dark', themeVariables: {{
    primaryColor: '#6366f1', primaryTextColor: '#e2e8f0',
    lineColor: '#64748b', background: '#0f172a',
    mainBkg: '#1e293b', nodeBorder: '#6366f1'
  }} }});

  function showSection(name) {{
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
    document.getElementById('section-' + name).classList.add('active');
    event.currentTarget.classList.add('active');
  }}

  // Live health checks for Services panel
  async function checkServices() {{
    document.querySelectorAll('.svc-status').forEach(async (td) => {{
      const url = td.getAttribute('data-url');
      if (!url) return;
      try {{
        const ctrl = new AbortController();
        setTimeout(() => ctrl.abort(), 3000);
        const resp = await fetch(url, {{ mode: 'no-cors', signal: ctrl.signal }});
        td.textContent = '🟢 UP';
        td.style.color = '#22c55e';
      }} catch (e) {{
        td.textContent = '🔴 DOWN';
        td.style.color = '#ef4444';
      }}
    }});

    // Cloud API key status — hit BILLY's own /health endpoint which knows key state
    try {{
      const resp = await fetch('/api/keys/status');
      if (resp.ok) {{
        const data = await resp.json();
        ['groq', 'openrouter', 'grok', 'deepseek'].forEach(k => {{
          const el = document.getElementById(k + '-key-status');
          if (el && data[k]) {{
            el.textContent = '🟡 KEY SET';
            el.style.color = '#f59e0b';
          }} else if (el) {{
            el.textContent = '⚪ NO KEY';
            el.style.color = '#64748b';
          }}
        }});
      }}
    }} catch (e) {{
      // key status endpoint not available yet — show unknown
      ['groq', 'openrouter', 'grok', 'deepseek'].forEach(k => {{
        const el = document.getElementById(k + '-key-status');
        if (el) {{ el.textContent = '⚪ UNKNOWN'; el.style.color = '#64748b'; }}
      }});
    }}
  }}

  // Run on load
  checkServices();
</script>
</body>
</html>"""
