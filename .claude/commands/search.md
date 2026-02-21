# /search - Natural Language Query Across Memory

**Purpose**: Search past work, patterns, decisions, notes, and bot history using natural language.

**Description**: Enables natural language queries across AutoMem memories using pgvector semantic search and BM25 hybrid retrieval. Returns semantically relevant results ranked by relevance.

## Arguments

- `$1` - Natural language search query (e.g., "how did we fix the trading bot", "Instagram content patterns", "revenue strategy decisions")

## Instructions

Execute the following steps:

### Step 1: Parse Query

Extract the search query from `$1`. If empty, ask the user what they want to search for.

### Step 2: Search AutoMem (Semantic + BM25 Hybrid)

Search personal notes using the hybrid search system:

```bash
PYTHONPATH=. python3.13 -c "
import sys
sys.path.insert(0, '.')
from utils.automem import AutoMem

mem = AutoMem()
results = mem.search_notes('QUERY_HERE', min_similarity=0.3, limit=10)
for r in results:
    print(f\"Sim: {r.get('similarity', 0):.2f} | Title: {r.get('title', 'N/A')} | Priority: {r.get('priority', 'N/A')} | Tags: {r.get('tags', [])}\")
    print(f\"  Content: {str(r.get('content', ''))[:200]}...\")
    print()
if not results:
    print('No matching memories found.')
"
```

### Step 3: Search Patterns

Search extracted patterns for related learnings:

```bash
PYTHONPATH=. python3.13 -c "
import sys
sys.path.insert(0, '.')
from utils.automem import AutoMem

mem = AutoMem()
# Search patterns table
results = mem.db_query(\"\"\"
    SELECT pattern_type, confidence, description, is_golden_rule, usage_count
    FROM patterns
    WHERE description ILIKE '%QUERY_HERE%'
    ORDER BY confidence DESC LIMIT 5
\"\"\")
for r in results:
    golden = ' [GOLDEN]' if r.get('is_golden_rule') else ''
    print(f\"Type: {r['pattern_type']} | Conf: {r['confidence']:.2f}{golden} | Uses: {r.get('usage_count', 0)}\")
    print(f\"  {r['description'][:200]}\")
    print()
if not results:
    print('No matching patterns found.')
"
```

### Step 4: Search Bot Activity (if relevant)

If the query mentions a specific bot or activity type, search recent bot history:

```bash
PYTHONPATH=. python3.13 -c "
import sys
sys.path.insert(0, '.')
from utils.automem import AutoMem

mem = AutoMem()
results = mem.db_query(\"\"\"
    SELECT bot_name, activity_type, status, title, detail, created_at
    FROM bot_activity
    WHERE title ILIKE '%QUERY_HERE%' OR detail::text ILIKE '%QUERY_HERE%'
    ORDER BY created_at DESC LIMIT 10
\"\"\")
for r in results:
    print(f\"{r['created_at']} | {r['bot_name']} | {r['activity_type']} | {r['status']}\")
    print(f\"  {r['title']}\")
    print()
if not results:
    print('No matching bot activity found.')
"
```

### Step 5: Search Confidence Audit Log (if relevant)

If the query relates to decisions or confidence scoring:

```bash
PYTHONPATH=. python3.13 -c "
import sys
sys.path.insert(0, '.')
from utils.automem import AutoMem

mem = AutoMem()
results = mem.db_query(\"\"\"
    SELECT bot_name, action, final_confidence, decision, reasoning, created_at
    FROM confidence_audit_log
    WHERE action ILIKE '%QUERY_HERE%' OR reasoning ILIKE '%QUERY_HERE%'
    ORDER BY created_at DESC LIMIT 5
\"\"\")
for r in results:
    print(f\"{r['created_at']} | {r['bot_name']} | {r['action']} | Conf: {r['final_confidence']:.2f} | Decision: {r['decision']}\")
    print(f\"  {r.get('reasoning', '')[:200]}\")
    print()
if not results:
    print('No matching confidence decisions found.')
"
```

### Step 6: Display Results

Present formatted results grouped by source:

```markdown
## Search Results: "QUERY"

### Memories (AutoMem - Semantic Search)
Found X relevant memories

1. **[title]** (relevance: X.XX, priority: [priority])
   [content - first 200 chars...]
   Tags: [tags]

2. ...

### Patterns
Found X matching patterns

1. **[pattern_type]** (confidence: X.XX) [GOLDEN if applicable]
   [description snippet]
   Usage count: [N]

### Bot Activity
Found X matching activities

1. **[bot_name]** - [activity_type] ([status])
   [title]
   [timestamp]

### Confidence Decisions
Found X matching decisions

1. **[bot_name]** - [action] (confidence: X.XX → [decision])
   [reasoning snippet]

---
Use `Read` tool on any result to see full details.
```

## Search Tips

Display these tips if query is empty or results are sparse:

```
### Search Tips

**Semantic Search** (AutoMem notes):
- "trading bot performance" -> finds notes about trading results
- "Instagram content strategy" -> finds content planning notes
- "revenue optimization" -> finds revenue-related decisions

**Keyword Search** (Patterns/Activity):
- Use specific terms: "VWAP", "Instagram", "confidence gate"
- Bot names: "trading_bot", "instagram_bot", "learning_bot"
- Activity types: "trade_execution", "content_planning", "article_store"
```

## Related Commands

- `/stats` - View system-wide statistics
- `/remember [content]` - Store new memory
- `/research-deep [question]` - Deep research with coverage requirements

---

*Natural language search across AutoMem memories, patterns, bot activity, and confidence decisions using pgvector hybrid search.*
