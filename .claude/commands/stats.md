# /stats - AutoMem & System Statistics Dashboard

**Purpose**: Display comprehensive AutoMem statistics, pattern health, golden rules, and system status.

**Description**: Shows detailed system health including database stats, confidence distribution, pattern effectiveness, bot fleet status, and memory health indicators. Provides transparency into the learning system.

## Arguments

- `$1` (optional): Focus area — `all` (default), `patterns`, `bots`, `confidence`, `revenue`

## Instructions

Execute the following steps:

### Step 1: Gather Core Statistics

Run the following PostgreSQL queries to collect stats:

```bash
PYTHONPATH=. python3.13 -c "
import sys
sys.path.insert(0, '.')
from utils.automem import AutoMem
import json

mem = AutoMem()

# Core counts
stats = {}

# Conversations
result = mem.db_query('SELECT COUNT(*) as total FROM conversations')
stats['conversations'] = result[0]['total'] if result else 0

# Patterns
result = mem.db_query('SELECT COUNT(*) as total, COUNT(CASE WHEN is_golden_rule THEN 1 END) as golden FROM patterns')
stats['patterns_total'] = result[0]['total'] if result else 0
stats['golden_rules'] = result[0]['golden'] if result else 0

# Personal notes
result = mem.db_query('SELECT COUNT(*) as total FROM personal_notes')
stats['notes'] = result[0]['total'] if result else 0

# Confidence audit
result = mem.db_query('SELECT COUNT(*) as total FROM confidence_audit_log')
stats['confidence_decisions'] = result[0]['total'] if result else 0

# Guardrails
result = mem.db_query('SELECT COUNT(*) as total FROM guardrails WHERE is_active = true')
stats['guardrails'] = result[0]['total'] if result else 0

# Bot activity (24h)
result = mem.db_query(\"\"\"
    SELECT bot_name, COUNT(*) as count,
           COUNT(CASE WHEN status = 'completed' THEN 1 END) as success,
           COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed
    FROM bot_activity
    WHERE created_at > NOW() - INTERVAL '24 hours'
    GROUP BY bot_name ORDER BY count DESC
\"\"\")
stats['bot_activity_24h'] = result if result else []

# Revenue results
result = mem.db_query('SELECT COUNT(*) as total, COALESCE(SUM(revenue_amount), 0) as total_revenue FROM revenue_results')
stats['revenue_entries'] = result[0]['total'] if result else 0
stats['total_revenue'] = float(result[0]['total_revenue']) if result else 0.0

# Pattern confidence distribution
result = mem.db_query(\"\"\"
    SELECT
        COUNT(CASE WHEN confidence >= 0.90 THEN 1 END) as high,
        COUNT(CASE WHEN confidence >= 0.50 AND confidence < 0.90 THEN 1 END) as moderate,
        COUNT(CASE WHEN confidence < 0.50 THEN 1 END) as low
    FROM patterns
\"\"\")
stats['confidence_dist'] = result[0] if result else {'high': 0, 'moderate': 0, 'low': 0}

print(json.dumps(stats, indent=2, default=str))
"
```

### Step 2: Display Statistics Dashboard

Present the collected data in this format:

```markdown
## AutoMem Statistics Dashboard

### System Health
| Component | Status | Count |
|-----------|--------|-------|
| PostgreSQL + pgvector | [status] | [conversation_count] conversations |
| Patterns | [status] | [pattern_count] patterns ([golden] golden rules) |
| Personal Notes | [status] | [note_count] notes |
| Confidence Gate | [status] | [decision_count] decisions logged |
| Guardrails | [status] | [guardrail_count] active rules |

---

### Confidence Score Legend

| Score Range | Meaning | Indicator |
|-------------|---------|-----------|
| 0.90 - 1.00 | **Golden Rule** - Promoted pattern, 5+ uses, 90%+ success | HIGH |
| 0.75 - 0.89 | **High Confidence** - Well-validated, reliable pattern | GOOD |
| 0.50 - 0.74 | **Moderate** - Some validation, use with verification | OK |
| 0.30 - 0.49 | **Low** - Limited validation or stale | LOW |
| 0.00 - 0.29 | **Untrusted** - New or unvalidated | UNTRUSTED |

---

### Confidence Distribution

```
High (0.90+):    [bar] XX%
Moderate (0.50+): [bar] XX%
Low (<0.50):      [bar] XX%
```

---

### Golden Rules Summary ([count] total)

[List each golden rule with confidence, usage count, description]

---

### Bot Fleet Activity (Last 24h)

| Bot | Activities | Success | Failed | Rate |
|-----|-----------|---------|--------|------|
| [bot_name] | [count] | [success] | [failed] | [success_rate]% |

---

### Revenue Tracking

| Metric | Value |
|--------|-------|
| Total Revenue Entries | [count] |
| Total Revenue | $[amount] |
| Active Revenue Streams | [count] |

---

### Recommendations

[Based on stats, provide 1-3 actionable recommendations like:]
- "You have [N] high-confidence patterns approaching golden rule threshold."
- "[N] bots had failures in the last 24h — investigate."
- "No golden rules yet — need more runtime to promote patterns."
```

### Step 3: Focus Area Details (if specified)

If user specified a focus area (`patterns`, `bots`, `confidence`, `revenue`), provide deeper drill-down for that area with additional queries.

## Quick Access

```bash
# Direct database query
psql postgresql://localhost/personal_agent_hub -c "SELECT * FROM bot_fleet_overview;"
psql postgresql://localhost/personal_agent_hub -c "SELECT * FROM active_golden_rules;"
psql postgresql://localhost/personal_agent_hub -c "SELECT * FROM roadmap_phase_progress;"
```

## Related Commands

- `/search [query]` - Search memories by content
- `/remember [content]` - Store new memory to AutoMem

---

*AutoMem statistics dashboard — transparency into the learning system's health, patterns, and revenue tracking.*
