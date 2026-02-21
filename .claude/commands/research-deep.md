# /research-deep Command Protocol: Methodical Authoritative Research

## Purpose

Execute methodical, authoritative research on any topic using structured workflow with coverage requirements, confidence gates, and validation. Default thorough research with optional `--quick` flag for tactical needs.

**Mission Statement**: Quality over speed - correct answer > fast answer.

---

## Usage

```bash
# Full methodology (default)
/research-deep "Should we implement WebSocket streaming for the trading bot?"
/research-deep "Best practices for Instagram Reels automation"

# Tactical research (--quick flag)
/research-deep --quick "What's the best async pattern in Python?"
/research-deep --quick "How does Alpaca WebSocket API handle reconnection?"
```

**Arguments**:
- `$1`: Research question (required)
- `--quick`: Optional flag for tactical research (reduced rigor)

---

## Full Methodology (8 Steps)

### Step 1: AutoMem Pre-Check
Before starting research, check AutoMem for past research on similar topics:

```python
from utils.automem import AutoMem
mem = AutoMem()
results = mem.search_notes('RESEARCH_TOPIC', min_similarity=0.5, limit=5)
```

Surface findings to user. If no relevant patterns found, proceed fresh.
If user says "skip", proceed without retrieval.

### Step 2: Context Gathering
Understand the scope, constraints, and stakeholders:
- What problem does this solve?
- What are the constraints (cost, time, infrastructure)?
- Who benefits (which bot, which revenue stream)?

### Step 3: Coverage-Based Research
**Minimum coverage requirements** (2 academic + 1 industry + 1 community):

| Source Type | Minimum | Examples |
|-------------|---------|----------|
| **Academic** (Tier 1) | 2 sources | arXiv papers, peer-reviewed journals, university research |
| **Industry** (Tier 2) | 1 source | Official documentation, vendor whitepapers, engineering blogs |
| **Community** (Tier 3) | 1 source | GitHub implementations, Reddit discussions, HN threads, Stack Overflow |

Use WebSearch and WebFetch tools to gather sources across all tiers.

### Step 4: Gap Analysis
Identify what's missing, conflicting, or uncertain:
- Are sources agreeing? Where do they conflict?
- What questions remain unanswered?
- What assumptions are we making?

### Step 5: Confidence Assessment
Score confidence using the same framework as our Confidence Gate:
- **Base confidence**: How strong is the evidence? (0.0-1.0)
- **Validation**: Do sources corroborate each other? (0.0-1.0)
- **Historical**: Has AutoMem seen this pattern succeed before? (0.0-1.0)
- **Reflexive**: Second-pass review - does the recommendation hold up? (0.0-1.0)

**Threshold**: 90% target. If below 90%, surface uncertainties and ask user whether to continue researching or accept current confidence.

### Step 6: Synthesis
Combine findings into clear recommendation:
- What's the recommendation?
- Why this over alternatives?
- What are the trade-offs?
- What's the implementation path?

### Step 7: Grok Validation (Mandatory for Full Mode)
Send research context and recommendation to Grok for independent validation:

```
/grok-super [full research context + recommendation + confidence]
```

If Grok disagrees, surface BOTH perspectives to user:
- Claude's recommendation + reasoning
- Grok's perspective + reasoning
- User decides which to follow

Can skip ONLY with explicit user acknowledgment.

### Step 8: Present Findings
Deliver comprehensive research report (see Output Format below).

After presentation, store learnings to AutoMem:
```python
mem.add_note(
    title='Research: [topic summary]',
    content='[key findings and recommendation]',
    priority='normal',
    tags=['research', 'topic-tag1', 'topic-tag2']
)
```

---

## --quick Flag Behavior

When `--quick` flag present, reduce rigor for tactical research:

1. Skip AutoMem check
2. Minimal context gathering
3. Reduced coverage (1 academic OR 1 industry, skip community)
4. Skip gap analysis
5. Lower confidence threshold (70%+ acceptable)
6. Quick synthesis
7. **Skip Grok validation**
8. Present findings with confidence acknowledgment

**NOT stored to AutoMem** (insufficient rigor for learning).

---

## Output Format

### Full Methodology Output

```markdown
# Research: [Research Question]

**Date**: YYYY-MM-DD
**Time Spent**: XX minutes
**Confidence**: XX%
**Mode**: Full Methodology

---

## Executive Summary

[1-2 sentences: What's the recommendation? Why?]

---

## Coverage

- Academic: X sources (minimum: 2)
- Industry: Y sources (minimum: 1)
- Community: Z sources (minimum: 1)

---

## Detailed Findings

### Academic Perspective
[What academic research says]
**Sources**: [with URLs]

### Industry Perspective
[What official docs/vendors say]
**Sources**: [with URLs]

### Community Validation
[What production experience shows]
**Sources**: [with URLs]

---

## Gap Analysis

[Comparison matrix, tradeoffs, pros/cons]

---

## Recommendation

[Clear recommendation with reasoning]

**Why this recommendation**:
1. [Reason 1 with source citation]
2. [Reason 2 with source citation]
3. [Reason 3 with source citation]

---

## Grok Validation

[Grok's independent assessment]
**Agreement**: [Confirms / Suggests alternatives / Identifies gaps]

---

## Confidence: XX%

[If <90%: Explain uncertainties and gaps]

---

## Sources Bibliography

**Academic**: [Full citations with URLs]
**Industry**: [Full citations with URLs]
**Community**: [Full citations with URLs]

---

*Research completed using /research-deep methodology*
```

### Quick Mode Output

```markdown
# Research (Quick Mode): [Question]

**Time**: XX minutes | **Confidence**: XX%

## Answer
[Clear, concise answer]

## Source
[Single authoritative source with URL]

## Pattern
[Code example or implementation pattern if applicable]

---
*Quick mode (reduced rigor). For strategic decisions, use full methodology.*
```

---

## Coverage Checkpoint (Guardrail)

Before synthesis, validate coverage meets minimums. If not:

```
RESEARCH COVERAGE CHECKPOINT

Current coverage:
- Academic: X sources (minimum: 2)
- Industry: Y sources (minimum: 1)
- Community: Z sources (minimum: 1)

Confidence: XX%

Options:
- Continue research (recommended)
- Proceed anyway (acknowledge incomplete coverage)
```

User says "proceed anyway" to continue with gaps. No hard blocks.

---

## Confidence Threshold (Guardrail)

Before presenting findings, check confidence. If below 90%:

```
CONFIDENCE BELOW TARGET

Research complete. Confidence: XX% (target: 90%)

Uncertainties:
- [uncertainty 1]
- [uncertainty 2]

Options:
- Continue research to resolve uncertainties
- Accept XX% confidence and surface uncertainties in recommendation
```

---

## Integration with Project Workflow

```
/research-deep "question"  ->  Research complete, recommendation made
        |
Store to AutoMem  ->  Pattern saved for future reference
        |
Implement recommendation  ->  Apply findings to bot/system
        |
Track results  ->  Measure if recommendation worked
```

---

## When to Use /research-deep vs Quick Lookup

| Scenario | Use |
|----------|-----|
| Strategic decision (architecture, new strategy) | `/research-deep "question"` |
| Quick factual question | `/research-deep --quick "question"` |
| Trading strategy evaluation | `/research-deep "question"` (full) |
| Content strategy change | `/research-deep "question"` (full) |
| API integration question | `/research-deep --quick "question"` |

---

*Methodical research command with coverage validation, confidence gating, and multi-LLM verification via /grok-super.*
