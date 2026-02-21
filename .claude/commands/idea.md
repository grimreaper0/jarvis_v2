# /idea Command Protocol: Catalog-First Context Building

## Purpose

Transform rough ideas into well-defined Codex entries with 90%+ CPQQRT confidence. The goal is to gather enough context that Claude (or another LLM) can execute the work autonomously.

**CRITICAL**: /idea operates in **read-only mode** (like Claude Code's /plan mode). No code edits. All outputs go to The Codex.

## Mission Statement
> Gather context iteratively until 90%+ confidence is achieved, then persist to The Codex for future execution.

## What /idea Does NOT Do

- ❌ **Does NOT edit code** - Read-only exploration only
- ❌ **Does NOT save local files** - No `ideas/<slug>.md` files
- ❌ **Does NOT proceed below 60% confidence** - Block and gather more context

## What /idea DOES Do

- ✅ **Creates/updates Codex entries** - QUE-XXX is the output
- ✅ **Gathers CPQQRT context** - All 6 fields populated
- ✅ **Classifies taxonomy** - domain + work_type mandatory
- ✅ **Calculates confidence** - 0-100 score from CPQQRT completeness
- ✅ **Uploads requirements file** - Attached to quest in catalog
- ✅ **Explores codebase** - Read files, search patterns (read-only)

## Argument Handling (v2.1.20+)

**NEW**: This command uses positional argument `$1` for input instead of `$ARGUMENTS` string parsing.
- **If `$1` provided**: Use directly (e.g., `/idea "description"` → `$1 = "description"` or `/idea QUE-085` → `$1 = "QUE-085"`)
- **If `$1` empty**: Prompt user interactively for idea or quest slug
- **Both syntaxes work**: Backward compatible with `$ARGUMENTS` for legacy scripts

## Required MCPs

**This command requires:**
- **Sequential Thinking MCP** - Required for structured reasoning

**This command optionally uses:**
- **AutoMem MCP** - For pattern retrieval and research
- **The Codex API** - For CRUD operations (REST API, not MCP)

## Usage

```bash
# Create new quest from description (PRIMARY)
/idea "create a new filter for reports"         # Text description via $1
/idea "build real-time customer dashboard"      # Supports multi-word descriptions

# Enrich existing quest
/idea QUE-085                                    # Quest slug via $1
/idea que-085                                    # Case insensitive

# Interactive mode (prompts for idea)
/idea                                            # $1 is empty, prompt user

# Reset current session (optional flag)
/idea --reset                                    # Reset ideation context
```

## Workflow Overview

```
/idea "description"
    │
    ├── STEP 1: Input Detection (QUE-XXX or new)
    ├── STEP 2: Quest Fetch (if QUE-XXX)
    ├── STEP 3: Silent Scan (internal gap analysis)
    ├── STEP 4: Research Phase (AutoMem, codebase)
    ├── STEP 5: Clarify Loop (interactive Q&A)
    ├── STEP 6: Taxonomy Classification
    ├── STEP 7: Echo Check (summary validation)
    ├── STEP 8: Score & Categorize
    └── STEP 9: Catalog Update (CREATE or UPDATE)
          │
          ▼
    QUE-XXX in The Codex
          │
          ▼
    /start QUE-XXX (when ready to begin work)
```

## Nine-Step Protocol

### STEP 0: Role Delegation

**MANDATORY**: This command is orchestrated by `requirements-analyst-role`.

When executing `/idea`:
1. Load the requirements-analyst-role context
2. Follow its CPQQRT scoring rubric
3. Apply its delegation patterns for specialists
4. Use its confidence thresholds

### STEP 1: Input Detection

**Detect input type and route accordingly:**

```python
input = $1 or ""  # Get positional argument; empty if not provided

# If no input, prompt user interactively
if not input:
    input = prompt("Enter idea description or quest slug (QUE-XXX): ").strip()

if re.match(r'^[Qq][Uu][Ee]-\d{3}$', input):
    # QUE-XXX pattern - ENRICHMENT mode (update existing)
    workflow = "enrichment"
    slug = input.upper()

else:
    # Text description - CREATION mode (new quest)
    workflow = "creation"
    description = input

    # Campaign Detection: Check for CAMP-XXX references in description
    camp_match = re.search(r'CAMP-\d{3}', input, re.IGNORECASE)
    if camp_match:
        campaign_ref = camp_match.group(0).upper()
        # After quest creation, prompt user to link to campaign task
        # "Detected campaign reference: {campaign_ref}. Link this quest to a campaign task?"
```

### STEP 2: Quest Fetch (if QUE-XXX)

**For existing quests, fetch current state:**

```python
# Fetch solution from The Codex
slug = "QUE-085"  # UPPERCASE - API is case-sensitive
url = f"http://localhost:8001/api/v1/solutions/slug/{slug}"

# Extract:
# - solution.id (UUID for updates)
# - solution.name, solution.description
# - solution.context, .purpose, .quantity, .quality, .resources, .timeline
# - solution.confidence, solution.status
# - solution.domain, solution.work_type
```

**Check for existing requirements file:**
```python
# List files attached to quest
GET /solutions/{solution_id}/files

# If requirements file exists:
# - Download and parse for existing context
# - Note file_id for later replacement
```

**Display current state:**
```
📦 Quest found: QUE-085 - [solution.name]
📊 Current confidence: [X]%
📝 Status: [status]

🎯 CPQQRT Completeness:
   Context: [✅/❌] [preview if exists]
   Purpose: [✅/❌] [preview if exists]
   Quantity: [✅/❌] [preview if exists]
   Quality: [✅/❌] [preview if exists]
   Resources: [✅/❌] [preview if exists]
   Timeline: [✅/❌] [preview if exists]
```

### STEP 3: Silent Scan (Internal)

**Privately analyze what's needed (don't show to user):**

- List every fact or constraint still needed
- Identify CPQQRT gaps (empty or minimal fields)
- Plan clarification questions by priority:
  1. Purpose (why does this exist?)
  2. Context (what problem does this solve?)
  3. Quality (what are acceptance criteria?)
  4. Quantity (what's the scope?)
  5. Resources (what's needed?)
  6. Timeline (when is it due?)

### STEP 4: Research Phase

**Research BEFORE asking the user:**

1. **AutoMem Patterns via MaestroAI**:
   ```
   Call: mcp__maestroai__orchestrate
   Request: "Recall patterns for: <quest name/type>"
   Context: {"operation": "recall"}

   Look for:
   - Similar quests with complete CPQQRT
   - Domain-specific patterns
   - Known constraints or requirements
   ```

2. **Codebase Exploration** (read-only):
   ```
   Use: Read, Grep, Glob tools

   Look for:
   - Related files or modules
   - Existing implementations to reference
   - Technical constraints
   ```

3. **Cross-Reference Quests**:
   ```
   Search existing quests of same type

   Look for:
   - CPQQRT field examples
   - Common patterns
   - Resource estimates
   ```

**Pre-fill from research findings** - Only ask questions that research couldn't answer.

### STEP 5: Clarify Loop (Interactive)

**Ask ONE question at a time until 90%+ confidence:**

**Process:**
1. Ask the highest-priority unanswered question
2. Wait for user response
3. Update CPQQRT fields with answer
4. Recalculate confidence score
5. Repeat until 90%+ OR max questions reached

**Coverage areas:**
- Purpose and business value
- Target audience/stakeholders
- Must-include facts and requirements
- Success criteria (measurable)
- Tech stack (if code/technical)
- Edge cases and exceptions
- Risk tolerances
- Dependencies and constraints

**Limits:**
- Maximum 15 questions total
- Escape hatch at 10 questions ("Good enough for now?")
- Show confidence progress after each answer

**Question format:**
```
📊 Current confidence: 72% (need 90%+)

Q[N]: [Clear, specific question targeting a CPQQRT gap]

[After answer]
✅ Quality field updated. Confidence: 72% → 78%
```

### STEP 6: Taxonomy Classification

**Classify domain and work_type:**

**Domain Detection:**

| If description mentions... | Set domain to |
|----------------------------|---------------|
| Claude, LLM, ML, embedding, agent, AI | `ai` |
| dbt, model, dimension, fact, schema, transformation | `data_model` |
| ETL, pipeline, Orchestra, Prefect, dlthub, ingestion | `pipeline` |
| Tableau, dashboard, workbook, viz | `dashboard` |
| report, extract, PDF, scheduled | `report` |
| Streamlit, React, web app, UI | `application` |
| API, webhook, sync, connector | `integration` |
| AWS, Snowflake infra, CI/CD, deploy | `infrastructure` |
| ticket, helpdesk, FreshService, support | `support` |
| (none of above) | `other` |

**Work Type Detection:**

| If task... | Set work_type to |
|------------|------------------|
| new, create, implement, add, build | `new_feature` |
| fix, bug, error, broken, issue | `bug_fix` |
| improve, enhance, update, optimize | `enhancement` |
| maintain, routine, cleanup, refactor | `maintenance` |
| research, investigate, explore, POC | `research` |
| document, README, runbook, guide | `documentation` |
| support, help, assist, ticket | `support` |
| infra, deploy, migrate, upgrade | `infrastructure` |

**If ambiguous:** Ask user to confirm classification.

### STEP 7: Echo Check (Validation)

**Present summary for user confirmation:**

```
## ECHO CHECK

[One crisp sentence: deliverable + #1 must-include fact + hardest constraint]

Example: "A Tableau dashboard showing MRR, churn, and active users for enterprise
customers (>$10K MRR), refreshing hourly, with the hardest constraint being
accurate churn calculation across multiple subscription tiers."

**User Confirmed**: [ ] Pending

---

Response options:
✅ YES → Lock requirements, proceed to catalog
❌ EDITS → Clarify what needs changing
⚫ BLUEPRINT → Show implementation phases (optional)
⚠️ RISK → Show top 3 failure scenarios (optional)
```

**WAIT for user response before proceeding.**

### STEP 8: Score & Categorize

**Calculate final CPQQRT confidence:**

```python
WEIGHTS = {
    "context": 20,
    "purpose": 20,
    "quantity": 15,
    "quality": 15,
    "resources": 15,
    "timeline": 15
}

def score_field(content, max_score):
    if not content or not content.strip():
        return 0  # empty
    length = len(content.strip())
    if length < 20:
        return int(max_score * 0.25)  # minimal
    elif length < 100:
        return int(max_score * 0.50)  # brief
    elif length < 300:
        return int(max_score * 0.75)  # detailed
    else:
        return max_score  # comprehensive

total_score = sum(score_field(fields[f], WEIGHTS[f]) for f in WEIGHTS)
```

**Set status based on confidence:**

| Confidence | Status | Autonomy Level |
|------------|--------|----------------|
| 90-100 | `ready` | autonomous |
| 60-89 | `defining` | needs_pairing |
| <60 | `idea` | needs_approval |

### STEP 9: Catalog Update

**CREATE new quest or UPDATE existing:**

**For NEW quests:**

**DUPLICATE DETECTION (before creation):**
```python
# Search for similar quests before creating
import urllib.request
import json
import os

# No API key needed
api_key = None
quest_name = "<from ECHO CHECK>"  # The name we're about to use

# Search existing quests
search_url = f"http://localhost:8001/api/v1/solutions?search={urllib.parse.quote(quest_name)}&status=ready,defining,idea,in_development"
req = urllib.request.Request(search_url, headers={})

with urllib.request.urlopen(req) as resp:
    search_results = json.loads(resp.read().decode())

# Check for potential duplicates
if search_results and len(search_results) > 0:
    similar_quests = search_results[:3]  # Top 3 matches

    print("⚠️  POTENTIAL DUPLICATES")
    print()
    print("Similar quests found:")
    for i, quest in enumerate(similar_quests, 1):
        print(f"{i}. {quest['slug']}: {quest['name']}")
        print(f"   Status: {quest['status']} | Confidence: {quest.get('confidence', 'N/A')}")
        print(f"   http://localhost:3001/solutions/{quest['slug'].lower()}")
        print()

    print("Options:")
    print("  - Type 'proceed' to create new quest anyway")
    print(f"  - Type 'enrich {similar_quests[0]['slug']}' to add context to existing")
    print("  - Type 'cancel' to abort")
    print()

    # WAIT for user decision
    # If user says "enrich QUE-XXX": Switch to UPDATE mode for that quest
    # If user says "proceed": Continue with POST below
    # If user says "cancel": Exit /idea
```

**If no duplicates found OR user said "proceed":**
```python
# POST /solutions/
payload = {
    "name": "<from ECHO CHECK>",
    "description": "<expanded description>",
    "solution_type": "<match domain for legacy>",
    "status": "<from confidence>",
    "domain": "<classified>",
    "work_type": "<classified>",
    "context": "<C>",
    "purpose": "<P>",
    "quantity": "<Q>",
    "quality": "<Q>",
    "resources": "<R>",
    "timeline": "<T>",
    "confidence": <0-100>,
    "owner_name": "FiestyGoat AI",
    "team": "D&A",
    "tags": ["<auto-detected>"]
}
```

**For EXISTING quests (QUE-XXX):**
```python
# PATCH /solutions/{solution_id}
payload = {
    "context": "<enriched>",
    "purpose": "<enriched>",
    "quantity": "<enriched>",
    "quality": "<enriched>",
    "resources": "<enriched>",
    "timeline": "<enriched>",
    "confidence": <new score>,
    "status": "<updated>",
    "domain": "<if changed>",
    "work_type": "<if changed>"
}
```

**Upload requirements file:**
```bash
# If file exists for this quest, delete it first
# DELETE /solutions/{solution_id}/files/{file_id}

# Then upload new requirements file
curl -X POST "/solutions/{solution_id}/files" \
  -F "file=@requirements.md" \
  -F "file_type=requirements" \
  -F "description=Requirements document for QUE-XXX"
```

**Display completion:**
```
✅ Quest created/updated!

📦 Quest: QUE-XXX - [name]
📊 Confidence: [X]% ([status])
🏷️  Domain: [domain]
🔧 Work Type: [work_type]
🤖 Autonomy: [autonomous|needs_pairing|needs_approval]

📄 Requirements file uploaded
🔗 http://localhost:3001/solutions/que-xxx

💡 Next steps:
   - Run `/start QUE-XXX` when ready to begin work
   - Or `/improve-confidence QUE-XXX` to raise confidence
   - Or review the quest at the URL above

📋 Campaign link available? (if CAMP-XXX detected)
   - Run `/campaigns link-quest --task <task_id> --quest QUE-XXX`
```

### STEP 9b: Campaign Linkage (Optional)

**If CAMP-XXX was detected in the input description during Step 1:**

After creating the quest, offer to link it to a campaign task:

```
📋 CAMPAIGN REFERENCE DETECTED: CAMP-XXX

Would you like to link QUE-XXX to a campaign task?

1. View campaign: /campaigns get CAMP-XXX
2. Link to task: /campaigns link-quest --task <task_id> --quest QUE-XXX
3. Skip (link later)
```

This enables quests created via `/idea` to be immediately associated with campaign delivery plans.

## CPQQRT Scoring Rubric

### Field Weights (Total: 100 points)

| Field | Weight | Description |
|-------|--------|-------------|
| **Context** (C) | 20 | Business background, problem statement, stakeholder needs |
| **Purpose** (P) | 20 | Why this quest exists, business value delivered |
| **Quantity** (Q) | 15 | Scale, volume, scope of work, data sizes |
| **Quality** (Q) | 15 | Acceptance criteria, SLAs, quality requirements |
| **Resources** (R) | 15 | Team, dependencies, budget, infrastructure |
| **Timeline** (T) | 15 | Milestones, deadlines, phases |

### Completeness Levels

| Level | Characters | Score % |
|-------|------------|---------|
| Empty | 0 | 0% |
| Minimal | <20 | 25% |
| Brief | 20-100 | 50% |
| Detailed | 100-300 | 75% |
| Comprehensive | 300+ | 100% |

## Requirements File Template

The requirements file uploaded to The Codex follows this structure:

```markdown
# Requirements: [Quest Name]

**Quest**: QUE-XXX
**Created**: YYYY-MM-DD
**Status**: [status]
**Confidence**: [X]%
**Autonomy Level**: [autonomous|needs_pairing|needs_approval]

---

## ECHO CHECK

[One sentence summary confirmed by user]

**User Confirmed**: ✅ YES (YYYY-MM-DD)

---

## CLARIFICATIONS

### Q1: [Question]
**A**: [Answer]

### Q2: [Question]
**A**: [Answer]

[... all Q&A ...]

---

## BLUEPRINT (Optional)

[Implementation phases if requested]

---

## RISKS (Optional)

### 1. [Risk Name] (Severity)
**Risk**: [Description]
**Mitigation**: [Strategy]

---

## SUCCESS CRITERIA

- [ ] [Criterion 1]
- [ ] [Criterion 2]
- [ ] [Criterion 3]

---

## CPQQRT SUMMARY

| Field | Score | Status |
|-------|-------|--------|
| Context | X/20 | [status] |
| Purpose | X/20 | [status] |
| Quantity | X/15 | [status] |
| Quality | X/15 | [status] |
| Resources | X/15 | [status] |
| Timeline | X/15 | [status] |
| **Total** | **X/100** | [ready/defining/idea] |

---

## THREE-DIMENSIONAL TAXONOMY

**Domain**: [domain]
**Work Type**: [work_type]
**Autonomy Level**: [computed from confidence]

---

*Generated by /idea command - YYYY-MM-DD*
```

## Claude Instructions

When user runs `/idea [input]`:

### Pre-Execution Check

1. **Verify read-only mode**: This command does NOT edit code
2. **Load Sequential Thinking**: Use for all complex reasoning
3. **Check AutoMem**: Retrieve relevant patterns before starting

### Execution Flow

**Use Sequential Thinking template:**
```
Thought 1: "Starting /idea. Determine input type: QUE-XXX pattern or new description?"
Thought 2: "If QUE-XXX: Fetch quest from catalog, check CPQQRT completeness"
Thought 3: "If QUE-XXX: Check for existing requirements file attachment"
Thought 4: "Step 3: SILENT SCAN - listing facts/constraints still needed"
Thought 5: "Step 4: RESEARCH - check AutoMem, codebase (read-only)"
Thought 6: "Step 5: CLARIFY LOOP - asking questions one at a time"
Thought 7: "Step 6: TAXONOMY - classify domain and work_type"
Thought 8: "Step 7: ECHO CHECK - present summary for confirmation"
Thought 9: "Step 8: SCORE - calculate CPQQRT confidence"
Thought 10: "Step 9: CATALOG UPDATE - create/update quest, upload file"
Thought 11: "Final verification - quest in catalog with requirements file?"
```

### API Patterns

**Fetch quest by slug:**
```python
import urllib.request, json, os
# No API key needed
api_key = None
slug = "QUE-085"  # UPPERCASE
url = f"http://localhost:8001/api/v1/solutions/slug/{slug}"
req = urllib.request.Request(url, headers={})
with urllib.request.urlopen(req) as resp:
    solution = json.loads(resp.read().decode())
```

**Create quest:**
```python
url = "http://localhost:8001/api/v1/solutions/"
req = urllib.request.Request(url, data=json.dumps(payload).encode(), method='POST',
    headers={"Content-Type": "application/json"})
```

**Update quest:**
```python
url = f"http://localhost:8001/api/v1/solutions/{solution_id}"
req = urllib.request.Request(url, data=json.dumps(payload).encode(), method='PATCH',
    headers={"Content-Type": "application/json"})
```

**Upload file:**
```bash
curl -X POST "http://localhost:8001/api/v1/solutions/{solution_id}/files" \
  \
  -F "file=@/path/to/requirements.md" \
  -F "file_type=requirements"
```

## Error Handling

**Quest not found:**
```
❌ Quest not found: QUE-999

Check the slug and try again, or create a new quest:
  /idea "your idea description"
```

**Confidence too low to proceed:**
```
⚠️ Confidence is [X]% (below 60% threshold)

This quest needs more context before it can be worked on.
Continue answering questions, or save as-is and return later.

[Continue] [Save as idea]
```

**API error:**
```
❌ Failed to update The Codex

Error: [error message]

Your requirements are preserved. Try again or contact support.
```

## Integration with ADLC

**ADLC Phase**: PLAN (context gathering)

**Workflow chain:**
```
/idea "description"  → Creates QUE-XXX with 90%+ confidence
        ↓
/start QUE-XXX       → Creates git branch + project scaffold
        ↓
Development          → Code changes
        ↓
/complete project    → Archives to AutoMem, updates quest
```

## Related Commands

### /improve-confidence - Iterative CPQQRT Enhancement

**When to use /improve-confidence instead of /idea:**

| Scenario | Use |
|----------|-----|
| **New idea, no quest exists** | `/idea "description"` |
| **Quest exists but confidence <90%** | `/improve-confidence QUE-XXX` |
| **/start blocked due to low confidence** | `/improve-confidence QUE-XXX` |
| **Adding context to existing quest** | Either works (prefer /improve-confidence) |

**Workflow relationship:**

```
/idea "new feature"
    ↓ Creates QUE-XXX at 65% confidence
    ↓
/improve-confidence QUE-XXX
    ↓ Raises to 85% (adds Quality, Resources)
    ↓
/improve-confidence QUE-XXX
    ↓ Raises to 92% (adds Timeline, clarifications)
    ↓
/start QUE-XXX
    ✅ Proceeds (confidence ≥90%)
```

**Key difference:**
- `/idea` - Focused on CREATION: Input detection, taxonomy, initial CPQQRT
- `/improve-confidence` - Focused on ITERATION: Gap analysis, targeted questions, score improvement

### /start - Begin Development

**Prerequisites:**
- Quest MUST exist in catalog (created by /idea)
- Confidence gate: 90%+ proceed, 60-89% warn, <60% block

**Handoff:**
```
/idea QUE-085 (creates quest, 92% confidence)
    ↓
/start QUE-085 (creates git branch, project scaffold)
```

### /complete - Archive Project

**When work is done:**
```
/complete feature-que-085_taxonomy
    ↓
Archives to AutoMem, updates quest status in The Codex
```

---

*Catalog-First Context Building - Transform rough ideas into well-defined quests*
