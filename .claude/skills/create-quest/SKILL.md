---
name: create-quest
description: |
  Create a new quest from current conversation context with full CPQQRT documentation.
  Captures context, generates requirements doc with 95% confidence context, creates quest in the codex.
  Triggers on: "/create-quest", "create a quest for this", "add to the codex", "make this a quest"
allowed-tools:
  - Bash
  - Read
  - Write
  - Glob
  - Grep
  - Task
user-invocable: true
---

# Create Quest from Context

## Overview

Transform current conversation context into a fully documented quest in the The Codex. This skill captures context, generates CPQQRT fields, creates a comprehensive requirements document, and registers the quest - all in one workflow.

## When This Skill Activates

**Trigger Keywords**: /create-quest, create a quest for this, add to the codex, make this a quest

This skill applies when:
- User wants to formalize current work into a tracked quest
- User says `/create-quest` command
- User wants to capture context for future sessions
- User needs a requirements doc with 95% confidence context

## When NOT to Trigger

Do NOT activate this skill when:
- User is updating an existing quest (use `/codex-quests update`)
- User just wants to view quests (use `/codex-quests list`)
- User just wants to capture a quick idea (use `/idea`)

## Workflow Steps

### Step 1: Context Capture

Before creating the quest, gather from conversation:

1. **Problem Statement**: What problem is being solved?
2. **Proposed Approach**: What solution approach was discussed?
3. **Key Decisions**: What decisions were made and why?
4. **Technical Details**: Technologies, systems, patterns involved
5. **Owner**: Who is responsible? (Default: FiestyGoat AI / grimreaper0)

**Ask if unclear**:
- "What should we call this quest?"
- "What type of quest is this?" (data_model, pipeline, dashboard, application, ai, etc.)
- "What's the target timeline?"

**Auto-determine from context** (QUE-085 taxonomy):
- **Domain**: What area of the platform? (ai, data_model, pipeline, dashboard, report, application, integration, infrastructure, support, other)
- **Work Type**: What kind of work? (new_feature, bug_fix, enhancement, maintenance, research, documentation, support, infrastructure)

### Step 2: Generate CPQQRT

From captured context, generate:

| Field | Source | Content |
|-------|--------|---------|
| **C**ontext | Conversation history | Business background, problem statement, existing systems |
| **P**urpose | Problem + approach | Why this exists, what it enables, expected outcomes |
| **Q**uantity | Technical details | Scale (records, users, frequency), systems affected |
| **Q**uality | Discussion/standards | Success criteria, SLAs, acceptance standards |
| **R**esources | Team/dependencies | Team members, dependencies, external systems |
| **T**imeline | Discussion/estimates | Phases, milestones, deadlines |

**Confidence Assessment**:
Evaluate quest confidence (0-100):
- 90-100: Clear requirements, proven approach, dependencies known
- 70-89: Good understanding, some unknowns remain
- 50-69: Exploratory, multiple approaches possible
- <50: Highly uncertain, needs more research

### Step 3: Create Requirements Document

Generate a comprehensive requirements document and **upload it to The Codex** (NOT saved locally - the project folder may be deleted before work starts).

**Workflow**:
1. Write requirements doc to scratchpad: `/private/tmp/claude/.../scratchpad/que-XXX-requirements.md`
2. Create quest in the codex (Step 4)
3. Upload requirements doc to quest (Step 5)
4. Delete local temp file

**Document Structure**:
```markdown
# [Quest Name] - Requirements

## Executive Summary
[2-3 sentence overview]

## Problem Statement
[Detailed problem description from context]

## Proposed Solution
[Solution approach with rationale]

## Technical Specifications
### Systems Involved
[List of systems: dbt, Snowflake, Tableau, etc.]

### Data Flow
[How data moves through the solution]

### Integration Points
[External systems, APIs, dependencies]

## CPQQRT Documentation
### Context
[Full business context]

### Purpose
[Why this quest exists]

### Quantity
[Scale and volume metrics]

### Quality
[Success criteria and SLAs]

### Resources
[Team, budget, dependencies]

### Timeline
[Phases and milestones]

## Implementation Guide
### Phase 1: [Phase Name]
- [ ] Task 1
- [ ] Task 2

### Phase 2: [Phase Name]
- [ ] Task 1
- [ ] Task 2

## Definition of Done
- [ ] [Acceptance criterion 1]
- [ ] [Acceptance criterion 2]

## Related Quests
[Links to related QUE-XXX entries]

---
*Generated: [timestamp]*
*Quest: QUE-XXX*
*Owner: [owner name]*
```

### Step 4: Create Quest in The Codex

Use The Codex API to create the quest:

```bash
python3 << 'PYEOF'
import urllib.request
import json
import os

# No API key needed
api_key = None
url = "http://localhost:8001/api/v1/solutions/"

payload = {
    "name": "SOLUTION_NAME",
    "description": "DESCRIPTION",
    "solution_type": "TYPE",  # data_model, pipeline, dashboard, application, ai, etc. (legacy)
    "status": "idea",
    # Three-dimensional taxonomy (QUE-085) - REQUIRED
    "domain": "DOMAIN",  # ai, data_model, pipeline, dashboard, report, application, integration, infrastructure, support, other
    "work_type": "WORK_TYPE",  # new_feature, bug_fix, enhancement, maintenance, research, documentation, support, infrastructure
    # CPQQRT fields
    "context": "CONTEXT_TEXT",
    "purpose": "PURPOSE_TEXT",
    "quantity": "QUANTITY_TEXT",
    "quality": "QUALITY_TEXT",
    "resources": "RESOURCES_TEXT",
    "timeline": "TIMELINE_TEXT",
    "owner_name": "FiestyGoat AI",
    "team": "FiestyGoat AI",
    "confidence": 90,  # Assessed confidence level (also determines autonomy_level)
    "tags": ["tag1", "tag2"]
}

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode(),
    method='POST',
    headers={
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }
)

with urllib.request.urlopen(req) as response:
    data = json.loads(response.read().decode())

print(f"Quest created: {data['slug']}")
print(f"Quest ID: {data['id']}")
print(f"URL: http://localhost:3001/solutions/{data['slug']}")
PYEOF
```

### Step 5: Upload Requirements Document

**CRITICAL**: Upload the requirements doc to the quest in the codex. This ensures documentation persists even if local files are deleted.

```bash
# Upload requirements doc to quest
curl -X POST "http://localhost:8001/api/v1/solutions/{solution_id}/files" \
  \
  -F "file=@/path/to/scratchpad/que-XXX-requirements.md" \
  -F "file_type=requirements" \
  -F "description=Comprehensive requirements document with 95% confidence context" \
  -F "uploaded_by=fiestygoat"
```

After successful upload:
- Delete the local temp file from scratchpad
- The requirements doc is now permanently attached to the quest

### Step 6: Save to AutoMem

Store quest context as a pattern for future retrieval:

```
mcp__maestroai__orchestrate:
  request: "Store quest context: Quest [QUE-XXX]: [Name] - [Brief description of approach]"
  context:
    operation: "store"
    category: "solution"
    tags: ["que-xxx", "solution-type", "key-technology"]
    metadata:
      solution_slug: "QUE-XXX"
      confidence: 90
      requirements_doc: "Uploaded to The Codex"
      session_id: "${CLAUDE_SESSION_ID}"  # v2.1.9+ session linking
```

**Session Linking (v2.1.9+)**:
- Use `${CLAUDE_SESSION_ID}` substitution to capture the session where quest was created
- Enables future sessions to link back to the originating conversation
- Useful for context retrieval: "What was discussed when QUE-XXX was created?"

## Quest Types Reference (Legacy)

| Type | Value | When to Use |
|------|-------|-------------|
| Data Model | `data_model` | dbt models, Snowflake tables/views |
| Pipeline | `pipeline` | ETL/ELT pipelines (Orchestra, Prefect, dlthub) |
| Dashboard | `dashboard` | Tableau dashboards |
| Report | `report` | Tableau reports, scheduled extracts |
| Application | `application` | Streamlit, React apps |
| Integration | `integration` | API integrations, data feeds |
| AI | `ai` | AI/ML solutions, agent enhancements |
| Other | `other` | Anything else |

## Three-Dimensional Taxonomy (QUE-085)

**CRITICAL**: When creating a quest, ALWAYS determine and set both `domain` and `work_type`.

### Domain Selection (What Area)

| If the request involves... | Set domain to... |
|----------------------------|------------------|
| AI, ML, LLM, Claude, chatbot, embeddings, vectors | `ai` |
| dbt models, Snowflake tables, data transformations, SQL | `data_model` |
| ETL, ELT, Orchestra, Prefect, dlthub, data loading | `pipeline` |
| Tableau dashboards, visualizations, KPIs, charts | `dashboard` |
| Scheduled reports, extracts, exports to users | `report` |
| Streamlit, React, web app, UI, frontend | `application` |
| API, webhook, external system, data feed | `integration` |
| AWS, infrastructure, DevOps, deployment, CI/CD | `infrastructure` |
| FreshService, support ticket, user help | `support` |
| Unclear or doesn't fit above | `other` |

### Work Type Selection (What Kind of Work)

| If the request is... | Set work_type to... |
|---------------------|---------------------|
| Building something brand new from scratch | `new_feature` |
| Fixing a bug, error, or defect | `bug_fix` |
| Improving or extending existing functionality | `enhancement` |
| Routine updates, dependency upgrades, housekeeping | `maintenance` |
| Investigation, POC, analysis, evaluation | `research` |
| Creating docs, runbooks, knowledge articles | `documentation` |
| Helping users, troubleshooting, answering questions | `support` |
| Platform/infra work (not tied to specific feature) | `infrastructure` |

### Autonomy Level (Computed from Confidence)

The API automatically computes `autonomy_level` based on the confidence score:
- **autonomous**: confidence >=85% (Claude can work independently)
- **needs_pairing**: confidence 60-84% (requires collaboration)
- **needs_approval**: confidence <60% (requires explicit approval)

### Examples

| Request | Domain | Work Type |
|---------|--------|-----------|
| "Build a Tableau dashboard for sales KPIs" | `dashboard` | `new_feature` |
| "Fix the broken dbt model for inventory" | `data_model` | `bug_fix` |
| "Add MaestroAI multi-LLM routing" | `ai` | `new_feature` |
| "Investigate why Orchestra pipeline is slow" | `pipeline` | `research` |
| "Update the Snowflake sync documentation" | `data_model` | `documentation` |

## Confidence Scoring Guide

| Score | Meaning | Requirements |
|-------|---------|--------------|
| 95-100 | Production-ready | All requirements clear, approach proven, no unknowns |
| 90-94 | High confidence | Clear requirements, approach validated, minor uncertainties |
| 80-89 | Good confidence | Requirements understood, approach reasonable, some gaps |
| 70-79 | Moderate confidence | General direction clear, details need work |
| 60-69 | Exploratory | Problem understood, solution approach uncertain |
| <60 | Needs research | Too many unknowns to proceed confidently |

**Target**: Requirements doc should provide 95%+ confidence for fresh context window.

## Validation Checklist

Before finalizing, verify:
- [ ] Quest name is descriptive and unique
- [ ] Quest type matches the deliverable
- [ ] **Domain correctly categorized** (QUE-085)
- [ ] **Work type correctly categorized** (QUE-085)
- [ ] All CPQQRT fields are populated with substantive content
- [ ] Requirements doc has enough context for 95% confidence
- [ ] Owner name is correct (default: FiestyGoat AI)
- [ ] Confidence score reflects actual assessment
- [ ] Requirements doc uploaded to The Codex (NOT saved locally)

## Common Mistakes to Avoid

1. **Mistake**: Creating quest without capturing conversation context
   - **Why it's wrong**: Loses valuable context that drove the quest
   - **Correct approach**: Always review conversation history before generating CPQQRT

2. **Mistake**: Setting confidence too high without evidence
   - **Why it's wrong**: Inflated confidence leads to failed implementations
   - **Correct approach**: Honestly assess unknowns, gaps, and risks

3. **Mistake**: Skipping requirements doc
   - **Why it's wrong**: Future sessions won't have context needed for 95% confidence
   - **Correct approach**: Always create comprehensive requirements doc

4. **Mistake**: Using wrong quest number
   - **Why it's wrong**: Creates confusion in the codex
   - **Correct approach**: Query API for highest existing number, increment by 1

## Examples

### Example 1: Creating Quest from Technical Discussion

**User Request**: "We've been discussing adding planner and security agents. Create a quest for this."

**Expected Behavior**:
1. Review conversation for agent enhancement details
2. Extract: 8 new agents, 3 phases, gap analysis findings
3. Generate CPQQRT with implementation details
4. Write requirements doc to scratchpad
5. Create quest in the codex
6. Upload requirements doc to quest
7. Save pattern to AutoMem

**Output**:
```
Quest Created: QUE-084

Name: Agent Roles & Specialists Enhancement
Type: ai
Status: idea
Confidence: 90

CPQQRT Summary:
- Context: Gap analysis against top GitHub repos revealed 8 missing agents
- Purpose: Expand platform capabilities with planner, security, error-resolver roles
- Quantity: 8 new agents across 3 phases
- Quality: Each agent must have comprehensive SKILL.md, tests, integration
- Resources: Claude development, FiestyGoat AI oversight
- Timeline: Phase 1 (Foundation), Phase 2 (Specialist), Phase 3 (Advanced)

Requirements Doc: Uploaded to The Codex (file: agent-enhancement-requirements.md)
URL: http://localhost:3001/solutions/QUE-084

Pattern saved to AutoMem for future retrieval.
```

### Example 2: Creating Quest from Bug Fix Discussion

**User Request**: "/create-quest - we need to fix the pagination issue in API queries"

**Expected Behavior**:
1. Capture: pagination bug details, affected systems, proposed fix
2. Assess confidence based on root cause clarity
3. Create quest with appropriate type (`integration` or `other`)
4. Generate focused requirements doc

## Response Format

When this skill activates, present results as:

```
QUEST CREATED

Quest: QUE-XXX
Name: [Quest Name]
Type: [solution_type] (legacy)
Domain: [domain] (QUE-085)
Work Type: [work_type] (QUE-085)
Autonomy: [autonomy_level] (computed)
Status: idea
Confidence: [score]%

CPQQRT Summary:
- Context: [1-line summary]
- Purpose: [1-line summary]
- Quantity: [1-line summary]
- Quality: [1-line summary]
- Resources: [1-line summary]
- Timeline: [1-line summary]

Requirements Doc: Uploaded to The Codex
   File: [name]-requirements.md
   Type: requirements

Links:
- Codex: http://localhost:3001/solutions/QUE-XXX
- AutoMem: Pattern stored for future retrieval

Next Steps:
1. Review requirements doc in The Codex
2. Use `/start QUE-XXX` when ready to begin development
3. Update confidence score as requirements clarify
```

## Integration Notes

This skill integrates with:
- **The Codex API**: Creates quest entry with full CPQQRT
- **AutoMem**: Stores quest pattern for context retrieval
- **ADLC Workflow**: Feeds into `/start` command for development kickoff
- **ADLC Workflow**: Quest feeds into `/start` command for development kickoff

---

*Create comprehensive quests from conversation context with 95% confidence documentation*
