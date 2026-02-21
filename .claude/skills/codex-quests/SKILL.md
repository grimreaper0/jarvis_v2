---
name: codex-quests
description: |
  Create, update, and delete quests in The Codex (FiestyGoat AI - localhost:3001).
  Use when user wants to create a new quest, update CPQQRT data, delete a quest, or manage quest metadata.
  Triggers on: "quest", "QUE-", "the codex", "create quest", "update quest", "delete quest", "CPQQRT"
allowed-tools:
  - Bash
  - Read
user-invocable: true
---

# Codex Quest Management

## Overview

Create and update quests in the The Codex. Quests represent business deliverables (reports, dashboards, data models, pipelines) tracked through their full lifecycle with CPQQRT documentation.

## When This Skill Activates

**Trigger Keywords**: quest, QUE-, the codex, create quest, update quest, CPQQRT

This skill applies when:
- User wants to create a new quest in the codex
- User wants to update an existing quest (by slug like QUE-028)
- User mentions CPQQRT fields (context, purpose, quantity, quality, resources, timeline)
- User says `/codex-quests` command

## Credentials

**Environment Variable**: `SC_API_KEY` (pre-loaded by `~/dotfiles/load-secrets-from-1password.sh`, cached at `~/.da-agent-hub-secrets-cache`)

**CRITICAL**: Do NOT call `op item get` for this key. It is already in the shell environment. Just use `$SC_API_KEY` directly. The Python scripts access it via `os.environ['SC_API_KEY']`.

**If missing**: Run `source ~/dotfiles/load-secrets-from-1password.sh` to refresh the cache.

## API Endpoint

**Base URL**: `http://localhost:8001/api/v1`
**Authentication**: `X-API-Key` header

## Owner Name Defaults

**Auto-Assignment**: When creating quests, the `owner_name` field is automatically populated from the `FIESTYGOAT_OWNER_NAME` environment variable (stored in `~/.da-agent-hub-secrets-cache`).

**Current User**: FiestyGoat AI (set 2026-02-02)

**How It Works**:
- Frontend (The Codex UI): Uses Azure AD (OIDC) for authentication, user identity known from JWT token
- Backend API: Uses system-level `X-API-Key` for authorization, no user identity
- **Workaround**: Environment variable provides default owner until QUE-082 (unified authentication) is implemented

**To Change**: Edit `~/.da-agent-hub-secrets-cache` and update `FIESTYGOAT_OWNER_NAME="Your Name"`

**Override**: Use `--owner-name` parameter when creating quests to override the default

**Future**: QUE-082 (Enterprise Role-Based Security Strategy) will enable JWT token pass-through so the API knows the authenticated user automatically.

## Subcommands

### create
Create a new quest in the codex.

**Syntax**: `/codex-quests create --name "Name" --description "Description" --type <type> --status <status> [options]`

**Required Parameters**:
- `--name`: Quest name (required)
- `--description`: Full description of the quest (required)
- `--type`: Quest type (required) - see Types table
- `--status`: Lifecycle status (required) - see Statuses table

**Optional Parameters**:
- `--domain`: Quest domain (see Domains table) - QUE-085 taxonomy
- `--work-type`: Work type (see Work Types table) - QUE-085 taxonomy
- `--context`: Business context and background (CPQQRT - C)
- `--purpose`: Why this quest exists (CPQQRT - P)
- `--quantity`: Scale/volume considerations (CPQQRT - Q)
- `--quality`: Quality requirements and SLAs (CPQQRT - Q)
- `--resources`: Team, budget, dependencies (CPQQRT - R)
- `--timeline`: Deadlines and milestones (CPQQRT - T)
- `--owner-name`: Owner's name
- `--team`: Team name
- `--tags`: Comma-separated tags
- `--freshservice-ticket-ids`: Comma-separated FreshService ticket IDs
- `--confidence`: Confidence level 0-100 (integer, for backlog prioritization)
- `--rank`: Priority rank 1+ (integer, 1=highest priority)

### get
Retrieve quest details by slug.

**Syntax**: `/codex-quests get <slug>`

### update
Update an existing quest.

**Syntax**: `/codex-quests update <slug> [options]`

All parameters are optional for update - only include what you want to change.

**Auto-Rerank Trigger**: When updating `--status` to `deployed` or `retired`, MUST automatically run `./scripts/rerank-solutions.sh --apply` after the update completes. This clears the rank from the completed quest and closes any gaps in the remaining backlog rankings.

### list
List quests with optional filters.

**Syntax**: `/codex-quests list [--status <status>] [--type <type>] [--domain <domain>] [--work-type <work_type>] [--limit <n>]`

### delete
Delete a quest from the codex.

**Syntax**: `/codex-quests delete <slug>`

**Warning**: This action is permanent and cannot be undone.

### upload-file
Upload a file attachment to a quest.

**Syntax**: `/codex-quests upload-file <slug> --file <path> [options]`

**Required Parameters**:
- `<slug>`: Quest slug (e.g., QUE-029)
- `--file`: Path to the file to upload

**Optional Parameters**:
- `--file-type`: Type of file (documentation, requirements, design, architecture, runbook, analysis, specification, report, other)
- `--description`: Description of the file
- `--uploaded-by`: User who uploaded the file

**Note**: Requires AWS_S3_BUCKET to be configured. Max file size: 100MB.

### list-files
List files attached to a quest.

**Syntax**: `/codex-quests list-files <slug>`

### delete-file
Delete a file attachment from a quest.

**Syntax**: `/codex-quests delete-file <slug> --file-id <uuid>`

### rerank
Clear ranks from deployed/retired quests and sequentially rerank the remaining backlog.

**Syntax**: `/codex-quests rerank`

**What it does**:
1. Finds all quests with status `deployed` or `retired` that still have a rank → clears rank to null
2. Takes remaining ranked quests, sorts by current rank, assigns sequential ranks starting from 1
3. Fixes any gaps or duplicates in the ranking

**Implementation**: Runs `./scripts/rerank-solutions.sh --apply`

**When it runs automatically**:
- After any `update` that changes status to `deployed` or `retired`
- After `/complete` sets a quest to deployed

**When to run manually**:
- After bulk status changes
- To fix ranking gaps from manual codex edits
- As a periodic cleanup

### link-pr
Link a GitHub PR to a quest. Appends to the quest's `github_prs` JSONB array (fetches existing PRs first to avoid overwriting).

**Syntax**: `/codex-quests link-pr <slug> --pr <number> --repo <owner/repo> [--title "..."] [--status open|merged|closed]`

**Required Parameters**:
- `<slug>`: Quest slug (e.g., QUE-110)
- `--pr`: PR number (e.g., 108)
- `--repo`: Repository in `owner/repo` format (e.g., `graniterock/MaestroAI`)

**Optional Parameters**:
- `--title`: PR title (auto-fetched from GitHub if omitted)
- `--state`: PR state — `open`, `merged`, or `closed` (default: `open`)

**Auto-Detection**: When creating a PR with `gh pr create` for quest work (branch contains `que-XXX`), Claude SHOULD automatically call `link-pr` after the PR is created. Don't wait for the user to ask.

```bash
python3 << 'PYEOF'
import urllib.request
import json
import os

# No API key needed for local Codex
api_key = None
slug = "QUE-110"  # Replace with actual slug
new_pr = {
    "pr_number": 108,  # Replace
    "number": 108,  # Include both for API + frontend compatibility
    "url": "https://github.com/graniterock/MaestroAI/pull/108",  # Replace
    "title": "feat(env-context): Add EnvironmentContextProvider (QUE-110)",  # Replace
    "repo": "graniterock/MaestroAI",  # Replace
    "state": "open"  # REQUIRED: "open", "merged", or "closed"
}

# Step 1: Get quest by slug to retrieve ID and existing PRs
url = f"http://localhost:8001/api/v1/solutions/slug/{slug}"
req = urllib.request.Request(url, headers={})
with urllib.request.urlopen(req) as resp:
    quest = json.loads(resp.read().decode())

solution_id = quest['id']
existing_prs = quest.get('github_prs', []) or []

# Step 2: Check for duplicate (same pr_number + repo)
already_linked = any(
    p.get('pr_number') == new_pr['pr_number'] and p.get('repo') == new_pr['repo']
    for p in existing_prs
)
if already_linked:
    print(f"PR #{new_pr['pr_number']} ({new_pr['repo']}) already linked to {slug}")
    exit(0)

# Step 3: Append new PR and PATCH
existing_prs.append(new_pr)
payload = {"github_prs": existing_prs}

url = f"http://localhost:8001/api/v1/solutions/{solution_id}"
req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode(),
    method='PATCH',
    headers={"Content-Type": "application/json"}
)

with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read().decode())

prs = result.get('github_prs', [])
print(f"PR linked to {slug}")
print(f"  #{new_pr['pr_number']}: {new_pr['title']}")
print(f"  Repo: {new_pr['repo']}")
print(f"  URL: {new_pr['url']}")
print(f"  Total PRs on quest: {len(prs)}")
PYEOF
```

### list-prs
List all GitHub PRs linked to a quest.

**Syntax**: `/codex-quests list-prs <slug>`

```bash
python3 << 'PYEOF'
import urllib.request
import json
import os

# No API key needed for local Codex
api_key = None
slug = "QUE-110"  # Replace with actual slug

url = f"http://localhost:8001/api/v1/solutions/slug/{slug}"
req = urllib.request.Request(url, headers={})
with urllib.request.urlopen(req) as resp:
    quest = json.loads(resp.read().decode())

prs = quest.get('github_prs', []) or []

if not prs:
    print(f"No PRs linked to {slug}")
else:
    print(f"PRs linked to {slug} ({len(prs)} total)")
    print()
    for pr in prs:
        status_icon = {"open": "🟢", "merged": "🟣", "closed": "🔴"}.get(pr.get('state', ''), '⚪')
        print(f"  {status_icon} #{pr.get('pr_number', '?')} [{pr.get('state', 'unknown')}]")
        print(f"     {pr.get('title', 'No title')}")
        print(f"     {pr.get('repo', 'unknown')} → {pr.get('url', 'N/A')}")
        print()
PYEOF
```

### unlink-pr
Remove a GitHub PR from a quest's `github_prs` array.

**Syntax**: `/codex-quests unlink-pr <slug> --pr <number> --repo <owner/repo>`

**Required Parameters**:
- `<slug>`: Quest slug (e.g., QUE-110)
- `--pr`: PR number to remove
- `--repo`: Repository in `owner/repo` format

```bash
python3 << 'PYEOF'
import urllib.request
import json
import os

# No API key needed for local Codex
api_key = None
slug = "QUE-110"  # Replace with actual slug
pr_number = 108    # Replace with PR number to remove
repo = "graniterock/MaestroAI"  # Replace with repo

# Step 1: Fetch current quest to get existing PRs and UUID
url = f"http://localhost:8001/api/v1/solutions/slug/{slug}"
req = urllib.request.Request(url, headers={})
with urllib.request.urlopen(req) as resp:
    quest = json.loads(resp.read().decode())

solution_id = quest['id']
existing_prs = quest.get('github_prs', []) or []

# Step 2: Filter out the matching PR
updated_prs = [
    pr for pr in existing_prs
    if not (pr.get('pr_number') == pr_number and pr.get('repo') == repo)
]

if len(updated_prs) == len(existing_prs):
    print(f"PR #{pr_number} from {repo} not found in {slug}")
else:
    # Step 3: PATCH with filtered array
    payload = {"github_prs": updated_prs}
    url = f"http://localhost:8001/api/v1/solutions/{solution_id}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        method='PATCH',
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode())

    removed = len(existing_prs) - len(updated_prs)
    remaining = len(updated_prs)
    print(f"Removed {removed} PR(s) from {slug}")
    print(f"Remaining linked PRs: {remaining}")
PYEOF
```

## File Types

| Type | Value | Description |
|------|-------|-------------|
| Documentation | `documentation` | General documentation |
| Requirements | `requirements` | Requirements documents |
| Design | `design` | Design documents |
| Architecture | `architecture` | Architecture diagrams/docs |
| Runbook | `runbook` | Operational runbooks |
| Analysis | `analysis` | Analysis documents |
| Specification | `specification` | Technical specifications |
| Report | `report` | Reports and analyses |
| Other | `other` | Other file types |

## Quest Types (Legacy)

| Type | Value | Description |
|------|-------|-------------|
| Data Model | `data_model` | dbt models, Snowflake tables/views |
| Pipeline | `pipeline` | ETL/ELT pipelines (Orchestra, Prefect, dlthub) |
| Dashboard | `dashboard` | Tableau dashboards |
| Report | `report` | Tableau reports, scheduled extracts |
| Application | `application` | Streamlit, React apps |
| Integration | `integration` | API integrations, data feeds |
| Data Dump | `data_dump` | One-time or scheduled data exports |
| AI | `ai` | AI/ML quests |
| Other | `other` | Anything else |

## Three-Dimensional Taxonomy (QUE-085)

Quests are now classified using three independent dimensions:

### Domains (What Area)

| Domain | Value | Description |
|--------|-------|-------------|
| AI | `ai` | AI/ML quests, LLM integrations |
| Data Model | `data_model` | dbt models, Snowflake tables/views |
| Pipeline | `pipeline` | ETL/ELT pipelines (Orchestra, Prefect, dlthub) |
| Dashboard | `dashboard` | Tableau dashboards |
| Report | `report` | Tableau reports, scheduled extracts |
| Application | `application` | Streamlit, React apps |
| Integration | `integration` | API integrations, data feeds |
| Infrastructure | `infrastructure` | Platform infrastructure, DevOps |
| Support | `support` | Support requests, service tickets |
| Other | `other` | Anything else |

### Work Types (What Kind of Work)

| Work Type | Value | Description |
|-----------|-------|-------------|
| New Feature | `new_feature` | Brand new capability |
| Bug Fix | `bug_fix` | Fixing defects |
| Enhancement | `enhancement` | Improving existing features |
| Maintenance | `maintenance` | Routine updates, dependency upgrades |
| Research | `research` | Investigation, POC, analysis |
| Documentation | `documentation` | Docs, runbooks, knowledge |
| Support | `support` | User support, troubleshooting |
| Infrastructure | `infrastructure` | Platform/infra work |

### Autonomy Level (Computed from Confidence)

| Level | Value | Confidence Range | Description |
|-------|-------|------------------|-------------|
| Autonomous | `autonomous` | ≥85% | Claude can work independently |
| Needs Pairing | `needs_pairing` | 60-84% | Requires collaboration |
| Needs Approval | `needs_approval` | <60% or null | Requires explicit approval |

**Note**: `autonomy_level` is computed, not stored. It appears in API responses but cannot be set directly.

## Auto-Categorization Guidelines

When creating a quest, Claude MUST automatically determine the correct `domain` and `work_type` based on the request:

### Domain Selection Logic

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

### Work Type Selection Logic

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

### Examples

| Request | Domain | Work Type |
|---------|--------|-----------|
| "Build a Tableau dashboard for sales KPIs" | `dashboard` | `new_feature` |
| "Fix the broken dbt model for inventory" | `data_model` | `bug_fix` |
| "Add filtering to the existing revenue dashboard" | `dashboard` | `enhancement` |
| "Investigate why Orchestra pipeline is slow" | `pipeline` | `research` |
| "Create MaestroAI multi-LLM routing" | `ai` | `new_feature` |
| "Update the Snowflake sync documentation" | `data_model` | `documentation` |
| "Help user understand their report data" | `report` | `support` |
| "Upgrade dbt to version 1.8" | `data_model` | `maintenance` |

### CRITICAL: Always Set Both Dimensions

When creating a quest, ALWAYS include both `domain` and `work_type` in the API payload:

```python
payload = {
    "name": "Revenue Dashboard Filters",
    "description": "Add department and region filters to the revenue dashboard",
    "solution_type": "dashboard",  # Legacy field
    "status": "idea",
    "domain": "dashboard",        # REQUIRED: What area
    "work_type": "enhancement",   # REQUIRED: What kind of work
    # ... other fields
}
```

## Quest Statuses

| Status | Value | Description |
|--------|-------|-------------|
| Idea | `idea` | Initial concept, not yet defined |
| Defining | `defining` | Requirements being gathered |
| Ready | `ready` | Ready for development |
| In Development | `in_development` | Actively being built |
| Testing | `testing` | In QA/UAT |
| Deployed | `deployed` | Live in production |
| Maintenance | `maintenance` | Ongoing support mode |
| Retired | `retired` | No longer active |
| Blocked | `blocked` | Stuck on dependency |
| On Hold | `on_hold` | Paused intentionally |

## API Patterns (Python)

### Create Quest

```bash
# SC_API_KEY is already in the environment (cached by dotfiles)

python3 << 'PYEOF'
import urllib.request
import json
import os

# No API key needed for local Codex
quest_name = "SOLUTION_NAME_HERE"  # Replace with actual name

# DUPLICATE DETECTION: Search for similar quests before creating
search_url = f"http://localhost:8001/api/v1/solutions/?search={urllib.parse.quote(quest_name)}&status=ready,defining,idea,in_development"
search_req = urllib.request.Request(search_url, headers={})

try:
    with urllib.request.urlopen(search_req) as resp:
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
        print(f"  - Type 'update {similar_quests[0]['slug']}' to modify existing quest instead")
        print("  - Type 'cancel' to abort")
        print()
        print(">>> Waiting for your decision...")
        # If user says "proceed": Continue with POST below
        # If user says "update QUE-XXX": Switch to update workflow
        # If user says "cancel": Exit

except Exception as e:
    # If search fails, proceed with creation
    print(f"ℹ️  Duplicate check skipped (search unavailable): {e}")

# If no duplicates OR user said "proceed", continue with creation:
url = "http://localhost:8001/api/v1/solutions/"

# REQUIRED fields for create
payload = {
    "name": quest_name,
    "description": "DESCRIPTION_HERE",
    "solution_type": "data_model",  # See Types table (legacy)
    "status": "idea"  # See Statuses table
}

# OPTIONAL fields - uncomment and fill as needed
# Three-dimensional taxonomy (QUE-085)
# payload["domain"] = "data_model"  # See Domains table
# payload["work_type"] = "new_feature"  # See Work Types table
# CPQQRT fields
# payload["context"] = "Business context..."
# payload["purpose"] = "Why this quest..."
# payload["quantity"] = "Scale considerations..."
# payload["quality"] = "Quality requirements..."
# payload["resources"] = "Team and dependencies..."
# payload["timeline"] = "Milestones..."
# Other fields
# AUTO-SET: Owner name from environment variable (default: current user)
owner_name = os.environ.get('FIESTYGOAT_OWNER_NAME')
if owner_name:
    payload["owner_name"] = owner_name
# payload["team"] = "FiestyGoat AI"
# payload["tags"] = ["tag1", "tag2"]
# payload["freshservice_ticket_ids"] = ["56743", "57181"]
# payload["confidence"] = 85  # Integer 0-100
# payload["rank"] = 1  # Integer, 1=highest priority

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode(),
    method='POST',
    headers={
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }
)

try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
    print(f"Quest created: {data['slug']}")
    print(f"Name: {data['name']}")
    print(f"Type: {data['solution_type']}")
    print(f"Status: {data['status']}")
    print(f"URL: http://localhost:3001/solutions/{data['slug']}")
except urllib.error.HTTPError as e:
    error = json.loads(e.read().decode())
    print(f"Error: {error.get('detail', str(e))}")
PYEOF
```

### Get Quest by Slug

```bash
# SC_API_KEY is already in the environment (cached by dotfiles)

python3 << 'PYEOF'
import urllib.request
import json
import os

# No API key needed for local Codex
slug = "QUE-028"  # Replace with actual slug

url = f"http://localhost:8001/api/v1/solutions/slug/{slug}"

req = urllib.request.Request(url, headers={})

try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())

    print(f"Quest: {data['slug']}")
    print(f"Name: {data['name']}")
    print(f"Type: {data['solution_type']}")
    print(f"Status: {data['status']}")
    print()
    print("Three-Dimensional Taxonomy (QUE-085):")
    print(f"  Domain: {data.get('domain', 'N/A')}")
    print(f"  Work Type: {data.get('work_type', 'N/A')}")
    print(f"  Autonomy Level: {data.get('autonomy_level', 'N/A')} (computed from confidence)")
    print()
    print(f"Description: {data.get('description', 'N/A')[:200]}...")
    print()
    print("CPQQRT:")
    print(f"  Context: {(data.get('context') or 'N/A')[:100]}...")
    print(f"  Purpose: {(data.get('purpose') or 'N/A')[:100]}...")
    print(f"  Quantity: {(data.get('quantity') or 'N/A')[:100]}...")
    print(f"  Quality: {(data.get('quality') or 'N/A')[:100]}...")
    print(f"  Resources: {(data.get('resources') or 'N/A')[:100]}...")
    print(f"  Timeline: {(data.get('timeline') or 'N/A')[:100]}...")
    print()
    print(f"URL: http://localhost:3001/solutions/{slug}")
except urllib.error.HTTPError as e:
    print(f"Error: {e.code} - Quest not found" if e.code == 404 else f"Error: {e}")
PYEOF
```

### Update Quest

```bash
# SC_API_KEY is already in the environment (cached by dotfiles)

python3 << 'PYEOF'
import urllib.request
import json
import os

# No API key needed for local Codex
slug = "QUE-028"  # Replace with actual slug

# First, get the solution ID from slug
url = f"http://localhost:8001/api/v1/solutions/slug/{slug}"
req = urllib.request.Request(url, headers={})

with urllib.request.urlopen(req) as response:
    solution = json.loads(response.read().decode())

solution_id = solution['id']

# Now update - include ONLY fields you want to change
payload = {
    # Uncomment and modify fields to update
    # "name": "Updated Name",
    # "description": "Updated description...",
    # "solution_type": "data_model",
    # "status": "defining",
    # Three-dimensional taxonomy (QUE-085)
    # "domain": "data_model",  # See Domains table
    # "work_type": "enhancement",  # See Work Types table
    # CPQQRT fields
    # "context": "Updated context...",
    # "purpose": "Updated purpose...",
    # "quantity": "Updated quantity...",
    # "quality": "Updated quality...",
    # "resources": "Updated resources...",
    # "timeline": "Updated timeline...",
    # Other fields
    # "owner_name": "Owner Name",
    # "team": "FiestyGoat AI",
    # "tags": ["tag1", "tag2"],
    # "confidence": 85,  # Integer 0-100
    # "rank": 1,  # Integer, 1=highest priority
    # GitHub PRs (JSONB array - REPLACES entire array, not append)
    # "github_prs": [{"pr_number": 123, "url": "https://github.com/...", "title": "...", "repo": "graniterock/repo", "state": "open"}],
}

url = f"http://localhost:8001/api/v1/solutions/{solution_id}"
req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode(),
    method='PATCH',
    headers={
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }
)

try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
    print(f"Quest updated: {data['slug']}")
    print(f"Name: {data['name']}")
    print(f"Status: {data['status']}")
    print(f"URL: http://localhost:3001/solutions/{slug}")
except urllib.error.HTTPError as e:
    error = json.loads(e.read().decode())
    print(f"Error: {error.get('detail', str(e))}")
PYEOF
```

### List Quests

```bash
# SC_API_KEY is already in the environment (cached by dotfiles)

python3 << 'PYEOF'
import urllib.request
import json
import os

# No API key needed for local Codex

# Optional filters
status = None  # e.g., "idea", "in_development", "deployed"
solution_type = None  # e.g., "data_model", "dashboard"
domain = None  # e.g., "ai", "data_model", "pipeline" (QUE-085)
work_type = None  # e.g., "new_feature", "bug_fix", "enhancement" (QUE-085)
page_size = 20

url = "http://localhost:8001/api/v1/solutions/?"
params = [f"page_size={page_size}"]
if status:
    params.append(f"status={status}")
if solution_type:
    params.append(f"solution_type={solution_type}")
if domain:
    params.append(f"domain={domain}")
if work_type:
    params.append(f"work_type={work_type}")
url += "&".join(params)

req = urllib.request.Request(url, headers={})

with urllib.request.urlopen(req) as response:
    data = json.loads(response.read().decode())

print(f"The Codex ({data['total']} total)")
print()
print("| Slug | Name | Type | Status |")
print("|------|------|------|--------|")
for s in data['items']:
    name = s['name'][:35] + "..." if len(s['name']) > 35 else s['name']
    print(f"| {s['slug']} | {name} | {s['solution_type']} | {s['status']} |")
print()
print("View all: http://localhost:3001")
PYEOF
```

### Delete Quest

```bash
# SC_API_KEY is already in the environment (cached by dotfiles)

python3 << 'PYEOF'
import urllib.request
import json
import os

# No API key needed for local Codex
slug = "QUE-029"  # Replace with actual slug

# First, get the solution ID from slug
url = f"http://localhost:8001/api/v1/solutions/slug/{slug}"
req = urllib.request.Request(url, headers={})

try:
    with urllib.request.urlopen(req) as response:
        solution = json.loads(response.read().decode())
except urllib.error.HTTPError as e:
    print(f"Error: Quest {slug} not found")
    exit(1)

solution_id = solution['id']
solution_name = solution['name']

# Delete the quest
url = f"http://localhost:8001/api/v1/solutions/{solution_id}"
req = urllib.request.Request(url, method='DELETE', headers={})

try:
    with urllib.request.urlopen(req) as response:
        pass  # 204 No Content on success
    print(f"Quest deleted: {slug}")
    print(f"Name: {solution_name}")
except urllib.error.HTTPError as e:
    print(f"Error deleting quest: {e.code}")
PYEOF
```

### Upload File to Quest

```bash
# SC_API_KEY is already in the environment (cached by dotfiles)

# Upload file using curl (multipart/form-data)
curl -X POST "http://localhost:8001/api/v1/solutions/{solution_id}/files" \
  \
  -F "file=@/path/to/your/file.pdf" \
  -F "file_type=requirements" \
  -F "description=Requirements document for QUE-029" \
  -F "uploaded_by=fiestygoat"
```

### List Quest Files

```bash
# SC_API_KEY is already in the environment (cached by dotfiles)

python3 << 'PYEOF'
import urllib.request
import json
import os

# No API key needed for local Codex
solution_id = "f2047372-979d-479b-a97a-f42a0abe6d6b"  # Replace with actual ID

url = f"http://localhost:8001/api/v1/solutions/{solution_id}/files"
req = urllib.request.Request(url, headers={})

try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())

    print(f"Files for quest ({data['total']} total)")
    print()
    for f in data['items']:
        print(f"- {f['filename']} (v{f['version']}, {f['file_type']})")
        print(f"  Size: {f['file_size_bytes']} bytes")
        if f.get('download_url'):
            print(f"  Download: {f['download_url'][:50]}...")
        print()
except urllib.error.HTTPError as e:
    print(f"Error: {e.code}")
PYEOF
```

### Get File Download URL

```bash
# SC_API_KEY is already in the environment (cached by dotfiles)

curl "http://localhost:8001/api/v1/solutions/{solution_id}/files/{file_id}/download?expiration=3600" \
  -H "X-API-Key: $SC_API_KEY"
```

## Response Formats

### Successful Create
```
Quest created: QUE-029
Name: Customer Data Integration
Type: integration
Status: idea
URL: http://localhost:3001/solutions/QUE-029
```

### Successful Get
```
Quest: QUE-028
Name: PlantDemand Deliveries
Type: data_model
Status: idea

Description: The rpt_hma_plant_demand report compares planned deliveries...

CPQQRT:
  Context: PlantDemand is a delivery scheduling system that maintains...
  Purpose: Sync PlantDemand customer records with JDE/Apex master...
  Quantity: 181,980 PlantDemand orders affected...
  Quality: Target: 100% customer match rate...
  Resources: PlantDemand Application Team: Implement customer sync...
  Timeline: Phase 1: PlantDemand app team adds JDE/Apex customer ID...

URL: http://localhost:3001/solutions/QUE-028
```

### Successful Update
```
Quest updated: QUE-028
Name: PlantDemand Deliveries
Status: defining
URL: http://localhost:3001/solutions/QUE-028
```

### Successful List
```
The Codex (28 total)

| Slug | Name | Type | Status |
|------|------|------|--------|
| QUE-028 | PlantDemand Deliveries | data_model | idea |
| QUE-027 | Sales Dashboard Refresh | dashboard | deployed |
| QUE-026 | Inventory ETL Pipeline | pipeline | in_development |
...

View all: http://localhost:3001
```

### Successful Delete
```
Quest deleted: QUE-029
Name: TEST - Delete Me
```

## Common Mistakes to Avoid

1. **Mistake**: Calling `op item get` for The Codex API key
   - **Why it's wrong**: Triggers 1Password credential prompts every time; the key is already cached in `$SC_API_KEY`
   - **Correct approach**: Use `$SC_API_KEY` from environment directly (loaded by `~/dotfiles/load-secrets-from-1password.sh`)

2. **Mistake**: Creating a quest without all required fields
   - **Why it's wrong**: API will reject the request
   - **Correct approach**: Always include name, description, solution_type, and status for create

3. **Mistake**: Using display names instead of enum values for type/status
   - **Why it's wrong**: API expects lowercase snake_case values
   - **Correct approach**: Use `in_development` not "In Development", `data_model` not "Data Model"

4. **Mistake**: Updating by slug instead of ID
   - **Why it's wrong**: PATCH endpoint requires solution ID (UUID)
   - **Correct approach**: First GET by slug to retrieve ID, then PATCH with ID

5. **Mistake**: Including empty optional fields
   - **Why it's wrong**: May overwrite existing data with empty strings
   - **Correct approach**: Only include fields you want to change in update payload

6. **Mistake**: Using strings for confidence/rank instead of integers
   - **Why it's wrong**: String "10" sorts before "2" lexicographically, breaking priority sorting
   - **Correct approach**: Always use integers: `"confidence": 85` not `"confidence": "85"`

## CPQQRT Framework

When documenting quests, use the CPQQRT framework:

| Letter | Field | Question to Answer |
|--------|-------|-------------------|
| C | Context | What is the business background? What problem exists? |
| P | Purpose | Why does this quest exist? What will it enable? |
| Q | Quantity | What is the scale? How many records/users/reports affected? |
| Q | Quality | What are the success criteria? What SLAs apply? |
| R | Resources | Who is responsible? What dependencies exist? |
| T | Timeline | What are the phases? When is each milestone? |

## Integration with ADLC

- FreshService tickets can be attached via `freshservice_ticket_ids`
- GitHub PRs are tracked via `github_prs` — use `link-pr` subcommand or include in `update` payload
- **Auto-linking**: When creating a PR for quest work (branch has `que-XXX`), Claude SHOULD call `link-pr` automatically after `gh pr create` succeeds
- Use `/start` to begin work on a quest (creates git branch + project scaffold)

---

*The Codex quest management for portfolio tracking and CPQQRT documentation*
