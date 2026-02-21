# Review Quest - Interactive Quest Refinement

## Purpose

Transform an existing Codex quest (QUE-XXX) from rough scope into fully scoped, documented work ready for `/start`. This skill enters **plan mode** for interactive, guided refinement.

**Key difference from `/idea`**:
- `/idea` = Create new quest from scratch
- `/review-quest` = Refine existing quest with locked scope, interactive phasing, requirements doc

## When to Use

Use `/review-quest` when:
- Quest exists in The Codex (QUE-XXX)
- Status is `idea`, `defining`, or `in_design` (pre-development)
- Need to scope phases, lock requirements, create detailed plan
- Goal is **confidence ≥90%** before `/start`

**DO NOT use** when:
- Quest is already in development (`in_development` status)
- Just want to raise confidence score (use `/improve-confidence`)
- Creating a brand new quest (use `/idea`)

## Input Pattern

```bash
/review-quest QUE-093                    # Review existing quest
/review-quest que-093                    # Case-insensitive
/review-quest 093                        # Short form (QUE- prefix added automatically)
```

## Workflow Overview

```
/review-quest QUE-093
    ↓
STEP 1: Fetch Quest + Enter Plan Mode
    (EnterPlanMode activated - read-only exploration)
    ↓
STEP 2: Create Task Tracking
    (TaskCreate for each major phase)
    ↓
STEP 3: Interactive Scoping
    (Ask ONE question at a time)
    (One question per user response)
    ↓
STEP 4: Build Requirements Doc
    (Accumulate responses in structured format)
    ↓
STEP 5: Sequential Thinking (if needed)
    (For complex phase decomposition or dependencies)
    ↓
STEP 6: Lock Scope
    (User confirms: "ready for /start")
    ↓
STEP 7: Update Quest + Upload Requirements
    (PATCH quest, upload requirements.md)
    ↓
STEP 8: Exit Plan Mode
    (Ready for /start QUE-093)
```

## Execution Steps

### STEP 1: Detect Input & Fetch Quest

**Input normalization:**
```python
input = "QUE-093" or "que-093" or "093"
if not input.startswith("QUE-"):
    input = f"QUE-{input.upper()}"
else:
    input = input.upper()

# Result: "QUE-093" (normalized, uppercase)
```

**Fetch from The Codex:**
```python
# Use direct API (always works)
import urllib.request, json, os

# No API key needed
api_key = None
url = f"http://localhost:8001/api/v1/solutions/slug/{input}"
req = urllib.request.Request(url, headers={})
with urllib.request.urlopen(req) as resp:
    quest = json.loads(resp.read().decode())
```

**Validate quest can be reviewed:**
```
If quest.status in ["in_development", "deployed", "retired"]:
    ❌ BLOCKED
    "This quest is already in development. Use /improve-confidence instead."

Else:
    ✅ Proceed to Step 2
```

### STEP 2: Enter Plan Mode + Create Task Tracking

**Enter plan mode:**
```
Call: EnterPlanMode
(This transitions to read-only, planning mode)
```

**Create task list for tracking phases:**
```python
# Main task for the review process
main_task = TaskCreate(
    subject="Review and Scope: " + quest.name,
    description=f"Interactive refinement of {quest.slug}. Lock requirements, create detailed requirements doc.",
    activeForm="Reviewing quest scope and requirements"
)

# Subtasks for each phase
phase_tasks = [
    TaskCreate(subject=f"PHASE 1: Critical Fixes", description="GitHub PR integration, Task Management, permissions"),
    TaskCreate(subject=f"PHASE 2: High-Value Features", description="New hooks, status line, prompt validation"),
    TaskCreate(subject=f"PHASE 3: Moderate Enhancements", description="Sandboxing, output styles, OTEL monitoring"),
    TaskCreate(subject=f"PHASE 4: Nice-to-Have", description="Company announcements, spinner verbs, analytics")
]
```

### STEP 3: Interactive Scoping - Ask ONE Question at a Time

**Core principle**: One question per response (per user rules).

**Question categories** (prioritized):
1. **Phase decomposition**: "How should we break this into phases?"
2. **Effort estimates**: "Which phase is most critical/urgent?"
3. **Acceptance criteria**: "What defines success for each phase?"
4. **Dependencies**: "Are there blockers or prerequisites?"
5. **Timeline**: "What's the target completion date?"
6. **Team/resources**: "Who needs to be involved?"

**Question format:**
```
📋 Review: QUE-093 - Claude Code Deep Alignment v2.1.19 to v2.1.29

[Current context/findings so far]

---

**Question 1:** [Your first question here]

[Provide context or options if helpful]

Waiting for your response...
```

**After user answers:**
- Record answer
- Update relevant CPQQRT field or phase detail
- Ask next question (adapted based on answer)
- Repeat until scope locked

### STEP 4: Build Requirements Doc Structure

**Accumulate user responses into structured doc:**

```markdown
# Requirements: [Quest Name]

**Quest**: QUE-093
**Created**: YYYY-MM-DD
**Status**: defining
**Confidence**: XX% (tracking as we go)

---

## ECHO CHECK

[One sentence summary - lock this with user]

---

## SCOPING DECISIONS

### Q1: [Question 1]
**A**: [User's answer]

### Q2: [Question 2]
**A**: [User's answer]

[... all Q&A recorded ...]

---

## PHASING & MILESTONES

### Phase 1: [Name]
**Scope**: [What's included]
**Effort**: [Estimate]
**Duration**: [Timeline]
**Blockers**: [If any]

### Phase 2: [Name]
...

---

## SUCCESS CRITERIA

- [ ] [Criterion 1]
- [ ] [Criterion 2]
- [ ] [Criterion 3]

---

## DEPENDENCIES & RISKS

### Dependencies
- [External blocker or dependency]
- [Prerequisites]

### Risks
- **[Risk 1]**: [Description & mitigation]
- **[Risk 2]**: [Description & mitigation]

---

## NEXT STEPS

- [ ] Confirm scope lock
- [ ] Assign to `/start QUE-093`

---

*Generated by /review-quest - YYYY-MM-DD*
```

### STEP 5: Sequential Thinking (If Needed)

**When to use sequential thinking:**
- Decomposing 4 phases from issue #496 + QUE-093's existing phases
- Understanding dependencies between phases
- Determining which phase(s) are CRITICAL vs optional
- Complex trade-off decisions

**Sequential thinking prompt:**
```
Use sequential thinking to reason through:
1. All 13 recommendations from issue #496 (2 CRITICAL + 4 RECOMMENDED + 4 OPTIONAL + 3 INFORMATIONAL)
2. QUE-093's existing 4 phases
3. How to merge and prioritize into a coherent Phase 1-4 structure
4. Dependencies between phases
5. What must complete before Phase 1 ends
```

### STEP 6: Lock Scope

**Present summary for user confirmation:**

```
═══════════════════════════════════════════════════════════════════

✅ SCOPE LOCKED

Quest: QUE-093 - Claude Code Deep Alignment v2.1.19 to v2.1.29

**Phases**:
  Phase 1 (CRITICAL): [Summary] - Duration: [time]
  Phase 2 (HIGH-VALUE): [Summary] - Duration: [time]
  Phase 3 (MODERATE): [Summary] - Duration: [time]
  Phase 4 (NICE-TO-HAVE): [Summary] - Duration: [time]

**Key Decisions**:
  - [Decision 1]
  - [Decision 2]
  - [Decision 3]

**Success Criteria**:
  - [Criterion 1]
  - [Criterion 2]
  - [Criterion 3]

═══════════════════════════════════════════════════════════════════

Ready to proceed?

  ✅ YES, lock scope and create requirements doc
  ❌ EDITS NEEDED (describe what to change)
  ❓ QUESTIONS (ask clarification)
```

**Waiting for confirmation before proceeding to STEP 7.**

### STEP 7: Update Quest + Upload Requirements

**Update quest fields in The Codex:**

```python
import urllib.request, json, os

# No API key needed
api_key = None
solution_id = quest['id']

# Update CPQQRT with locked scope
payload = {
    "context": "[refined context from Q&A]",
    "purpose": "[refined purpose]",
    "quantity": "[refined scope - all 4 phases]",
    "quality": "[success criteria]",
    "resources": "[team, tools, dependencies]",
    "timeline": "[phase timeline]",
    "confidence": "92",  # Raised from 85% after interactive refinement
    "status": "ready"  # Status moved to "ready" for /start
}

req = urllib.request.Request(
    f"http://localhost:8001/api/v1/solutions/{solution_id}",
    data=json.dumps(payload).encode(),
    method='PATCH',
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req) as resp:
    updated = json.loads(resp.read().decode())

print(f"✅ Quest updated: {updated['slug']}")
```

**Upload requirements.md:**

```bash
curl -X POST "http://localhost:8001/api/v1/solutions/{solution_id}/files" \
  \
  -F "file=@/tmp/requirements.md" \
  -F "file_type=requirements" \
  -F "description=Requirements document for QUE-093"
```

### STEP 8: Exit Plan Mode

**Summary:**

```
✅ Quest Review Complete!

📦 QUE-093: Claude Code Deep Alignment v2.1.19 to v2.1.29

**Status**: ready (was: defining)
**Confidence**: 92% (was: 85%)
**Requirements**: Locked and documented ✓

🔗 Next step: /start QUE-093

Ready to begin Phase 1 (Critical Fixes)?
```

**Mark tasks complete:**
```python
TaskUpdate(main_task_id, status="completed")
```

## Key Behaviors

### One Question at a Time
- Per user rule: Ask ONE question per response
- Do not bundle questions
- Adapt follow-ups based on answers

### Read-Only Exploration
- Use Read/Glob/Grep to explore codebase (existing configs, hooks, etc.)
- Do NOT edit files during plan mode
- Use only for understanding scope

### Task Tracking Throughout
- Create main task + phase subtasks at start
- Mark phases as `in_progress` as we discuss them
- Mark phases as `completed` when locked
- Final update at end

### Sequential Thinking for Complex Decisions
- When phasing gets complex (merging 13 recommendations + 4 existing phases)
- When dependencies unclear
- When trade-offs between phases need reasoning

### Build Requirements Doc Iteratively
- Start with empty structure
- Fill in sections as Q&A progresses
- Finalize before STEP 7
- Upload to The Codex as attachment

## Integration

### With EnterPlanMode
- Call `EnterPlanMode` at STEP 2
- Establishes read-only sandbox
- All work stays in `.claude/plans/` directory

### With TaskCreate/TaskUpdate
- Create task list for tracking phases
- Update status as review progresses
- Mark complete at end

### With ExitPlanMode
- Call at STEP 8 after requirements locked
- Stores plan in `.claude/plans/[quest-slug].md`
- Ready for `/start QUE-093`

## Requirements Precedence

This skill creates **detailed requirements** before `/start` is executed. The requirements doc serves as the single source of truth:

1. Requirements doc created in STEP 4 (accumulates Q&A)
2. User confirms scope lock in STEP 6
3. Requirements uploaded to Codex in STEP 7
4. `/start QUE-093` reads these requirements and begins Phase 1

## Error Handling

**Quest not found:**
```
❌ Quest not found: QUE-093
Check the slug and try again.
```

**Quest already in development:**
```
❌ Cannot review QUE-093 - already in development

This quest has status "in_development".
To refine an active quest, use: /improve-confidence QUE-093
```

**User interrupts mid-review:**
```
⏸️  Review paused: QUE-093

Current progress:
  ✅ Phase 1 scoped
  ✅ Phase 2 scoped
  ⏳ Phase 3 in progress
  ⭕ Phase 4 pending
  ⭕ Requirements doc pending

Resume with: /review-quest QUE-093
```

## Related Commands

- `/idea` - Create brand new quest
- `/improve-confidence` - Raise confidence on existing quest (use when requirements doc exists)
- `/start QUE-093` - Begin development after review-quest completes

---

*Interactive quest refinement with task tracking, sequential thinking, and comprehensive requirements documentation*
