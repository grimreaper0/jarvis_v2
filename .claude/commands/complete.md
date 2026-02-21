# /complete Command Protocol

## Purpose
Complete and archive projects with automated knowledge extraction, performance metrics tracking, and intelligent knowledge dissemination. Implements prompt-first ADLC project completion with intelligent knowledge preservation.

## Usage
```bash
# Complete and archive project
/complete feature-que-085_taxonomy           # Project name via $1
/complete feature-que-123_sales_dashboard

# Interactive mode (prompts if $1 empty)
/complete                                    # Prompts for project name
```

## Argument Handling (v2.1.20+)

**NEW**: This command uses positional argument `$1` for project name instead of `$ARGUMENTS` string parsing.
- **If `$1` provided**: Use directly (e.g., `/complete feature-que-085` → `$1 = "feature-que-085"`)
- **If `$1` empty**: Prompt user interactively for project name
- **Both syntaxes work**: Backward compatible with `$ARGUMENTS` for legacy scripts

## Required MCPs
**This command uses the following MCPs:**
- **automem** - Required for storing project context and creating associations
- **solutions-catalog-mcp** - Required for updating quest status in The Codex (D&A portfolio catalog)

**Note**: These MCPs are typically enabled by default. If missing, Claude will prompt you to enable them.

## Protocol

### 1. Project Analysis & Knowledge Extraction
**Claude analyzes completed project for:**
- **Technical Documentation**: Architecture patterns, implementation strategies
- **Agent Knowledge**: Tool-specific learnings and best practices
- **Process Insights**: Workflow improvements and organizational patterns
- **Integration Patterns**: Cross-system coordination strategies
- **Performance Metrics**: Agent effectiveness and routing intelligence

### 2. Automated Knowledge Dissemination
**Before archiving, Claude automatically:**
- **Extracts key patterns** from project specifications and findings
- **Updates agent knowledge bases** with tool-specific insights
- **Tracks performance metrics** for continuous improvement
- **Updates confidence scores** based on project outcomes
- **Preserves technical documentation** in knowledge directory
- **Creates cross-references** for future project reference

### 3. Complete Project Archival
- **Archives project**: Moves to `projects/completed/YYYY-MM/[project-name]/`
- **Extracts patterns**: Auto-populates memory system via finish.sh
- **Git workflow guidance**: Provides PR creation and merge options
- **Updates related ideas**: Links completion back to original archived ideas
- **Enables operations**: Project ready for ADLC Operate phase monitoring

## Claude Instructions

**MANDATORY: Use Sequential Thinking for Step Execution**

When executing `/complete`, you MUST use the `mcp__sequential-thinking__sequentialthinking` tool to walk through each step. This prevents missed steps like rank reprioritization.

**Sequential Thinking Template for /complete:**
```
Thought 1: "Starting /complete for [project]. Step 0: Quality evaluation - offer to user"
Thought 2: "Step 1: Analyze project - read spec.md, context.md, tasks/"
Thought 3: "Step 2: Execute knowledge updates (after approval)"
Thought 4: "Step 3: Store in AutoMem with associations (minimum 3-5)"
Thought 5: "Step 4: Git workflow guidance"
Thought 6: "Step 5: Handle related ideas"
Thought 7: "Step 6: The Codex - update status to deployed"
Thought 8: "Step 6.5: CRITICAL - Clear rank and reprioritize backlog"
Thought 9: "Final verification - all steps complete?"
```

Each thought should:
- State the step being executed
- Confirm completion before moving to next
- Flag any blockers or missing information

### Input Detection

**Detect and validate project name:**

```python
project_name = $1 or ""  # Get positional argument; empty if not provided

# If no input, prompt user interactively
if not project_name:
    project_name = prompt("Enter project name (e.g., feature-que-085_taxonomy): ").strip()

# Validate project exists
project_path = f"projects/{project_name}"
if not os.path.exists(project_path):
    # Try alternative locations
    if os.path.exists(f"projects/completed/{project_name}"):
        print(f"⚠️  Project already completed: {project_name}")
        exit()
    else:
        print(f"❌ Project not found: {project_name}")
        print(f"Valid projects in: projects/")
        exit()
```

When user runs `/complete [project-name]`:

### Step 0: Optional Quality Evaluation (Evaluator-Optimizer Pattern)
**BEFORE knowledge extraction, offer quality evaluation:**

1. **Prompt user**: "Would you like to run quality evaluation on deliverables? (y/n)"
   - If **yes**: Continue to quality evaluation workflow
   - If **no** or **skip**: Jump directly to Step 1 (Analyze Project)

2. **If evaluation requested**:
   a) **Identify deliverables** in project:
      - dbt models: Check `tasks/dbt-expert-findings.md`, spec.md, context.md
      - AWS infrastructure: Check `tasks/aws-expert-findings.md`
      - Dashboards: Check `tasks/tableau-expert-findings.md`
      - Code files: Check `tasks/*-findings.md` for code deliverables

   b) **List deliverables for evaluation**:
      ```
      📋 Deliverables identified for quality evaluation:

      dbt Models:
      - analytics.stg_customer_orders
      - analytics.fct_sales_daily

      AWS Infrastructure:
      - ALB OIDC authentication setup
      - ECS service configuration

      Dashboards:
      - Sales Performance Dashboard

      Proceed with evaluation? (y/n)
      ```

   c) **For each deliverable** (if user confirms):
      - Run evaluator-optimizer loop (max 3 iterations)
      - Use pattern from `.claude/memory/patterns/evaluator-optimizer-loop.md`
      - **Capture metrics**: Per-iteration metrics in `<evaluation_metrics>` block (see `.claude/memory/patterns/evaluation-metrics-tracking.md`)
      - Track quality score progression: initial → final
      - Document improvement delta and iteration count
      - **Save per-deliverable metrics**: `projects/<project>/metrics/evaluation/<deliverable>.json`

   d) **Evaluation summary**:
      ```
      📊 Quality Evaluation Complete:

      ✅ Deliverables Accepted (quality ≥0.8):
      - analytics.stg_customer_orders: 0.355 → 0.82 (2 iterations, +131%)
      - ALB OIDC authentication: 0.88 (1 iteration, excellent)

      ⚠️  Deliverables Need Work (quality <0.8):
      - Sales Dashboard: 0.58 → 0.71 (3 iterations, max reached)
        Recommendation: Fundamental redesign needed

      📈 Aggregate Quality Metrics:
      - Initial quality: 0.52 (across all deliverables)
      - Final quality: 0.81 (across all deliverables)
      - Average improvement: +67%
      - Total iterations: 7 across 3 deliverables
      - Acceptance rate: 67% (2/3 accepted)

      💾 Metrics saved to:
      - projects/<project>/metrics/evaluation/project-summary.json
      - Per-deliverable metrics in same directory

      How would you like to proceed?
      1. Fix issues now (continue iteration)
      2. Accept with caveats (document technical debt)
      3. Defer improvements (create follow-up issues)
      ```

   e) **User decision handling**:
      - **Fix now**: Continue evaluator-optimizer loop for flagged items
      - **Accept with caveats**: Document as technical debt in knowledge extraction
      - **Defer**: Create GitHub issues for improvements, continue completion

3. **After evaluation (or if skipped)**: Proceed to Step 1

### Step 1: Analyze Project and Propose Knowledge Changes
1. **Read project files**: spec.md, context.md, tasks/, README.md
2. **Identify extractable knowledge**:
   - Architecture patterns and technical decisions
   - Tool-specific insights for specialist agents
   - Process improvements and workflow learnings
   - Integration strategies and coordination patterns

3. **Present proposed changes BEFORE making them**:
   - List specific agent files to update with exact content additions
   - Identify new knowledge documents to create
   - Show proposed updates to README.md or CLAUDE.md if relevant
   - Request explicit approval: "Should I proceed with these knowledge updates?"

4. **WAIT for user approval** before making any changes

### Step 1.5: Extract Performance Metrics
**Track quantitative data for continuous improvement:**

1. **Agent invocation patterns**:
   - Count agent invocations by type (dbt-expert: 3, snowflake-expert: 2, etc.)
   - Document success/retry patterns for each agent
   - Measure estimated execution times

2. **Performance indicators**:
   - Success rate (completed without retries vs total attempts)
   - Task complexity indicators (simple query vs complex transformation)
   - Cross-agent coordination effectiveness

3. **Failure pattern documentation**:
   - Common error types encountered
   - Retry scenarios and resolution methods
   - Knowledge gaps identified during execution

4. **Confidence score updates**:
   - Identify successful patterns that should increase agent confidence
   - Document areas where agents struggled (decrease confidence)
   - Calculate confidence adjustments based on task outcomes

### Step 2: Execute Approved Knowledge Updates (Only After Approval)
**Check for updates to:**

#### Agent Knowledge (`/.claude/agents/`)
- **dbt-expert.md**: SQL patterns, model architectures, testing strategies + confidence updates
- **snowflake-expert.md**: Query optimization, cost management patterns + confidence updates
- **tableau-expert.md**: Dashboard design patterns, visualization strategies + confidence updates
- **data-architect-role.md**: System design patterns, data flow architectures + confidence updates
- **documentation-expert.md**: Documentation standards and templates + confidence updates
- **business-context.md**: Stakeholder management and requirement patterns + confidence updates
- **[other-agents].md**: Tool-specific insights and best practices + confidence updates

**Confidence Score Management:**
- Update agent confidence levels based on project outcomes
- Document successful patterns that warrant confidence increases (+0.05 to +0.15)
- Identify knowledge gaps that suggest confidence decreases (-0.05 to -0.10)
- Create routing recommendations for future similar tasks

#### Technical Documentation (`/knowledge/`)

**Production Application Knowledge** → `knowledge/applications/<app-name>/`
- **When**: Deploying new apps or major app updates
- **Structure**: Three-tier pattern (Tier 2 - comprehensive docs)
  - `architecture/` - System design, data flows, infrastructure details
  - `deployment/` - Complete deployment runbooks, Docker builds, AWS configuration
  - `operations/` - Monitoring, troubleshooting guides, incident response
- **Examples**: ALB OIDC authentication, ECS deployment patterns, multi-service Docker
- **Updates Required**:
  1. Create/update knowledge base docs for the application
  2. Update agent pattern index (e.g., aws-expert.md with confidence scores)
  3. Add to "Known Applications" in relevant role agents (e.g., ui-ux-developer-role.md)
  4. Create lightweight README in actual repo (Tier 1) linking to knowledge base

**Platform/Tool Patterns** → `knowledge/da-agent-hub/`
- **When**: Discovering reusable patterns for ADLC workflow
- **Structure**: Organized by ADLC phase (planning/, development/, operations/)
- **Examples**: Testing frameworks, git workflows, cross-system analysis patterns

**Three-Tier Documentation Principle**:
- **Tier 1**: Repository README (lightweight, < 200 lines, developer-focused)
- **Tier 2**: Knowledge base (comprehensive source of truth, unlimited size)
- **Tier 3**: Agent pattern index (pointers with confidence scores)

#### Memory System Updates (`/.claude/memory/`)
**Note**: Pattern extraction happens automatically via `finish.sh`:
- Extracts patterns marked with PATTERN:, SOLUTION:, ERROR-FIX:, etc.
- Saves to `.claude/memory/recent/YYYY-MM.md`
- No manual action needed - automatic during archival

### Step 3: Store Project in AutoMem with Associations

**CRITICAL**: Always create associations when storing projects in AutoMem to enable relationship queries and pattern discovery.

#### 3.1: Store Project Context
Use MaestroAI orchestrate to store to memory (routes to AutoMem REST API internally):

```json
{
  "content": "Complete project summary from context.md with all phases, decisions, and outcomes",
  "category": "project",
  "importance": 1.0,
  "metadata": {
    "project_name": "feature-name-TASK-123",
    "status": "completed",
    "completion_date": "YYYY-MM-DD",
    "start_date": "YYYY-MM-DD",
    "work_type": ["bi-analytics", "data-engineering", "analytics-engineering", "architecture", "ui-development"],
    "systems": ["dbt", "snowflake", "tableau", "orchestra", "react", "python"],
    "outcome": "success|deferred|cancelled",
    "github_pr": "#123"
  }
}
```

**Returns**: `memory_id` for creating associations

#### 3.2: Create Associations (MANDATORY)

**Search for related projects first**:
```bash
claude mcp call automem recall '{
  "query": "[project work types and systems]",
  "category": "project",
  "limit": 10
}'
```

**Create associations** using MaestroAI orchestrate (routes to AutoMem REST API):

**DEPENDS_ON relationships** (for foundation projects):
- Projects that this work directly relied upon
- Infrastructure or system improvements that enabled this work
- Example: Project depends on AutoMem infrastructure setup

**RELATES_TO relationships** (for similar projects):
- Projects with shared work types (both use analytics-engineering)
- Projects with shared systems (both use dbt + Snowflake)
- Projects with similar patterns or approaches
- Minimum 3-5 associations per project recommended

```json
{
  "from_memory_id": "[current project memory_id]",
  "to_memory_id": "[related project memory_id]",
  "relationship_type": "RELATES_TO|DEPENDS_ON",
  "properties": {
    "reason": "Shared work types: analytics-engineering, data-engineering",
    "from_project": "current-project-name",
    "to_project": "related-project-name"
  }
}
```

**Why associations matter**: Enables queries like "Find projects related to this work" and "What projects did this depend on?" - critical for pattern discovery and knowledge navigation.

#### 3.3: Remove from Filesystem and Sync Repos

**After AutoMem storage confirmed**, clean up the project sandbox:

1. **Remove git worktree** (if project is a worktree):
   ```bash
   git worktree remove projects/[project-name] --force
   ```

2. **Delete project directory** (if not a worktree or removal failed):
   ```bash
   rm -rf projects/[project-name]/
   ```

3. **Sync all repositories** to ensure clean state:
   ```bash
   ./scripts/pull-all-repos.sh
   ```

4. **Display cleanup summary**:
   ```
   🧹 Project Cleanup Complete:
      ✅ Worktree removed: projects/[project-name]/
      ✅ Repositories synced via pull-all-repos.sh
      ✅ Context preserved in AutoMem
   ```

**Result**: Clean filesystem, complete context preserved in AutoMem with discoverable relationships

### Step 4: Git Workflow Guidance
**Provide branch-aware options:**
- **Feature branch**: Recommend PR creation for review
- **Main branch**: Confirm direct merge readiness
- **Stay on branch**: Option to continue working

### Step 5: Handle Related Ideas (If Any)
- **Search for source ideas**: Look for original idea that led to this project
- **Handle based on what's found**:
  - **If source idea exists**: Move to archive and update with completion status
  - **If no source idea**: Note as ad-hoc project (no idea cleanup needed)
  - **If orphaned ideas found**: Clean up any related unarchived ideas
- **Cross-reference completion**: Maintain idea → project → completion traceability when applicable
- **Clean up workflow**: Ensure no orphaned ideas remain in inbox/organized

### Step 6: The Codex Integration

When completing a project, update the associated Codex entry:

#### 1. Find or Create Codex Entry

**Search for existing quest by project name**:
```python
# Find existing quest by name
solution = mcp__solutions_catalog__search_solutions(
    query=project_name
)

# If not found, create new entry with three-dimensional taxonomy (QUE-085)
# Note: MCP tool names use legacy "solution" naming - this is expected
if not solution.get('solutions'):
    solution = mcp__solutions_catalog__create_solution(
        name=project_name,
        description=project_summary,
        status="deployed",
        # Three-dimensional taxonomy (QUE-085) - see /start for selection logic:
        domain=detected_domain,      # ai, data_model, pipeline, dashboard, report, application, integration, infrastructure, support, other
        work_type=detected_work_type # new_feature, bug_fix, enhancement, maintenance, research, documentation, support, infrastructure
    )
else:
    solution_id = solution['solutions'][0]['id']
```

#### 2. Update Quest Status to Deployed

```python
mcp__solutions_catalog__update_solution(
    solution_id=solution_id,
    status="deployed",  # Update from draft/in_development to deployed
    description=updated_description,  # Include completion summary
    quantity=quantity_metrics,    # CPQQRT - Final metrics
    quality=quality_outcomes,     # CPQQRT - Quality achievements
    resources=resources_used,     # CPQQRT - Resources consumed
    timeline=timeline_actual,     # CPQQRT - Actual timeline
    # Three-dimensional taxonomy (QUE-085) - set if not already populated:
    domain=detected_domain,       # Only if quest domain is null/other
    work_type=detected_work_type  # Only if quest work_type is null
)
```

**Note**: Only update `domain` and `work_type` if they weren't set during `/start`. Use the same selection logic from `/start` to auto-detect values based on project content.

#### 3. Display Codex Update

```
📦 Codex Updated:
   URL: http://localhost:3001/solutions/{slug}
   Status: deployed ✅
   CPQQRT: Updated with completion metrics
```

### Step 6.5: Rank Reprioritization (Automatic)

After setting a quest to deployed in Step 6, run the rerank script:

```bash
./scripts/rerank-solutions.sh --apply
```

This script:
1. Clears rank from any deployed/retired quest that still has one
2. Sequentially reranks the remaining backlog (closes gaps, fixes duplicates)

**Note**: This same script is also triggered by the solutions-catalog skill whenever a status update to deployed/retired is made outside of `/complete`. The script is idempotent — running it when rankings are clean produces no changes.

## Response Format

### Phase 0: Optional Quality Evaluation (If Requested)
```
🔍 Starting /complete for project: [project-name]

🎯 Quality Evaluation Available

Before knowledge extraction, would you like to evaluate deliverable quality using the Evaluator-Optimizer pattern?

Benefits:
- Systematic quality assessment (0.0-1.0 scoring)
- Iterative improvement (up to 3 iterations per deliverable)
- Identifies issues before production deployment
- Tracks quality metrics for continuous learning

⏱️  Estimated time: 2-5 minutes per deliverable

Would you like to run quality evaluation? (y/n)
```

**If user responds 'y':**
```
📋 Scanning project for deliverables...

Deliverables identified for quality evaluation:

dbt Models (2):
- analytics.stg_customer_orders (from tasks/dbt-expert-findings.md)
- analytics.fct_sales_daily (from spec.md)

AWS Infrastructure (1):
- ALB OIDC authentication setup (from tasks/aws-expert-findings.md)

Dashboards (1):
- Sales Performance Dashboard (from context.md)

Code (1):
- React sales journal components (from tasks/react-expert-findings.md)

Total: 5 deliverables

Proceed with evaluation? (y/n)
```

**If user confirms, for each deliverable:**
```
🔬 Evaluating: analytics.stg_customer_orders

[Iteration 1]
Running evaluator-role assessment...

<evaluation_reasoning>
[Full chain-of-thought evaluation from evaluator-role]
</evaluation_reasoning>

<evaluation_result>
**Overall Quality Score**: 0.355 (Needs Improvement)
**Criteria Breakdown**: [scores with evidence]
**Improvement Feedback**: [prioritized CRITICAL/HIGH/MEDIUM items]
**Iteration Recommendation**: CONTINUE (1/3)
</evaluation_result>

[Iteration 2]
Refining based on feedback...
Specialist: dbt-expert implementing improvements...

<evaluation_result>
**Overall Quality Score**: 0.82 (Good)
**Quality Improvement**: +131% (0.355 → 0.82)
**Iteration Recommendation**: ACCEPT (production-ready)
</evaluation_result>

✅ analytics.stg_customer_orders: ACCEPTED (0.82 quality, 2 iterations)
```

**After all evaluations:**
```
📊 Quality Evaluation Summary:

✅ Deliverables Accepted (quality ≥0.8): 4/5
- analytics.stg_customer_orders: 0.355 → 0.82 (2 iter, +131%)
- analytics.fct_sales_daily: 0.91 (1 iter, excellent)
- ALB OIDC authentication: 0.88 (1 iter, excellent)
- React components: 0.67 → 0.84 (3 iter, +25%)

⚠️  Deliverables Below Threshold (<0.8): 1/5
- Sales Dashboard: 0.58 → 0.71 (3 iter, max reached)
  Issues: Performance (0.6), Data accuracy (0.65)
  Recommendation: Fundamental redesign with Tableau extract optimization

📈 Average Quality Metrics:
- Initial quality: 0.52 (across all deliverables)
- Final quality: 0.81 (across all deliverables)
- Average improvement: +56%
- Total iterations: 10 across 5 deliverables

How would you like to proceed with Sales Dashboard?
1. Fix now (continue evaluation/iteration)
2. Accept with caveats (document as technical debt)
3. Defer improvements (create GitHub issue for follow-up)
```

**If user skips evaluation:**
```
⏭️  Skipping quality evaluation, proceeding to knowledge extraction...
```

### Phase 1: Analysis and Proposal
```
🔍 Analyzing project: [project-name]
📊 Extracting performance metrics...

📈 Project Performance Summary:
   • Agents invoked: 5 (dbt-expert: 3, snowflake-expert: 2)
   • Success rate: 100% (0 retries needed)
   • Estimated execution time: 18 minutes
   • Task complexity: Medium (cross-system integration)
   • New patterns discovered: 3

🎯 Confidence Updates:
   ↗️ dbt-expert: +0.10 (incremental model optimization)
   ↗️ snowflake-expert: +0.05 (query performance tuning)
   ➡️ tableau-expert: No change (limited involvement)

📚 Identifying knowledge for preservation...

💡 Proposed Knowledge Updates:

### Agent Files to Update:
📝 .claude/agents/data-architect-role.md
   + GitHub Actions automation patterns section
   + AI-powered workflow design best practices
   + Confidence: +0.08
   + [show exact content additions]

📝 .claude/agents/dbt-expert.md
   + Incremental model optimization patterns
   + Confidence: +0.10
   + [specific additions with exact content]

### New Knowledge Documents:
📄 knowledge/applications/[app-name]/ (if deploying new app)
   + architecture/system-design.md - System architecture and data flows
   + deployment/production-deploy.md - Complete deployment runbook
   + operations/troubleshooting.md - Monitoring and incident response
   + Three-tier pattern integration:
     - Update aws-expert.md pattern index (Tier 3) with confidence scores
     - Add to ui-ux-developer-role.md Known Applications section
     - Create lightweight README in app repo (Tier 1) linking to knowledge base

📄 knowledge/da-agent-hub/[new-pattern].md (if platform improvement)
   + [document purpose and key content outline]

### Memory Extraction (Automatic):
🤖 finish.sh will automatically extract:
   - 3 PATTERN markers from task findings
   - 2 SOLUTION markers
   - 1 ERROR-FIX marker
   → Saved to memory/recent/YYYY-MM.md

🤔 **Should I proceed with these knowledge updates?**
   - Type 'yes' to execute all proposed changes
   - Type 'modify' to adjust specific updates
   - Type 'skip' to complete project without knowledge updates
```

### Phase 2: Execution (After Approval)
```
✅ Executing approved knowledge updates...

💡 Knowledge Updates Applied:
   ✅ Updated: agents/data-architect-role.md (integration patterns + confidence: +0.08)
   ✅ Updated: agents/dbt-expert.md (incremental model patterns + confidence: +0.10)
   ✅ Updated: agents/documentation-expert.md (process standards + confidence: +0.03)
   ✅ Added: knowledge/applications/[app-name]/ (three-tier docs for production app)
      - architecture/system-design.md, deployment/production-deploy.md, operations/troubleshooting.md
      - Updated aws-expert.md pattern index + ui-ux-developer-role.md Known Applications
   ✅ Added: knowledge/da-agent-hub/[new-pattern].md (platform improvement)

📦 Archiving project...
   ✅ Stored in AutoMem: memory_id [uuid]
   ✅ Associations created: 3 relationships
   🧹 Pattern extraction: 6 patterns saved to memory/recent/

🧹 Project Cleanup:
   ✅ Worktree removed: projects/[project-name]/
   ✅ Repositories synced via pull-all-repos.sh

📦 Codex Updated:
   ✅ URL: http://localhost:3001/solutions/[slug]
   ✅ Status: deployed
   ✅ CPQQRT: Completion metrics recorded

🔢 Backlog Reprioritized:
   (output from ./scripts/rerank-solutions.sh --apply)

🔀 Git workflow options:
   1. Create PR: gh pr create --title "Complete [project-name]" --body "Project completion with knowledge extraction"
   2. Merge to main: git checkout main && git merge [branch]
   3. Stay on branch: Continue working

💡 Recommended: Create PR for review

🤖 Routing Recommendations for Future Projects:
   • For incremental model work: Prefer dbt-expert (confidence: 0.92)
   • For query optimization: dbt-expert + snowflake-expert (high coordination success)
   • For cross-system integration: data-architect-role → dbt-expert → snowflake-expert (proven sequence)

🔗 Related ideas handled:
   ✅ Source idea found and archived: ideas/[location]/[idea-file] → ideas/archive/
   OR
   💡 No source idea found - ad-hoc project (no cleanup needed)

✅ Project '[project-name]' completed with knowledge preserved and metrics tracked!

🎉 Next steps:
   - Review performance metrics and confidence updates
   - Review completed work and extracted knowledge
   - Create PR if on feature branch
   - Plan next project: ./scripts/build.sh [idea-name]
```

## Knowledge Extraction Criteria

### Agent Knowledge Updates
**Update agent files when project contains:**
- **New tool patterns**: Implementation strategies specific to each tool
- **Best practices**: Proven approaches for tool configuration/usage
- **Integration patterns**: How tools coordinate with other systems
- **Troubleshooting insights**: Common issues and resolution patterns
- **Performance optimizations**: Efficiency improvements and cost management
- **Confidence adjustments**: Success/failure patterns affecting agent reliability

### Technical Documentation
**Add to knowledge/ when project demonstrates:**
- **Production applications** (`knowledge/applications/<app-name>/`): New deployments or major app updates
  - Follow three-tier pattern: Tier 2 comprehensive docs (architecture/, deployment/, operations/)
  - Update agent pattern indexes (Tier 3) and Known Applications in role agents
  - Create lightweight repo README (Tier 1) linking to knowledge base
- **System architecture**: Novel integration or design patterns
- **Process improvements**: Workflow enhancements worth preserving
- **Standards evolution**: Updated team practices and conventions
- **Cross-system coordination**: Multi-tool orchestration patterns

### Performance Metrics Documentation
**Track and update when project reveals:**
- **Agent effectiveness patterns**: Which agents excel at specific tasks
- **Coordination strategies**: Successful multi-agent workflows
- **Failure modes**: Common pitfalls and prevention strategies
- **Routing intelligence**: Optimal agent selection for task types

### Dissemination Decision Framework
- **High impact + high confidence**: Core system changes → Update multiple agent files + increase confidence
- **Tool-specific + proven**: Single tool insights → Update relevant agent only + confidence boost
- **Process innovation**: Workflow improvements → Update knowledge/ + document success patterns
- **Team learning**: Collaborative insights → Update da_team_documentation/ + share metrics
- **Failed experiments**: Document what didn't work → Decrease confidence + capture lessons

## Integration with ADLC & Memory System
- **ADLC Deploy Completion**: Final deployment with knowledge preservation and metrics
- **ADLC Operate Transition**: Project ready for operations with documented patterns and performance data
- **ADLC Observe Setup**: Knowledge base + metrics enable better monitoring and issue resolution
- **Cross-layer context**: Full traceability from idea to operations with preserved learnings
- **Memory System**: Automatic pattern extraction via finish.sh populates `.claude/memory/recent/`
- **Confidence Routing**: Performance metrics inform future agent selection and coordination

## Success Criteria

### Quality Evaluation (Optional)
- [ ] **Optional quality evaluation offered** before knowledge extraction
- [ ] **Deliverables identified and listed** when evaluation requested
- [ ] **Evaluator-optimizer loop executed** for each deliverable (max 3 iterations)
- [ ] **Quality scores tracked**: initial → final with improvement delta
- [ ] **Per-iteration metrics captured**: `<evaluation_metrics>` blocks with timestamp, quality score, criteria scores, feedback
- [ ] **Per-deliverable metrics saved**: `projects/<project>/metrics/evaluation/<deliverable>.json`
- [ ] **Project aggregate metrics saved**: `projects/<project>/metrics/evaluation/project-summary.json`
- [ ] **System-level metrics appended**: `.claude/memory/metrics/evaluation/<year>-<month>.json`
- [ ] **Evaluation summary provided** with acceptance recommendations and aggregate metrics
- [ ] **User decision captured** for below-threshold deliverables (fix/accept/defer)

### Knowledge Extraction & Performance
- [ ] Project knowledge automatically extracted and preserved
- [ ] Performance metrics tracked and analyzed (including quality metrics if evaluated)
- [ ] Agent confidence scores updated based on outcomes
- [ ] Relevant agent files updated with new insights + confidence adjustments
- [ ] Technical documentation created when warranted
- [ ] Memory system populated with patterns (automatic via finish.sh)
- [ ] Routing recommendations generated for future projects

### Project Archival & AutoMem Storage
- [ ] **Project stored in AutoMem** with rich metadata (work_type, systems, outcome, dates)
- [ ] **Associations created (MANDATORY)**: Minimum 3-5 relationships per project
  - [ ] DEPENDS_ON relationships created for foundation/infrastructure projects
  - [ ] RELATES_TO relationships created for projects with shared work types or systems
  - [ ] Association properties include reason and project names
- [ ] **AutoMem storage verified**: memory_id returned and associations confirmed
- [ ] **Worktree removed**: `git worktree remove projects/[project-name] --force`
- [ ] **Project directory removed** from filesystem after AutoMem storage
- [ ] **Repositories synced**: `./scripts/pull-all-repos.sh` executed
- [ ] Git workflow guidance provided based on current branch
- [ ] Related archived ideas updated with completion status
- [ ] Clear next steps for continued development cycle

### The Codex Integration
- [ ] Existing quest found by project name or slug
- [ ] If not found, new quest entry created in The Codex
- [ ] Quest status updated to "deployed"
- [ ] CPQQRT fields updated with completion metrics (quantity, quality, resources, timeline)
- [ ] Quest URL displayed in completion summary
- [ ] **Rank reprioritization**: Run `./scripts/rerank-solutions.sh --apply` to clear deployed ranks and close gaps

---

## Final Verification Checkpoint (MANDATORY)

Before declaring `/complete` finished, use sequential thinking to verify ALL steps:

```
Final Thought: "Verification checkpoint - checking all steps completed"

☐ Step 0: Quality evaluation offered (y/n recorded)
☐ Step 1: Project analyzed, knowledge changes proposed
☐ Step 2: Knowledge updates executed (or skipped with approval)
☐ Step 3: AutoMem storage confirmed (memory_id returned)
☐ Step 3.1: Associations created (minimum 3-5 relationships)
☐ Step 3.2: Project directory removed from filesystem
☐ Step 4: Git workflow guidance provided
☐ Step 5: Related ideas handled
☐ Step 6: Codex updated to "deployed"
☐ Step 6.5: Rank cleared AND backlog reprioritized ← CRITICAL, often missed

If ANY checkbox is unchecked → STOP and complete that step before finishing.
```

**Why this matters**: Missing Step 6.5 leaves stale ranks in the backlog, breaking priority ordering. Missing Step 3.1 (associations) leaves orphaned memories without relationships.

---

*ADLC project completion with intelligent knowledge preservation, performance tracking, and confidence-based routing - from active development to operational wisdom with compound learning.*