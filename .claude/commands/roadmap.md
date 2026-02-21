# /roadmap Command Protocol

## Purpose
Strategic planning and prioritization using GitHub Issues for ADLC Plan phase completion. Creates impact vs effort analysis with execution-ready roadmaps from tracked ideas.

## Usage
```bash
# Create quarterly roadmap (default)
claude /roadmap

# Create roadmap for specific timeframe via $1
claude /roadmap "sprint"
claude /roadmap "annual"

# Interactive mode (if $1 empty, uses default "quarterly")
claude /roadmap
```

## Argument Handling (v2.1.20+)

**NEW**: This command uses positional argument `$1` for input instead of `$ARGUMENTS` string parsing.
- **If `$1` provided**: Use directly (e.g., `/roadmap "sprint"` → `$1 = "sprint"`)
- **If `$1` empty**: Default to "quarterly" (optional argument with sensible default)
- **Both syntaxes work**: Backward compatible with `$ARGUMENTS` for legacy scripts

**Valid Timeframes**:
- `quarterly` (default) - 3-month strategic planning window
- `sprint` - 2-week tactical planning window
- `annual` - 12-month long-term strategic planning

## Protocol

### Input Detection (STEP 1)

Parse the positional argument for the timeframe:

```python
# Get optional timeframe from $1 (positional argument)
timeframe = $1 or "quarterly"  # Default to "quarterly" if not provided

# Validate timeframe is one of supported options
valid_timeframes = ["quarterly", "sprint", "annual"]
if timeframe.lower() not in valid_timeframes:
    print(f"Invalid timeframe: {timeframe}")
    print(f"Valid options: {', '.join(valid_timeframes)}")
    exit()

# Proceed with validated timeframe
timeframe = timeframe.lower()
```

### 1. Execute roadmap.sh Script
```bash
./scripts/roadmap.sh [timeframe]
```

### 2. Strategic Planning Workflow
- **Analyzes GitHub issues**: Reviews all open issues with 'idea' label
- **Categorizes automatically**: Groups by labels (bi-analytics, data-engineering, etc.)
- **Creates prioritization matrix**: Impact vs effort analysis framework
- **Generates execution plan**: Ready-to-build priorities with sequencing
- **Stakeholder alignment**: Templates for cross-departmental coordination

## Claude Instructions

When user runs `/roadmap [timeframe]`:

1. **Execute the script**: Run `./scripts/roadmap.sh [timeframe]`
2. **Monitor progress**: Display analysis and roadmap creation
3. **Guide completion**: Help user fill prioritization matrix if requested
4. **Suggest next steps**: Identify top priorities for `/build` command

### Response Format
```
🗺️  Creating [timeframe] roadmap from GitHub issues...
📊 Found X open ideas to analyze...
✅ Roadmap created: docs/roadmaps/[timeframe]-[date].md

📋 Next steps:
   1. Review and fill in the prioritization matrix
   2. Identify top 2-3 ideas for execution
   3. Build highest priority: ./scripts/build.sh <issue-number>

💡 Tip: Open the roadmap file to complete the prioritization analysis
🔗 View all ideas: gh issue list --label idea --state open
```

## Integration with ADLC
- **ADLC Plan Phase**: Strategic planning and stakeholder feedback
- **GitHub Issues integration**: All ideas visible and trackable by team
- **Impact analysis**: Business value vs implementation effort
- **Implementation planning**: Dependencies, sequencing, and resource allocation
- **Cross-layer context**: Links planning directly to development execution

## Prioritization Framework
### Impact vs Effort Matrix
- **High Priority**: High impact, low-medium effort (quick wins + strategic)
- **Medium Priority**: Medium impact, any effort OR high impact, high effort
- **Low Priority**: Low impact, any effort OR parking lot items

### Dependencies & Sequencing
- Technical prerequisites identification
- Cross-system coordination requirements
- Resource availability and timeline constraints

## Roadmap Output Structure
```markdown
# [Timeframe] Roadmap - [Month Year]

## Overview
Strategic planning session for [timeframe] execution priorities.

## Ideas Analysis

### Available Ideas (from GitHub Issues)

#### BI/Analytics Ideas
- [#59](url): Cross-tooling lineage visualization
- [#85](url): Executive KPI dashboard

#### Data Engineering Ideas
- [#86](url): Real-time customer data pipeline

[... other categories ...]

## Prioritization Framework
[Impact vs Effort matrix to fill in]

## Execution Plan
[Ready to build items]

## Quick Actions
- View all ideas: gh issue list --label idea --state open
- Build top priority: ./scripts/build.sh <issue-number>
```

## Examples

### Example 1: Quarterly Planning
```bash
claude /roadmap quarterly
# → Creates comprehensive quarterly execution plan from all open idea issues
```

### Example 2: Sprint Planning
```bash
claude /roadmap sprint
# → Creates 2-week focused execution plan from high-priority ideas
```

### Example 3: Annual Strategic Planning
```bash
claude /roadmap annual
# → Creates long-term strategic roadmap from all ideas
```

## Success Criteria
- [ ] Roadmap file created with all GitHub idea issues analyzed
- [ ] Ideas automatically categorized by labels
- [ ] Prioritization matrix template provided
- [ ] Clear execution plan with sequencing
- [ ] Dependencies and blockers identified
- [ ] Ready for immediate `/build <issue-number>` execution

## Follow-up Actions
After roadmap creation, typically:
1. **Review and prioritize**: Complete the impact vs effort analysis
2. **Select top priorities**: Choose 1-3 issues for immediate execution
3. **Execute**: Use `/build <issue-number>` for highest priority items
4. **Update GitHub**: Comment on issues with priority decisions

## Viewing and Managing Ideas

### View All Ideas
```bash
gh issue list --label idea --state open
```

### Filter by Category
```bash
gh issue list --label idea --label bi-analytics
gh issue list --label idea --label data-engineering
gh issue list --label idea --label architecture
```

### Search Ideas
```bash
gh issue list --label idea --search "dashboard"
```

---

*Strategic ADLC Plan phase completion - from GitHub-tracked ideas to execution-ready roadmap.*
