---
name: grok-super
description: |
  Route prompts to Grok via browser automation on grok.com using Chrome CDP.
  Uses the authenticated Grok Business web UI when API access is unavailable.
  Triggers on: "/grok-super", "grok super", "browser grok", "grok ui"
allowed-tools:
  - Bash
  - Read
  - Write
user-invocable: true
---

# /grok-super - Grok via Chrome CDP

## Overview

Routes prompts to xAI Grok through the grok.com web UI using Chrome DevTools Protocol (CDP).
Uses `utils/grok_cdp.py` which connects to a Chrome debug profile on port 9222.

**VERIFIED WORKING**: 2026-02-17. Full round-trip tested — injection, submission, extraction all confirmed.

## Requirements

Chrome debug profile must be running:
```bash
grok-chrome   # alias in ~/.zshrc
# OR manually:
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --remote-debugging-port=9222 \
  --user-data-dir="$HOME/.chrome-debug-profile" \
  "https://grok.com"
```
User must be logged into Grok in that Chrome window. Login persists in profile.

## When This Skill Activates

**Trigger Keywords**: /grok-super, grok super, browser grok, grok ui

This skill applies when:
- User types `/grok-super [prompt]` to send a question to Grok via the web UI
- Used as Step 7 validation in `/research-deep` command

## When NOT to Trigger

Do NOT activate when user just mentions "grok" casually or no prompt is provided.

---

## Implementation Steps

### Step 1: Check Chrome is Running

```bash
curl -s http://localhost:9222/json/version 2>/dev/null | python3.13 -c "import sys,json; d=json.load(sys.stdin); print('Chrome ready:', d.get('Browser','?'))" || echo "ERROR: Chrome not on port 9222"
```

If not running, tell user to run `grok-chrome` in a terminal and log into Grok, then retry.

### Step 2: Send Prompt via grok_cdp.py

**For short/simple prompts:**
```bash
cd /Users/TehFiestyGoat/Development/personal-agent-hub
echo 'YOUR PROMPT HERE' | PYTHONPATH=. python3.13 utils/grok_cdp.py - 2>&1
```

**For long or complex prompts (research validation, code, special chars):**
```bash
cd /Users/TehFiestyGoat/Development/personal-agent-hub
PYTHONPATH=. python3.13 - << 'PYEOF'
import asyncio, sys
sys.path.insert(0, '.')
from utils.grok_cdp import send_to_grok

prompt = """
YOUR FULL MULTI-LINE PROMPT HERE
"""

response, completed = asyncio.run(send_to_grok(prompt, verbose=True))
if not completed:
    print("[PARTIAL - timed out]")
print(response)
PYEOF
```

### Step 3: What grok_cdp.py Does Internally

The script handles all of this automatically:
1. Connects to `http://localhost:9222` (Chrome debug port)
2. Finds the existing Grok tab's **browser context ID** (the auth context)
3. Creates a **new tab in that same browser context** — this is what gives it the auth cookies
4. Waits for grok.com to load and React to hydrate (~2-3s)
5. Finds `.tiptap.ProseMirror[contenteditable="true"]` and calls `editor.focus()` then `document.execCommand('insertText', false, prompt)`
6. Clicks `button[aria-label="Submit"]`
7. Polls for completion: waits for `button[aria-label="Stop model response"]` to disappear AND `.response-content-markdown` to appear
8. Extracts response via `.response-content-markdown` (last element)
9. Returns response text

**Why it creates a new tab instead of reusing the existing one:**
The existing Grok tab may be mid-conversation. A fresh tab starts a new conversation.

### Step 4: Present the Response

```markdown
## Grok Super Response

**Source**: grok.com (Grok Business)
**Model**: Auto (defaults to best available)
**Method**: CDP — port 9222, browser context AC1FBF86...

---

[Grok's response here, verbatim]

---

**Claude's Note**: [Agreement/disagreement/additional context]
```

---

## Verified Selectors (2026-02-17)

| Element | Selector | Notes |
|---------|----------|-------|
| Text editor | `.tiptap.ProseMirror[contenteditable="true"]` | Must call `.focus()` before `execCommand` |
| Submit button | `button[aria-label="Submit"]` | Disabled until text is inserted |
| Stop streaming | `button[aria-label="Stop model response"]` | Gone = response complete |
| Grok response content | `.response-content-markdown` | Last element = latest response |
| Grok message bubble | `.message-bubble` with `w-full max-w-none` | Alternative extraction target |
| User message bubble | `.message-bubble` with `max-w-[100%]` | Skip these when extracting |

**Completion detection** (both must be true):
- `button[aria-label="Stop model response"]` is GONE
- `.response-content-markdown` element count > 0

## Key Gotchas

1. **New tab doesn't inherit auth unless it's in the same browser context** — `grok_cdp.py` handles this automatically by reading the context ID from the existing Grok tab
2. **Tab URL changes after submission** — goes from `grok.com` to `grok.com/c/[uuid]` — grok_cdp.py retries tab lookup by target ID, not URL
3. **Never use MCP tools (`mcp__chrome-devtools__*`) on the debug Chrome for Grok** — they create tabs in a different browser context (no cookies)
4. **Cloudflare blocks login if CDP is active during auth** — always log in manually first, THEN use CDP

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Chrome not on port 9222` | Run `grok-chrome` in terminal |
| `No existing Grok tab found` | Navigate to grok.com in the debug Chrome and make sure you're logged in |
| Injection returns empty | Editor may not be loaded yet — increase `asyncio.sleep` before inject |
| Response extraction empty | Selectors may have changed — check `.response-content-markdown` exists |
| Timeout (120s) | Partial response still returned; Grok may be slow on long prompts |
| Cloudflare block during login | Stop all CDP activity, log in manually, then use grok_cdp.py |

## Setup

Debug Chrome profile at `~/.chrome-debug-profile` — cookies persist across restarts.
`grok-chrome` alias defined in `~/.zshrc` — run once per machine restart.

---

*Verified working 2026-02-17. Uses utils/grok_cdp.py via CDP WebSocket.*
