# /remember — Store a Memory in AutoMem

You are storing a memory for the user. The input to remember is:

**$ARGUMENTS**

Follow these steps precisely:

## Step 1: Parse the Input

Extract from the input:
- **Title**: A concise summary under 80 characters, written as a factual statement (not a command). Example: "Altair downgraded to 4.2.2 due to streamlit-mermaid dependency"
- **Content**: The full detail of what was said, preserving context and the "why" behind it. If the input is short, expand it with any relevant context from the current conversation.

## Step 2: Generate Tags

Analyze the content and generate 3-6 tags. Pull from:
- Technical terms (library names, tools, languages)
- Categories (dependency, architecture, bug, config, trading, revenue, infrastructure)
- Component names from the project (automem, chromadb, jarvis, learning_bot, etc.)
- Domain terms (if about trading, content, social media, etc.)

Tags should be lowercase, single words or hyphenated compounds.

## Step 3: Determine Priority

Infer priority from context clues:
- **urgent**: Blocking work, system broken, data loss risk
- **high**: Architectural decisions, dependency issues, money-related, security
- **normal**: General knowledge, observations, preferences (DEFAULT)
- **low**: Trivia, nice-to-know, cosmetic issues

## Step 4: Check for Related Memories

Search AutoMem for semantically related existing notes before storing:

```bash
python3.13 -c "
import sys
sys.path.insert(0, '.')
from utils.automem import AutoMem
mem = AutoMem()
results = mem.search_notes('SEARCH_QUERY', min_similarity=0.5, limit=5)
for r in results:
    print(f\"ID: {r.get('id', 'N/A')} | Sim: {r.get('similarity', 0):.2f} | Title: {r.get('title', 'N/A')} | Tags: {r.get('tags', [])}\")
if not results:
    print('No related memories found.')
"
```

Use the title you generated as the SEARCH_QUERY.

## Step 5: Store the Memory

Store via AutoMem:

```bash
python3.13 -c "
import sys
sys.path.insert(0, '.')
from utils.automem import AutoMem
mem = AutoMem()
note_id = mem.add_note(
    title='TITLE_HERE',
    content='CONTENT_HERE',
    priority='PRIORITY_HERE',
    tags=['tag1', 'tag2', 'tag3']
)
print(f'Note ID: {note_id}')
"
```

Replace placeholders with the actual parsed values. Escape single quotes in strings properly.

## Step 6: Confirm to User

Report back with:
- **Stored**: The title
- **Tags**: The generated tags
- **Priority**: The determined priority
- **Note ID**: The UUID returned from AutoMem
- **Related memories**: List any related notes found in Step 4 (title + similarity score), or say "None found" if empty
