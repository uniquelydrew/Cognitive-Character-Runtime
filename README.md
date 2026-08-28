# Cognitive Character Runtime

A runnable MVP for a persistent synthetic-character architecture built from three independent cognitive roles:

- **Left** — factual consistency, constraints, causality, planning.
- **Right** — affect, association, social interpretation, tone.
- **Executive** — arbitration, action/speech selection, self-continuity, and post-interaction reflection.

The character is **not stored in the model weights**. Character identity, mutable state, beliefs, event history, self-history, and derived memory are external state managed by a dedicated memory service.

## Current vertical slice

The repository currently implements:

- Docker Compose bridge network.
- One stateless cognitive-worker container per cognitive role.
- Parallel Left/Right inference.
- Executive arbitration after both hemispheres return.
- Character primers loaded from YAML.
- Character switching in a browser UI.
- Persistent SQLite event/memory store using WAL mode.
- Append-only raw interaction history.
- Explicit self-history of character statements.
- Detection of repeated questions across interactions.
- Recognition of several paraphrases as one semantic topic in the bootstrap resolver.
- Explicit response awareness of repetition (for example, `Northbridge. You asked me that before.`).
- Post-interaction Executive reflection.
- Reflection retrieval of earlier interactions on the same resolved topic.
- Persistent event-to-event `revisits` links.
- Reflection idempotency until new conversational events occur.
- Mutation policy validation.
- Immutable runtime core identity/biography.
- Evidence requirements for belief and mutable-state revisions.
- Belief revision history rather than destructive replacement.
- Provenance on derived memories.
- Developer cognition panel exposing Left, Right, Executive, interaction classification, and reflection output.
- A deterministic mock backend so the whole architecture can be tested without downloading a model.
- An OpenAI-compatible backend adapter so each cognitive worker can be pointed at a local LLM server.

## Architecture

```text
                                  Browser UI
                                      |
                                      v
                               +--------------+
                               | Orchestrator |
                               +------+-------+
                                      |
                   +------------------+------------------+
                   |                  |                  |
                   v                  v                  v
             +-----------+      +-----------+      +-----------+
             |   Left    |      |   Right   |      | Executive |
             |  Worker   |      |  Worker   |      |  Worker   |
             +-----------+      +-----------+      +-----------+
                   ^                  ^                  ^
                   |                  |                  |
                   +------------------+------------------+
                                      |
                                      v
                               +--------------+
                               | Memory API   |
                               +------+-------+
                                      |
                                      v
                               +--------------+
                               | SQLite / WAL |
                               +--------------+
```

The browser never talks directly to a model worker. Model workers never directly access persistent storage.

## Interaction lifecycle

```text
user message
    |
    v
resolve topic / inspect prior interaction history
    |
    v
append raw user event
    |
    v
retrieve character state + relevant memories
    |
    +-------------------------+
    |                         |
    v                         v
Left inference           Right inference
    |                         |
    +------------+------------+
                 |
                 v
         Executive inference
                 |
                 v
       append character event
                 |
                 v
       append self-history
```

Left and Right run concurrently. Executive runs only after both are available.

## Reflection lifecycle

Reflection is a separate Executive mode and normally occurs when an interaction is ended.

```text
completed interaction
        |
        v
read immutable raw events
        |
        v
retrieve prior topic-related history
        |
        v
Executive reflection
        |
        v
mutation proposals
        |
        v
policy validator
   +----+-------------+----------------+
   |                  |                |
 allowed           versioned        rejected
   |                  |                |
   +---------+--------+                |
             |                         |
             v                         v
      derived memory             audit record
      belief revisions
      event links
      goal/state changes
```

The invariant is:

> Reflection may reinterpret and build on history, but it does not silently rewrite raw history.

## Memory categories

### Core character document

Loaded from the primer and considered immutable during runtime unless changed deliberately outside the character's normal cognition.

Examples:

```text
identity.name
identity.birthplace
identity.siblings
identity.occupation
biography
```

An Executive proposal using `update_core` is rejected by the memory service.

### Mutable state

Current transient or semi-persistent state. This is deliberately separate from the primer.

Examples:

```text
mood
trust_player
fear
anger
current_location
```

Runtime revisions require evidence and are audited.

### Beliefs

Character beliefs are allowed to be wrong. They are not the same as canonical facts.

A revision updates the current belief while preserving the previous value in `belief_history`.

### Raw events

Conversation turns, reflections, and later world events belong in an append-only event stream.

### Self-history

Character statements are persisted separately from canonical truth. This lets the runtime distinguish:

```text
canonical fact       character belief       previous self-commitment
```

That distinction is required to represent lies, mistakes, changing opinions, repeated answers, and explicit acknowledgement of earlier statements.

### Derived memories

Reflection can create summaries or other derived memories, but derived assertions retain source event IDs and an epistemic type.

Supported epistemic types currently include:

```text
fact
observation
self_statement
other_statement
belief
inference
suspicion
rumor
lie
unknown
```

## Event relationships

Reflection can build a graph over immutable events. The MVP currently proves the mechanism with `revisits` links when a completed interaction resolves to a topic that has prior history.

The table is intentionally generic enough to later support relationships such as:

```text
caused
contradicts
supports
resolves
supersedes
explains
revisits
retrospectively_relevant_to
```

This lets meaning change without rewriting the original event.

## Mutation policy

The Executive does not write to the database directly. It proposes typed operations:

```text
set_mutable_state
set_belief
add_goal
update_goal
add_memory
link_events
supersede_memory
update_core
```

The memory service validates each proposal and records the result in `mutation_audit`.

Current rules include:

```text
update_core                         -> rejected
set_belief without evidence        -> rejected
set_mutable_state without evidence -> rejected
belief/state revision with evidence-> versioned
inference memory without evidence  -> rejected
append-only sourced memory         -> allowed
```

This policy boundary is deterministic infrastructure rather than an LLM instruction.

## Repository layout

```text
.
├── characters/
│   ├── elena_voss.yaml
│   └── tomas_reed.yaml
├── services/
│   ├── common.py
│   ├── cognitive_worker/
│   │   ├── app.py
│   │   └── Dockerfile
│   ├── memory/
│   │   ├── app.py
│   │   └── Dockerfile
│   └── orchestrator/
│       ├── app.py
│       └── Dockerfile
├── tests/
│   ├── test_e2e.py
│   └── test_policy.py
├── ui/
│   ├── index.html
│   ├── nginx.conf
│   └── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── pyproject.toml
```

## Run with Docker Compose

The default configuration uses deterministic mock cognitive workers. No LLM download is required.

```bash
docker compose up --build
```

Then open:

```text
http://localhost:3000
```

The orchestrator API is exposed separately at:

```text
http://localhost:8080
```

The memory and cognitive-worker ports remain internal to the Docker bridge network.

## Character primers

A primer is data rather than model state. Example:

```yaml
id: elena_voss
identity:
  name: Elena Voss
  age: 42
  occupation: Harbormaster
  birthplace: Northbridge
  siblings: []
traits:
  cautious: 0.80
  patient: 0.62
  irritable: 0.28
cognition:
  left_weight: 0.67
  right_weight: 0.33
speech:
  verbosity: 0.35
  formality: 0.65
values:
  - duty
  - institutional_stability
initial_goals:
  - protect_port
mutable_state:
  mood:
    fear: 0.18
    trust_player: 0.50
beliefs:
  port_is_secure: false
biography: >-
  Elena has spent most of her adult life working around Northbridge's docks.
```

Adding another YAML file creates another selectable character after the memory service starts with a fresh database or is restarted and reloads primers.

Existing mutable state is not reset when the primer is reloaded.

## Repeated-question continuity

The bootstrap topic resolver recognizes several common identity questions. For example:

```text
Where were you born?
What's your hometown again?
Where are you from?
```

all resolve to:

```text
self.birthplace
```

The orchestrator queries prior interaction history before invoking the cognitive workers. The Executive therefore receives structured context like:

```json
{
  "interaction_type": "repeated_question",
  "topic": "self.birthplace",
  "prior_answer": "Northbridge",
  "times_asked": 2,
  "related_event_ids": ["evt_..."]
}
```

The repetition itself becomes part of the current cognitive context rather than relying on the language model to notice a distant transcript entry.

The deterministic topic resolver is intentionally temporary. A later milestone should replace it with a hybrid semantic resolver while retaining stable topic IDs.

## Using actual local models

Each cognitive worker supports `WORKER_BACKEND=openai_compatible` and can target a different OpenAI-compatible inference endpoint.

Environment variables are role-specific in Compose:

```text
LEFT_BACKEND
LEFT_MODEL_BASE_URL
LEFT_MODEL_NAME
LEFT_MODEL_API_KEY

RIGHT_BACKEND
RIGHT_MODEL_BASE_URL
RIGHT_MODEL_NAME
RIGHT_MODEL_API_KEY

EXEC_BACKEND
EXEC_MODEL_BASE_URL
EXEC_MODEL_NAME
EXEC_MODEL_API_KEY
```

Example `.env` shape:

```dotenv
LEFT_BACKEND=openai_compatible
LEFT_MODEL_BASE_URL=http://host.docker.internal:11434/v1
LEFT_MODEL_NAME=left-model
LEFT_MODEL_API_KEY=unused

RIGHT_BACKEND=openai_compatible
RIGHT_MODEL_BASE_URL=http://host.docker.internal:11434/v1
RIGHT_MODEL_NAME=right-model
RIGHT_MODEL_API_KEY=unused

EXEC_BACKEND=openai_compatible
EXEC_MODEL_BASE_URL=http://host.docker.internal:11434/v1
EXEC_MODEL_NAME=executive-model
EXEC_MODEL_API_KEY=unused
```

The adapter sends structured character/context state and requests a JSON object back from the model. No persistent state is kept inside the worker.

For the final local deployment, the inference endpoint can either run in each cognitive container or as a dedicated local inference service behind it. The API contract does not change.

## API outline

### Orchestrator

```text
GET  /health
GET  /characters
GET  /characters/{character_id}/state
POST /sessions
GET  /sessions/{session_id}/events
POST /sessions/{session_id}/chat
POST /sessions/{session_id}/reflect
POST /sessions/{session_id}/close
GET  /debug/{character_id}
```

### Cognitive worker

```text
GET  /health
POST /infer
```

The same container image runs all three roles using `COGNITIVE_ROLE=left|right|executive`.

### Memory service

The memory API owns characters, sessions, raw events, memories, mutation validation, beliefs, revision history, event links, goals, and debug state.

## Tests

Run locally:

```bash
python -m pytest -q
```

The integration test starts the complete service graph as local HTTP processes and verifies:

```text
first answer is stable
paraphrased repeat is detected
character explicitly acknowledges repetition
self-history is written
interaction close triggers reflection
reflection mutation is policy-validated
later interactions connect back to earlier topic history
reflection is idempotent
closed interactions reject additional chat
core mutation policy rejects runtime biography changes
```

Current result:

```text
5 passed
```

## Deliberately deferred

The MVP does not yet implement:

```text
embedding/vector retrieval
learned semantic topic resolution
full causal contradiction detection
belief extraction from arbitrary dialogue
automatic reflection after inactivity
memory salience decay
memory consolidation tiers
cross-character shared world memory
separate dialogue-realization model
world/action tool interface
game-engine integration
voice/TTS
model fine-tuning
GPU scheduling
```

Those should be added only after testing the three-role cognitive decomposition with real lightweight models.

## Next implementation milestones

1. Replace the bootstrap topic normalizer with stable semantic proposition IDs and embedding-assisted retrieval.
2. Define strict JSON Schemas for each cognitive role rather than accepting arbitrary model JSON.
3. Add contradiction detection between incoming claims, prior self-statements, beliefs, and canonical facts.
4. Expand reflection to create causal/support/contradiction/supersession graph edges.
5. Add typed belief extraction and confidence revision from completed interactions.
6. Add a world-state/tool boundary so the Executive can propose actions without directly mutating the simulation.
7. Run heterogeneous 1B-4B local models for Left, Right, and Executive and measure latency, coherence, contradiction rate, and repeat-answer stability against a single-model baseline.
