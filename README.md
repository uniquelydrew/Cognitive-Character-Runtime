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
- Source-controlled Left/Right priorities that enforce lobe attention budgets and
  an auditable Executive arbitration plan after both hemispheres return.
- Character primers loaded from YAML.
- Character switching in a browser UI.
- A separate Profile Studio for creating, viewing, editing, importing, and
  exporting validated YAML character snapshots.
- A profile-side snapshot comparison view that reports source and runtime diffs
  before a restore.
- A separate source-controlled, hierarchical general-knowledge corpus with
  label-indexed retrieval and character-derived access subsets.
- Knowledge Studio for viewing, validating, importing, exporting, and editing
  the complete general-knowledge catalog as YAML.
- Persistent SQLite event/memory store using WAL mode.
- Append-only raw interaction history.
- Explicit self-history of character statements.
- Detection of repeated questions within the active interaction.
- Recognition of several paraphrases as one semantic topic in the bootstrap resolver.
- Bounded raw recent-session transcripts supplied explicitly to all three live cognitive roles.
- A post-lobe executive repeat review that can recognize rephrased repeats from shared
  analysis subjects, fact anchors, or a compact embedding comparison, even when the
  initial topic keys differ.
- Conversation-level patience plus audited, subject-specific defensiveness, with a
  deterministic confused/defensive delivery guard when a local model ignores the
  executive's tone instruction.
- Post-interaction Executive reflection.
- Reflection retrieval of earlier interactions on the same resolved topic.
- Persistent event-to-event `revisits` links.
- Reflection idempotency until new conversational events occur.
- Durable deferred reflection jobs: a failed close-time reflection is leased and
  retried with exponential backoff after the conversation closes.
- Mutation policy validation.
- Immutable runtime core identity/biography.
- Evidence requirements for belief and mutable-state revisions.
- Belief revision history rather than destructive replacement.
- Provenance on derived memories.
- Developer cognition panel exposing Left, Right, Executive, interaction classification, and reflection output.
- An Ollama local-model service, with an initialization service that pulls the configured model before workers start.
- OpenAI-compatible JSON-mode model calls with per-role, mode-specific contracts and server-side Pydantic validation.
- Model-aware worker readiness checks: a worker is healthy only after its configured model is available.
- Runtime Monitor for per-role health, model latency, effective character priority,
  semantic-repeat readiness, and reflection retry state.
- Loopback-only public ports, an authenticated API proxy, no permissive CORS,
  disabled-by-default debug routes, and bounded request/turn execution.

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

## Runtime safety assumptions

The orchestrator's per-session turn lock is intentionally process-local and is
reference-counted so completed sessions do not accumulate locks. Run one
orchestrator replica for each SQLite memory database. Horizontal orchestrator
scaling requires replacing this local lock with a database-backed lease.

Executive turns carry an explicit `factual_claims` list. Each claim must cite a
stable key from the authorized `claim_evidence` catalog (identity, established
belief, or permitted knowledge). Unknown citations reject the turn before it is
committed; accepted claims are written to `executive_claim_audit` alongside the
reply. SQLite startup applies versioned migrations and enforces cross-character
event/session/link constraints with foreign keys and triggers.

## Measuring the multi-perspective design

The repository includes a paired evaluator for the original architecture
question. Run a fixed scenario set against the normal Left/Right/Executive
deployment and an Executive-only control, save results with stable `id`,
`expected`, and response fields, then run:

```text
python -m services.evaluation --multi multi.json --control control.json
```

`benchmarks/core.json` is the committed starting manifest. Produce each input
against separately started normal and `EXECUTIVE_ONLY_CONTROL=true` stacks:

```text
python -m services.evaluation --benchmark benchmarks/core.json --base-url http://127.0.0.1:8080 --token "$API_AUTH_TOKEN" --output multi.json
```

It reports paired correctness deltas, wins/losses/ties, and claim-verification
counts. It intentionally measures observable responses rather than hidden model
reasoning; a conclusion requires a representative scenario set and repeated
runs, not one favorable conversation.

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

The orchestrator admits at most `MAX_CONCURRENT_TURNS` full live turns at once
(default `2`). This protects local model capacity: when saturated, it returns a
retryable `429` instead of allowing browser retries or multiple tabs to create
an unbounded model queue.

After inference succeeds, the character reply, sourced memory writes, and
policy-checked mutations are committed as one memory-service transaction. The
initial user event remains durable for idempotent retry recovery; a failed
derived write cannot leave a partial character reply or partial conclusions.

Before Executive inference, the orchestrator reviews the completed Left and Right
artifacts against the bounded recent event stream. It produces a repeat candidate,
an inherited semantic subject key, and an interaction posture. This lets the lobes
reason freely while making repeat detection an explicit executive responsibility.

Every fresh lobe request includes the most recent raw user/character events, capped
by both event count and character count (`MAX_LOBE_TRANSCRIPT_EVENTS`,
`MAX_LOBE_TRANSCRIPT_CHARS`, and `MAX_LOBE_TRANSCRIPT_EVENT_CHARS`). The current
user message remains a separate request field. This preserves literal conversation
continuity without letting an old session grow model context indefinitely.

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

If close-time reflection fails, the session still closes immediately. The memory
service records a durable `reflection_jobs` entry, and the orchestrator later leases
and retries it with bounded exponential backoff. A late retry may append only its
reflection event to a closed session; new user or character messages remain rejected.

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
├── examples/
│   ├── characters/              # committed starter primers and test fixtures
│   └── knowledge/               # committed starter catalog and schema example
├── runtime-data/
│   ├── characters/              # local, ignored Profile Studio content
│   └── knowledge/               # local, ignored Knowledge Studio content
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
│   ├── fake_openai_provider.py  # protocol fixture, test-only
│   ├── test_e2e.py
│   └── test_policy.py
├── ui/
│   ├── index.html
│   ├── profiles.html
│   ├── nginx.conf
│   └── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── pyproject.toml
```

## Run with Docker Compose

The default configuration starts a live local [Ollama](https://docs.ollama.com/docker) provider and pulls `llama3.2:3b` into the persistent `ollama-models` volume. The first start downloads model weights and can take several minutes; subsequent starts reuse them. Set `OLLAMA_MODEL` in `.env` before starting to choose another locally supported instruct model.

Before the first start, create a private `.env` and set a cryptographically
random `API_AUTH_TOKEN`; `.env.example` includes a one-line Python generator.
The stack deliberately refuses to start without it.

On Windows, the included initializer creates the private file and never prints
the token:

```powershell
.\Initialize-LocalRuntime.ps1
```

```bash
docker compose up --build
```

Then open:

```text
http://localhost:3000
```

Profile Studio is available at:

```text
http://localhost:3000/profiles.html
```

Knowledge Studio is available at:

```text
http://localhost:3000/knowledge.html
```

Runtime Monitor is available at:

```text
http://localhost:3000/status.html
```

The orchestrator API is exposed separately at:

```text
http://localhost:8080
```

It is loopback-only and requires `X-API-Key: <API_AUTH_TOKEN>` for every route
except `/health`. The browser UI does not receive the secret: nginx injects it
only while proxying same-origin `/api` requests. Do not expose either port on a
network without an additional authenticated reverse proxy and TLS.

The memory, Ollama, and cognitive-worker ports remain internal to the Docker bridge network. A worker reports healthy only after the provider exposes its configured model.

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

### Cognitive role priorities

`cognition.left_weight` and `cognition.right_weight` are normalized on every
turn. They do not permit the Right role to override factual constraints: Left
constraints remain binding. Instead, the weights control each lobe's bounded
generation attention budget and create a stored Executive arbitration plan. When
non-factual delivery choices conflict, the plan selects the higher-weight role's
packet; equal or near-equal weights choose a balanced packet. The resulting
`cognitive_priorities` and `weighted_arbitration` are retained with the character
event and returned in `cognition` for audit.

## Profile Studio and source of truth

Profile Studio edits the canonical YAML file in `runtime-data/characters/` and immediately
updates the runtime's design-time character document. It exposes identity,
traits, cognition, speech, values, inhibitions, goals, biography, source
defaults, and an advanced complete-document JSON editor.

Conversation-derived mutable state, beliefs, and goals are shown alongside the
profile for inspection but are deliberately not overwritten by a profile save.
This keeps an author changing a character's primer from silently erasing an
ongoing character's accumulated experience.

Profile Studio can also download a `*.snapshot.yaml` export. Unlike a primer
alone, it contains the canonical source plus all durable runtime material:
mutable state, versioned beliefs, goals, derived memories, sessions/events,
event links, and mutation/belief provenance. Export before a conversation and
again afterward, then use **Compare YAML** to see the character's accumulated
conclusions directly. Import
accepts either a source-only character YAML (which updates or creates the primer)
or a full snapshot YAML (which restores the captured runtime for that profile ID).
Because a snapshot restore replaces the profile's runtime history, the UI asks for
confirmation before uploading it.

Full snapshot uploads are size- and parser-complexity-bounded, schema-validated
before any write, and reference-checked. Source YAML is staged and atomically
swapped immediately before the SQLite commit; a failed import compensates both
stores rather than leaving a mixed primer/runtime state.

## General knowledge and classification hierarchy

General knowledge is not stored as a recent event or a character memory. It is a
catalog under `runtime-data/knowledge/`, organized by classification nodes and indexed by
those labels. A node may inherit from one or more parent nodes; the loader and
Knowledge Studio reject unknown parents and cycles.

Each knowledge record has retrieval labels plus an access rule:

```yaml
id: northbridge.port_staff_manifest_triage
labels: [community.port_staff, domain.recordkeeping]
access:
  require_all: [community.port_staff]
assertions:
  - Staff reconcile manifests with berth logs.
epistemic_type: canon_policy
```

`epistemic_type` on a general-knowledge record is a bounded, lower-case
setting label, not the closed runtime-memory enum. It can therefore preserve
categories such as `canon_policy`, `contested_principle`, `historical_fact`, or
`provisional_model`. Runtime memories and mutations continue to use the
separate, fixed epistemic-type enum above.

At turn time, the runtime resolves only taxonomy aliases in the user’s message,
uses the label index to select candidates, derives a character’s label closure
from their birthplace, faction, occupation, and explicit `knowledge_labels`, and
then applies `require_all` / `require_any`. Only the allowed subset is included
in cognitive-worker context. The model never receives denied records or the full
corpus. Recent events and memories can affect interpretation, but cannot mutate
or grant this general-knowledge source.

### Knowledge Studio, catalog import, and schema sample

`examples/knowledge/northbridge_port.yaml` is the starter corpus. The first save or
import from Knowledge Studio writes `runtime-data/knowledge/catalog.yaml`; that managed file
then becomes the single authoritative source and takes precedence over bundled
starter files. `knowledge/catalog.example.yaml` is a complete valid schema
example that is intentionally ignored by the loader. It provides the shape for
classification IDs, parents/aliases, record retrieval labels, access rules,
assertions, epistemic categories, confidence, and source metadata.

Knowledge Studio downloads the exact active catalog, lets an author validate a
full YAML replacement without changing runtime state, and imports or saves only
after the hierarchy and Pydantic schema pass. A successful write atomically
replaces both `catalog.yaml` and the SQLite label index; a failed validation or
write preserves the previously active source and index. The authenticated
management API is the only in-container writer for that mounted catalog. A
conversation, reflection, or cognitive worker has no endpoint to edit it.

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

The runtime maintains two separate signals:

- **Conversation patience** declines gradually with the current session's turns and
  active repeat streak; changing subject removes only the active repeat penalty.
- **Subject defensiveness** is durable mutable state keyed by the resolved semantic
  subject. It rises only for that subject, cools slowly on a non-repeat return, and
  is revision-audited with the user event as evidence.

Their intersection produces a suggested `normal`, `reclarify`, `confused`, or
`defensive` posture. It is evidence for the Executive, not an automatic emotional
escalation: the Executive explicitly selects `hold`, `increase`, or `deescalate`
before durable subject defensiveness changes.

For an immediate exact repeat in the same session that already received a character
answer, the runtime reuses the stored compact Left/Right artifacts and invokes only
the Executive for two linked tasks: it first produces a compact, fallible assessment
of plausible repeat intent (for example, seeking a different angle, checking
consistency, or not understanding), then it produces the spoken response from that
assessment. The Executive has a separate, larger repeat token budget for both calls.
If a speaking attempt merely echoes the prior answer, it gets one Executive revision
using the same assessment. Only after both attempts fail does an intent-specific
fallback prevent another visible echo; it is not a single canned clarification phrase.
Rephrases still use fresh lobe analysis so the Executive can detect non-exact semantic
repetition safely. `cognition.timing_ms` reports the Executive assessment and any
revision separately.

For fresh lobe turns, the post-lobe review also batches the current question and up
to `SEMANTIC_REPEAT_MAX_CANDIDATES` earlier answered questions through Ollama's
compact `OLLAMA_EMBED_MODEL` (default `all-minilm`) embedding endpoint. Cosine
similarity at or above `SEMANTIC_REPEAT_SIMILARITY_THRESHOLD` (default `0.80`) is
recorded as embedding evidence for the Executive. If that optional call is down,
the turn continues with the existing lexical and lobe-artifact checks; it never
silently invents a repeat match.

The deterministic topic resolver is intentionally temporary. It is not used for
general-knowledge retrieval: that path is taxonomy-label indexed.

## Live model providers and output contracts

Every cognitive worker calls an OpenAI-compatible `POST /v1/chat/completions` endpoint. Docker Compose defaults to one bundled Ollama service at `http://ollama:11434/v1`; this is the low-resource option, but it can serialize competing model work. On an NVIDIA-enabled Docker host, enable GPU access for that default provider with:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

Each role can independently target another provider through these `.env` variables:

```text
OLLAMA_MODEL
OLLAMA_EMBED_MODEL
LEFT_MODEL_BASE_URL    LEFT_MODEL_NAME    LEFT_MODEL_API_KEY    LEFT_MODEL_TIMEOUT_SECONDS    LEFT_MODEL_MAX_TOKENS    LEFT_MODEL_OUTPUT_ATTEMPTS
RIGHT_MODEL_BASE_URL   RIGHT_MODEL_NAME   RIGHT_MODEL_API_KEY   RIGHT_MODEL_TIMEOUT_SECONDS   RIGHT_MODEL_MAX_TOKENS   RIGHT_MODEL_OUTPUT_ATTEMPTS
EXEC_MODEL_BASE_URL    EXEC_MODEL_NAME    EXEC_MODEL_API_KEY    EXEC_MODEL_TIMEOUT_SECONDS    EXEC_MODEL_MAX_TOKENS    EXEC_MODEL_OUTPUT_ATTEMPTS
```

For an independent local-provider experiment, apply the dedicated topology:

```powershell
docker compose -f docker-compose.yml -f docker-compose.dedicated-providers.yml up --build -d
```

It starts `ollama-left`, `ollama-right`, and `ollama-executive`, reserves NVIDIA GPU access for them, and repoints the three worker containers to their matching provider. It removes the shared scheduler choke point, but can load three copies of the model, so it needs enough available GPU memory. The API includes `cognition.timing_ms` on every chat turn to compare the lobe critical path and executive time between topologies. Confirm the active processor with `docker compose exec ollama-left ollama ps`; a CPU report means provider splitting will usually be slower, not faster.

In dedicated mode the base Ollama service is retained only for the one-time model
pull and the compact repeat embedding call; it has no GPU reservation and keeps no
language model resident after bootstrap.

The workers request documented OpenAI-compatible JSON mode and validate the returned JSON before returning it to the orchestrator. The accepted contracts are intentionally different for each task:

```text
left / turn             topic, fact references, constraint codes, action code, confidence
right / turn            action code, affect, tone code, risk code, association keys
executive / turn        speech, strategy, typed mutations, memory writes
executive / reflection  summary, event links, typed mutations
```

Left and Right artifacts deliberately use compact semantic keys rather than user-facing sentences. The Executive receives both artifacts and is the only role that turns them into natural language. The repeat review understands both this compact format and historical stored turns from the previous format.

If a provider is unreachable, the model is missing, or its output violates the contract, the worker returns a controlled error and no derived memory or mutation is written. Workers remain stateless; the memory service remains the sole owner of durable character state.

## API outline

### Orchestrator

```text
GET  /health
GET  /status?character_id={optional_character_id}
GET  /characters
GET  /characters/{character_id}/state
GET  /profiles
GET  /profiles/{character_id}
POST /profiles
PUT  /profiles/{character_id}
GET  /profiles/{character_id}/export
POST /profiles/{character_id}/diff
POST /profiles/import
POST /sessions
GET  /sessions/{session_id}/events
POST /sessions/{session_id}/chat
POST /sessions/{session_id}/reflect
POST /sessions/{session_id}/close
POST /reflection-retries/run
GET  /debug/{character_id}  # only when ENABLE_DEBUG_API=true
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
model-provider output is validated against the role contract
core mutation policy rejects runtime biography changes
```

Current result:

```text
50 passed
```

## Deliberately deferred

The MVP does not yet implement:

```text
embedding/vector retrieval for the general-knowledge corpus
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

The current runtime uses real lightweight local models; the remaining work is about richer retrieval, reasoning, and simulation integration.

## Next implementation milestones

1. Replace the bootstrap topic normalizer with stable semantic proposition IDs and embedding-assisted retrieval.
2. Add contradiction detection between incoming claims, prior self-statements, beliefs, and canonical facts.
3. Expand reflection to create causal/support/contradiction/supersession graph edges.
4. Add typed belief extraction and confidence revision from completed interactions.
5. Add a world-state/tool boundary so the Executive can propose actions without directly mutating the simulation.
6. Run heterogeneous local models for Left, Right, and Executive and measure latency, coherence, contradiction rate, and repeat-answer stability against a single-model baseline.
