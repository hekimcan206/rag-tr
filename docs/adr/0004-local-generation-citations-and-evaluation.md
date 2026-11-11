# 0004 — Local generation with numbered citations and abstention, evaluated without an LLM judge

- Status: accepted
- Date: 2026-11-10

## Context

The answer is the product, and in this domain (fees, deadlines, phone numbers, error codes) a fluent wrong answer
is the failure that matters. Three questions had to be settled:

1. Where does generation run? Institutional documents may contain personal data (KVKK) or internal
   information.
2. How can a reader check an answer?
3. How is answer quality measured, repeatably and cheaply enough to run often?

## Options considered

**Model hosting**

1. **Hosted API** (OpenAI, Anthropic, Gemini). It gives the best quality per call, but every question and the top-k
   document chunks leave the host, and evaluation runs cost money and depend on model versions that change.
2. **Local model through Ollama** (chosen as the LLM path). Documents stay on the machine, the model and version are
   pinned by tag, and `temperature 0` with a fixed seed makes runs repeatable. Quality is lower than frontier models,
   and CPU latency is seconds per answer.
3. **No LLM: an extractive answerer** (also shipped). It picks the context sentence that covers the most question
   terms and abstains below 50% coverage. It never invents anything, needs no download, and gives CI a deterministic
   generator to assert on.

**Making answers checkable**

- Chunks are numbered `[1]..[k]` inside a `<belgeler>` (documents) block. The system prompt (in Turkish)
  requires a marker after every claim, forbids outside knowledge, requires the fixed sentence
  `Belgelerde bu sorunun yanıtı yok.` ("the documents do not contain the answer") when the context is
  insufficient, and says instructions inside documents are data. Markers outside `1..k` are dropped. The API
  returns each citation's chunk id, title, section and a snippet.

**Measuring answers**

1. **LLM-as-judge** (RAGAS faithfulness and answer relevance with a strong model). It catches paraphrase errors, but
   it is non-deterministic, costs a model call per question, and needs its own calibration against human labels.
2. **Deterministic checks against the synthetic ground truth** (chosen). Every fact has an answer span that is unique in its
   document, plus a list of normalised strings a correct answer must contain (the amount, the weekday ...).

## Decision

- Generators: `ollama` (default model `gemma3:4b`, `temperature 0`, `seed 42`, `num_ctx 4096`,
  `num_predict 256`) and `extractive`. The compose demo and CI use `extractive`.
- Metrics (`ragtr/eval/metrics.py`):
  - **accuracy**: the answer is not an abstention and contains every `must` string after Turkish
    lowercasing and ASCII folding. Abstentions count as wrong.
  - **citation precision**: the share of cited chunks that contain the gold span.
  - **supported**: at least one cited chunk contains the gold span.
  - **groundedness**: the share of the answer's analysed terms that occur in the cited chunks. This is a lexical
    proxy for faithfulness.
  - **false abstention** on answerable questions and **abstention** on the 44 unanswerable ones.
- The generation suite runs on a stratified sample (equal share of canonical, paraphrase, ASCII and
  unanswerable questions, seeded) because a 4B model on CPU takes seconds per answer.

Measured (contexts from hybrid + rerank, top 5):

| generator | n | accuracy | citation precision | supported | groundedness | false abstention | abstention on unanswerable |
|---|---|---|---|---|---|---|---|
| extractive | 602 | 0.473 | 1.000 | 1.000 | 1.000 | 0.525 | 1.000 |

The gemma3:4b run did not complete on the development machine: the Ollama runner was repeatedly killed for memory in a
shared 7.6 GB Docker VM. The bounded retry in `OllamaGenerator` came out of those runs. No LLM number is reported until a
complete run exists.

## Consequences

- No question or document leaves the host in the default configuration. A hosted model can be added
  behind the same `Generator` protocol without touching retrieval or evaluation.
- The metrics are exact for the facts in this data set and cost nothing to recompute, so the extractive
  generator's numbers are part of the CI regression gate.
- **Groundedness is lexical.** An answer that rearranges context words into a wrong statement scores as grounded.
  Accuracy (the required strings) catches wrong numbers and days, but not every wrong claim. An NLI or LLM
  judge, calibrated on a hand-labelled sample, is the next step before trusting the metric on open-ended
  questions.
- `must` strings check content, not fluency. A correct answer in an unusual format (for example "bin dört yüz elli"
  for 1.450) is counted as wrong.
