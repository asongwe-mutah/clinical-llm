# Safety, ethics & responsible use

This project is a **research and portfolio artifact**. It is not a medical
device and must not be used to make decisions about real patients.

## Hard rules baked into the project

1. **No PHI, ever.** The repo ships no patient data. `.gitignore` blocks
   datasets, checkpoints, and `*.jsonl`. The credentialed note-summarization
   task reads from a *local* path you control under your PhysioNet DUA — it is
   never committed.
2. **Safety framing at every layer.**
   - The training system prompt instructs the model to defer to clinicians and
     flag uncertainty, so the behavior is *learned*, not just bolted on.
   - The API attaches a disclaimer to every response.
   - The web UI shows a persistent "not medical advice / no PHI" banner.
3. **Clear train/test separation** so reported metrics are honest (MedMCQA's
   validation split is never used for fine-tuning).

## Known failure modes

- Fluent hallucination of drugs, doses, guidelines, or citations.
- Overconfidence on ambiguous or under-specified questions.
- Distribution shift: strong on exam-style MCQs, weaker on messy real notes.

## If you extend this

- Do **not** deploy to end users (patients) without clinician-in-the-loop
  review and a formal risk assessment.
- Respect every dataset license and the base model license.
- Consider adding retrieval grounding + citation (pairs naturally with a
  RAG system) before trusting free-text medical answers.
