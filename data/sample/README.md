# sample/

A **small, committed slice** of the corpus so the repo is usable and inspectable without running a
full generation. Regenerated deterministically via `make sample` (seed 42, `sample` profile). The
**full** corpus is gitignored and published to HuggingFace — see [BRIEF.md](../BRIEF.md).

## Contents

Generated from 21 entities (6 policyholders, 3 agents, 2 adjusters, 5 policies, 5 claims) →
79 documents, 25 golden questions:

```
model.json                          canonical entity model (the spine)
roster.tsv                          id-keyed master-data index (the join target)
schema/                             JSON Schema for model.json + roster.tsv
docs/policy/*-declarations.pdf      declarations page (one per policy)
docs/policy/*-contract.docx         full policy contract — Word (one per policy)
docs/policy/*-endorsements.pdf      endorsement schedule (one per policy)
docs/policy/*-schedule.pdf          coverage schedule (one per policy)
docs/claim/*-fnol.pdf               First Notice of Loss form (one per claim)
docs/claim/*-adjuster-report.pdf    adjuster findings + disposition (one per claim)
docs/claim/*-estimate.pdf           damage/repair estimate (open & closed claims)
docs/claim/*-settlement-letter.pdf  settlement letter (closed claims)
docs/claim/*-denial-letter.pdf      denial letter (denied claims)
docs/claim/*-accident-statement.pdf Motor accident statement (own EAS-inspired form; Motor claims)
docs/claim/*-scanned.jpg            scanned (OCR-target) variants of FNOL + letters + accident statement
docs/identity/*-id-card.pdf         policyholder ID card — real synthetic PII + ICAO MRZ + portrait (per holder)
docs/identity/*-id-card-scanned.jpg scanned (OCR-target) variant of each ID card
docs/tabular/loss-run.xlsx          loss run — every claim (Excel)
docs/tabular/reserve-register.xlsx  open-claim reserves (Excel)
docs/tabular/premium-register.xlsx  policy premiums (Excel)
docs/tabular/commission-summary.csv per-agent commission (CSV)
docs/kb/underwriting-guidelines.md  underwriting guidelines (Markdown)
docs/kb/claims-handling-manual.docx claims handling manual (Word)
docs/kb/customer-faq.md             customer FAQ (Markdown)
evidence/*-evidence.jpg             AI damage/property photos — one per claim (rendered sample pixels)
faces/*-face.jpg                     AI ID portraits — one per policyholder (rendered sample pixels; embedded in the ID card)
image-prompts.jsonl                 seeded prompt-spec recipe for every AI image — evidence photos + ID portraits (the reproducible artifact)
manifest.json                       every document + provenance + sha256
golden.jsonl                        golden eval questions (semantic + aggregation) — mirrored to ../golden/golden.jsonl
pii-index.jsonl                     redaction ground truth — every PII span (doc · type · field · value · modality)
```

The `evidence/` and `faces/` images are AI-generated (non-deterministic pixels); `image-prompts.jsonl` is their
committed, reproducible recipe. The full corpus commits only the recipe — pixels are produced on-demand for the HF
release. The ID-card PDF embeds the portrait when its pixels are present, else a neutral placeholder, so the card
itself stays byte-reproducible. Each face carries an EXIF synthetic marker; the portraits depict fully synthetic,
non-existent people (realistic PII is the redaction-test material — PII handling is the consuming RAG layer's job).

Validate it: `make validate OUT=sample`.

More formats (knowledge base, scanned variants, images) are added to this slice as M2–M3 land.
See [docs/format-matrix.md](../docs/format-matrix.md) for the full plan.
