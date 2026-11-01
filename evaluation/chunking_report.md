# AIClaimsAssist Chunking Comparison

The report is based on the 8 source documents currently present in
`data/policies`. Original PDF and DOCX files were not modified. No embeddings,
Pinecone indexing, BM25 retrieval, or LLM calls were performed.

## Strategies

- **Fixed-size:** LangChain `RecursiveCharacterTextSplitter` with the
  `cl100k_base` token encoding, a 300-token target, and 50-token overlap.
- **Section-aware:** explicit policy headings are retained with their clauses.
  Sections shorter than 300 tokens remain intact; oversized sections use the
  same token-aware splitter.

## Measured results

| Strategy | Total chunks | Average tokens | Average characters | Empty chunks | Chunks with missing metadata |
|---|---:|---:|---:|---:|---:|
| Fixed-size | 10 | 218.70 | 887.70 | 0 | 0 |
| Section-aware | 32 | 63.97 | 257.84 | 0 | 0 |

The fixed-size average is below the 300-token target because the source files
are short: six files produced one chunk each and the two contract files
produced two chunks each. The target is a maximum-oriented splitter setting,
not a requirement to pad short documents.

### Chunks by source document

| Source filename | Fixed-size | Section-aware |
|---|---:|---:|
| `MOT-0000002-contract.docx` | 2 | 6 |
| `MOT-0000002-declarations.pdf` | 1 | 4 |
| `MOT-0000002-endorsements.pdf` | 1 | 3 |
| `MOT-0000002-schedule.pdf` | 1 | 3 |
| `MOT-0000004-contract.docx` | 2 | 6 |
| `MOT-0000004-declarations.pdf` | 1 | 4 |
| `MOT-0000004-endorsements.pdf` | 1 | 3 |
| `MOT-0000004-schedule.pdf` | 1 | 3 |

## Sample chunks and metadata

### Fixed-size sample

```text
Text: SYNTHETIC ... Meridian Mutual (fictional)
Meridian Mutual Insurance SE
Motor Policy ... Contract
This policy is a contract between Dott. Alfio Leone (the "named insured") ...

Metadata:
{
  "policy_id": "MOT-0000002",
  "document_id": "MOT-0000002-contract",
  "source_filename": "MOT-0000002-contract.docx",
  "document_type": "docx",
  "section_heading": "Document",
  "chunk_id": "MOT-0000002-contract:fixed_size:0000:bb7b63ea3e",
  "chunking_strategy": "fixed_size"
}
```

### Section-aware sample

```text
Text: Insuring Agreement
We will pay damages for bodily injury and property damage for which any
insured becomes legally responsible because of an auto accident, subject to
the limits shown on the Declarations.
Collision and Other Than Collision (comprehensive) coverages apply to a
covered auto less the applicable deductible shown on the Declarations.

Metadata:
{
  "policy_id": "MOT-0000002",
  "document_id": "MOT-0000002-contract",
  "source_filename": "MOT-0000002-contract.docx",
  "document_type": "docx",
  "section_heading": "Insuring Agreement",
  "chunk_id": "MOT-0000002-contract:section_aware:0002:717fb75477",
  "chunking_strategy": "section_aware"
}
```

The sample text is taken from the actual extracted source text. The terminal
extraction renders some source glyphs as replacement characters; those glyphs
were not normalized or invented by chunking.

## Metadata and policy separation checks

Every generated chunk contains all seven required metadata fields:

`policy_id`, `document_id`, `source_filename`, `document_type`,
`section_heading`, `chunk_id`, and `chunking_strategy`.

`MOT-0000002` and `MOT-0000004` remain separate because every chunk ID includes
its document ID, and every document ID includes its policy ID. Chunk IDs are
deterministic for unchanged source text, strategy, and chunk order; a short
content digest also prevents silent reuse when a chunk's content changes.

## Validation

```text
7 passed in 2.01s
```

No empty chunks and no missing metadata were found in either strategy. The
implementation is limited to document chunking and does not perform downstream
vector or lexical retrieval work.
