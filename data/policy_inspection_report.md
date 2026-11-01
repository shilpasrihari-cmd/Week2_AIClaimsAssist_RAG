# AIClaimsAssist Policy Content Inspection

This report validates the extracted text for the two policy IDs present in the
corpus. It is based only on the text extracted from the source documents. The
source documents were not modified.

## Source and metadata verification

| Policy ID | Source filenames | Extracted document IDs |
|---|---|---|
| `MOT-0000002` | `MOT-0000002-contract.docx`, `MOT-0000002-declarations.pdf`, `MOT-0000002-endorsements.pdf`, `MOT-0000002-schedule.pdf` | `MOT-0000002-contract`, `MOT-0000002-declarations`, `MOT-0000002-endorsements`, `MOT-0000002-schedule` |
| `MOT-0000004` | `MOT-0000004-contract.docx`, `MOT-0000004-declarations.pdf`, `MOT-0000004-endorsements.pdf`, `MOT-0000004-schedule.pdf` | `MOT-0000004-contract`, `MOT-0000004-declarations`, `MOT-0000004-endorsements`, `MOT-0000004-schedule` |

The loader preserves `source_filename` from the filesystem and derives a
stable `document_id` from the filename stem. The policy ID is also present in
the contract, declarations, endorsement schedule, and coverage schedule text.
The currency glyph appears as a replacement character (`�`) in the extracted
terminal text, so the report preserves that observed representation rather than
silently normalizing it.

## Comparison of stated content

| Information | MOT-0000002 | MOT-0000004 |
|---|---|---|
| Named insured | Dott. Alfio Leone | Mr. Stewart Montague |
| Policy period | 26/11/2023 to 25/11/2024 | 04/12/2024 to 04/12/2025 |
| Vehicle | Peugeot 208 (2016, black), registration HX-9830-IX | Opel Astra (2008, red), registration XD-9797-YI |
| Annual premium | `�2,075.00` as extracted | `�960.00` as extracted |
| Stated deductible | `�1,000.00` as extracted; schedule says per occurrence and repeats it for each listed coverage | `�500.00` as extracted; schedule says per occurrence and repeats it for each listed coverage |
| Bodily injury, per person | `�100,000.00` as extracted | `�100,000.00` as extracted |
| Bodily injury, per accident | `�300,000.00` as extracted | `�300,000.00` as extracted |
| Property damage | `�100,000.00` as extracted | `�100,000.00` as extracted |
| Coverage types named in contract | Bodily injury, property damage, collision, and other-than-collision (comprehensive) | Bodily injury, property damage, collision, and other-than-collision (comprehensive) |
| Coverage types with schedule limits | Bodily injury per person, bodily injury per accident, property damage | Bodily injury per person, bodily injury per accident, property damage |
| Endorsements | `NEW_CAR_REPL`: New Car Replacement; replaces a totaled new vehicle with a comparable new model | `GAP`: pays difference between actual cash value and loan balance; `NEW_CAR_REPL`: replaces a totaled new vehicle with a comparable new model; `UM_UIM`: covers injury caused by an at-fault driver lacking adequate limits |
| Explicit exclusions | Racing or speed contests; public livery or ride-share without applicable endorsement; intentional damage by an insured | Same exclusions |
| Other stated conditions | Prompt notice, protect property, cooperate, signed proof of loss; intentional concealment/misrepresentation voids policy; premium payable as condition of insurance | Same conditions |

## Policy-specific findings

### MOT-0000002

The documents state liability coverage for bodily injury and property damage,
with limits of `�100,000` per person, `�300,000` per accident, and `�100,000` for
property damage. The contract also names collision and other-than-collision
(comprehensive) coverage, subject to the applicable deductible. The schedule,
however, lists only the three liability entries and does not provide separate
collision or comprehensive limits.

The `NEW_CAR_REPL` endorsement is listed and described, but the extracted text
does not state its eligibility rules, valuation method, maximum benefit, or
claim conditions.

### MOT-0000004

The stated liability limits and listed coverages match MOT-0000002. The
deductible is `�500.00` rather than `�1,000.00`. The policy additionally lists GAP,
NEW_CAR_REPL, and UM/UIM endorsements. The extracted text gives short
descriptions, but no endorsement limits, eligibility rules, exclusions,
procedures, or interaction with the base coverage.

The insured vehicle is a 2008 Opel Astra while the NEW_CAR_REPL description
refers to replacing a totaled "new vehicle." This is not enough text to call
the policy internally contradictory, but it is an unresolved eligibility
ambiguity. The source does not say whether the endorsement applies to this
vehicle.

## Missing, contradictory, or ambiguous information

1. Collision and comprehensive are named in both contracts, but their own
   limits, covered perils, valuation rules, and claim conditions are not in the
   extracted schedules.
2. It is unclear whether the generic/per-occurrence deductible applies to the
   liability entries. The schedule repeats a deductible beside those entries,
   but the contract only says collision and comprehensive are subject to the
   applicable deductible.
3. No separate limit is stated for UM/UIM on MOT-0000004.
4. GAP has no stated maximum, eligibility criteria, exclusions, loan-document
   requirements, or claim procedure.
5. NEW_CAR_REPL has no stated eligibility criteria, time or mileage rules,
   valuation cap, or claim procedure. This is especially ambiguous for the
   2008 vehicle on MOT-0000004.
6. The listed exclusions are not necessarily a complete exclusion set; the
   extracted contract says only that loss arising from the three described
   situations is not covered. No additional policy forms were assumed.
7. The date strings use numeric day/month/year-looking formatting, but the
   extraction does not define a date locale. The report preserves the source
   strings without normalizing them.

No direct contradiction was found in the stated liability limits, deductibles,
policy IDs, or source-to-policy associations. The NEW_CAR_REPL wording versus
the 2008 vehicle on MOT-0000004 is an eligibility ambiguity, not a proven
contradiction.

## Basic auto-insurance question readiness

**Partially sufficient.** The documents are sufficient to answer basic factual
questions such as:

- who the named insured is;
- the source policy ID and policy period as written;
- the insured vehicle and registration;
- the listed bodily-injury and property-damage limits;
- the stated deductible values;
- which endorsement codes are listed; and
- the exclusions explicitly present in the extracted contract.

They are **not sufficient on their own** to reliably answer complete claim
coverage questions involving collision, comprehensive, UM/UIM, GAP, or new-car
replacement eligibility because material limits, definitions, conditions, and
endorsement terms are absent from the extracted set or only summarized.

## Validation basis

The inspection was run against all eight files under `data/policies`:

- `MOT-0000002`: 4 documents, all extracted successfully.
- `MOT-0000004`: 4 documents, all extracted successfully.

No empty or unreadable source document was observed during extraction.
