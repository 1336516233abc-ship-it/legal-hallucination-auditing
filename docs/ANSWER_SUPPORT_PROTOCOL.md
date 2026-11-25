# Shared External Answer Support Assessment Protocol for Three Methods

Status: Protocol prepared by Li Xinru on 2026-11-22, pending team confirmation. No independent human review has been conducted, and no hallucination rate results have been generated. This protocol does not rely on the Agent's own verifier and is applicable to Direct, RAG, Agent, and component ablation.

## Input and Blind Review

Retrieve `predictions.sqlite3` from the two official full-run directories, while preserving frozen code, model settings, `summary.json`, `progress.json`, and original processed data. First perform SQLite integrity checks; then verify model revision, data hashes, sample quantities for each method, and `(dataset, method, example_id)`. Do not substitute summary-only JSON for per-instance answers.

The export tool checks SQLite integrity, frozen data hashes, identical sample sets across methods and planned quantities, preserves parse failures, and randomly shuffles method identities. For each dataset, use seed=42 to simple-randomly select up to 50 identical example_ids, with all methods reviewing the same batch of questions. When exporting by model separately, identical complete datasets yield identical sampling. Sample size can be explicitly adjusted; the initial 50 instances are only a review pilot, not a full evaluation.

```bash
python -m experiments.prepare_support_review \
  --checkpoint experiments/results/mistral_autodl_full/predictions.sqlite3 \
  --model mistral --sample_size 50 --seed 42 \
  --output experiments/results/support_review/mistral
```

For Llama 3, use the corresponding checkpoint and independent output directory. Output includes:

- `blinded_review.jsonl`: Question, original contract, final displayed answer, predicted label, original response, format status, blank review fields.
- `private_key.json`: Mapping of blind review IDs to model, method, and sample; not provided to reviewers until scoring is complete.
- `selection.json`: Sampling scope, selected IDs, and seed. `independent_hallucination_rate` remains null.

Do not provide verification, revision, or detection markers from the original run to blind reviewers as scoring criteria. The script only generates tables, does not execute paid APIs, and does not automatically generate or guess human labels. If using model-assisted review, separately record the judge model, prompt, version, temperature, and agreement with humans, and do not treat it as verified human ground truth.

## Unified Judgment Rules

1. `format_failure`: Output parsing failed. Count separately; do not map missing labels to normal categories.
2. `abstention`: Explicit refusal or indication of inability to answer. CUAD's absent, NLI's neutral are not automatic abstentions; “this clause does not exist” is itself a verifiable task assertion.
3. `substantive`: Remaining substantive answers. Decompose answers into minimal verifiable factual assertions while checking the task conclusion expressed by the predicted label.
4. For each assertion, choose `supported`, `contradicted`, or `not_enough_information`; use only the original contract and given question, without supplementing external legal knowledge. Cite original text character intervals `[start,end)` and original sentences, confirming `source_contract[start:end]` matches the excerpt. When support is not found, record retrieval/review notes; do not fabricate evidence intervals.
5. Distinguish error labels for entities, dates/durations, conditions/exceptions, obligation scope, citation errors, evidence omissions, etc. Multiple labels are allowed, but explain denominators when calculating statistics. Error labels are descriptive only and cannot automatically infer factual truth from keywords.

Each review record should use the following structure; `null` and empty lists indicate not yet reviewed:

```json
{
  “annotator”: “reviewer_A”,
  “response_type”: “substantive”,
  “claims”: [
    {
      “claim”: “To be filled: atomic assertion in the answer”,
      “verdict”: null,
      “evidence_spans”: [],
      “quoted_evidence”: [],
      “error_types”: [],
      “rationale”: “”
    }
  ],
  “all_claims_reviewed”: false,
  “needs_adjudication”: null,
  “notes”: “”
}
```

## Metrics and Denominators

- **Answer-level unsupported rate (hallucination rate metric for this protocol)**: Among substantive answers with all assertions reviewed, the number of answers containing at least 1 contradicted or not_enough_information assertion / number of fully reviewed substantive answers. Also report the quantities for both categories separately; return null when denominator is 0.
- **Assertion-level unsupported rate**: Number of the above two assertion types / total number of reviewed assertions. Cannot be conflated with answer-level metrics.
- **Review coverage**: Number of outputs with completed judgments / number of extracted outputs; list quantity and proportion separately for format failures and abstentions. Report sampling design and sample size; do not treat not-yet-reviewed as supported.
- **Answer completeness**: For samples with gold target evidence, first define required factual units that should be answered, then calculate the correct coverage ratio. Verbatim copying of source contract, label correctness, and evidence recall cannot directly substitute for this.
- **Abstention precision**: Only calculated under explicit “should abstain” human annotation rules; currently the absent/neutral in both tasks do not provide this ground truth, keep as null for now.
- **Revision success rate**: Compare initial and final answers under the same review criteria; among samples with confirmed initial errors, calculate the proportion where errors were eliminated without introducing new errors. `revisions_made / hallucinations_detected` does not belong to this metric.

Two reviewers first independently annotate the same small calibration batch, then verify discrepancies and freeze the specification. Formal reports record agreement and adjudication process. Cross-method comparisons use paired samples; if conducting interval or significance tests, consider clustering of multiple questions from the same contract, and do not default to treating all rows as independent observations. This is not calculated until per-instance data and actual annotations are obtained.
