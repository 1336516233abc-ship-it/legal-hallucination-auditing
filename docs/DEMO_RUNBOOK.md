# Official UI Demonstration Guidelines

This page is for on-site operation and result explanation of the current UI, not a course report. For official entry points, dependencies, and model loading methods, see [UI_HANDOFF.md](UI_HANDOFF.md). Demonstration total duration follows the latest course notification.

## First Explain the Purpose

“We allow users to paste contracts, verify propositions, or search for clauses. The system retrieves evidence from the current contract, generates answers, checks the basis, and attempts one revision if needed. The page preserves the original text and check status for users to verify themselves. Model checking may be incorrect, so the system does not promise to eliminate hallucinations.”

## Operation Sequence

1. Load the model in advance. Open the local page and run one input to confirm the service is available. Model loading time and inference time are explained separately.
2. Click “30-day notice” and run. Point out that the left side is the current contract, and the right side is the model's conclusion and original text evidence for this run.
3. Click “Change to 60-day notice”. Let the audience see the old result cleared first, then submit; check that the new answer's evidence comes from the 60-day contract. Can also modify the English original text or proposition based on audience suggestions.
4. When time permits, click “Find liability cap” to demonstrate the second type of task. Extracted sentences must match the original text.
5. Expand check and run records, distinguishing initial verification from final revision. Explain that “revision occurred” does not equal “revision succeeded”; format or citation errors will stop answer display.

These examples are manually constructed functional acceptance inputs, not treated as official dataset results. On-site presentation follows actual output; when failures occur, display errors and evidence, don't substitute real runs with handwritten success answers. Pre-recorded or existing acceptance records can only be explicitly shown as backup records.

## Experimental Conclusions That Can Be Stated

- Verified 18,666 server predictions and 37,834 model calls across two models, two datasets, and three methods, completed full-scale error statistics and 20 traceable cases; this analysis did not rerun models.
- New runs of 36 retrieval evaluation configurations across complete test splits, plus two models, three-variant Agent pilot on the same 10 questions, totaling 60 outputs.
- Agent's benefit to label accuracy is inconsistent: Llama 3 / ContractNLI improved from RAG's 42.52% to 55.52%; the other three model/task combinations declined.
- Among 11 revisions in the component pilot, 1 had label change from wrong to correct, with residual and newly introduced citation errors also present. Cannot claim “100% revision success”.
- Independent hallucination rate, semantic answer completeness, and abstention precision with independent ground truth have not yet been measured. Full-scale metric files separately list lexical overlap and legacy rule proxy values, which cannot be treated as these semantic metrics; label accuracy, internal verifier status, and independent hallucination rate must also be distinguished.

Complete records see [Full-scale Analysis](validation/part4_2026-11-24/README.md) and [Report](../FINAL_REPORT.md). Early pilot values and sources see [Results Summary](validation/part4_2026-11-22/RESULTS.md), [RQ Findings](validation/part4_2026-11-23/RQ_FINDINGS.md), and [Error Cases](validation/part4_2026-11-22/CASE_ANALYSIS.md). Do not extrapolate the 10-question pilot per model as full-scale component effects; both early 9 pilot cases and subsequent 20 full-scale cases are purposively selected and cannot be treated as overall error type proportions.

## Common Follow-up Questions

**Why does it still require manual verification when “check passed” is displayed?** The page checks output format, citation number range, and extracted sentences; semantics and evidence sufficiency may still be incorrect; the verifier itself also uses a model.

**Has it been proven to reduce hallucinations?** Independent support assessment across the three methods has not yet been completed; existing label accuracy cannot substitute for this metric. Evaluation materials have been prepared; specific method and human division of labor await team confirmation.

**Can the model be changed?** At startup, choose `--model llama3` or `--model mistral` and provide the corresponding model directory; load one model at a time. The current model on the page will be explicitly displayed.

**Has full-scale component ablation been completed?** No. The existing component ablation is a paired pilot of fixed 10 questions; expanding scope should be determined in combination with team experimental requirements and server resources.
