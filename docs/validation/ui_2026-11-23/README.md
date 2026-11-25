# Official UI Acceptance Record

Acceptance target is `ui/app.py`, not historical local candidates. Acceptance separately records code regression, real model requests, and browser operations; does not treat them as manual blind review or formal dataset experiments.

## Automated Regression

Final code: **182 passed**. Includes contract-switching isolation, concurrent requests, format/citation failures, CUAD original sentence constraints, empty input, exception recovery, complete token statistics, MLX same-thread loading/inference, and queued request recovery of submitted input before displaying results. Existing 60 manual review materials remain empty; did not use tests to generate scores.

[JUnit](tests.xml) records test count and item-by-item results. Hostname has removed local identifiers.

## Real Service Invocation

Same official Gradio service separately loads two existing original models; Apple Silicon / MLX, temperature 0, seed 42. Calls same `/review` handler function via Gradio client; each model processes three manually constructed inputs (30-day, changed to 60-day, liability cap) in sequence, then checks empty input. Did not modify original experiment summary.

|Model|30-day claim|Contract changed to 60-day|Liability cap extraction|Empty input|
|---|---|---|---|---|
|Llama 3|entailment, as expected|contradiction, as expected|present, sentence matches contract|rejected, model not called|
|Mistral|entailment, as expected|invalid label, answer display stopped|present, sentence matches contract|rejected, model not called|

- [Llama 3 Real Results](llama3_service_fixed.json): Three model cases passed; every piece of evidence consistent with current contract character intervals.
- [Mistral Real Results](mistral_service.json): Two cases passed, one format failure intercepted. Interception does not equal correct answer.
- [Initial Run Failure Record](llama3_service.json): Initially all three requests encountered MLX cross-thread stream error. After official code changed to load and execute model in dedicated thread, retested and saved to new file without overwriting failure record.

Finally added “restore submitted input when queue starts” page protection; Agent and model invocation logic unchanged. Above two groups' complete acceptance records contain source code hash at that time. Final file hashes and one-time 60-day targeted acceptance also see [final_smoke.json](final_smoke.json).

Model weight sources reuse existing Part 4: Llama 3 revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`; Mistral revision `63a8b081895390a26e140280378bc85ec8bce07a`. Weight and environment information see previously delivered manifest; do not re-count this UI demo as model baseline experiment.

## Browser Operations

Checked via Chrome native interface: page rendering, example population, button submission, input disabled during inference, 30-day conclusion, clearing old results after changing to 60-day, and new results with new evidence. Record see [browser_checks.json](browser_checks.json). When Chrome connector unavailable, used native application operations; did not use interface calls to fake all browser click checks.

## Not Yet Covered

- CUDA/CPU/MPS physical model inference, fresh installation on another machine, demo venue network and hardware.
- Independent manual hallucination rate, full component ablation, answer completeness, abstention precision, and revision success rate.
- Mistral's 60-day input still produces invalid format. This is preserved model failure; official demo defaults to Llama 3 which passed above cases; cannot promise all new inputs succeed.

Launch method, functionality, and integration boundaries see [UI Handoff](../../UI_HANDOFF.md).
