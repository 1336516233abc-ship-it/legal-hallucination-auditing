# Page Acceptance Record

## Original Entry Reproduction

Original code version is branch li-xinru/part4-experiments commit 29b1d4643f7b40f654c3d525d0d9cceb3e2c67b3. Local source commit 382d6d9057c5018717cbb36c6f31d465d289f42a has identical tree.

Original ui/app.py did not pass dataset when calling Agent.answer, entering old path. Old path only indexes input context when indexed=False, so after changing contract on same instance, old index is still used.

After connecting real Mistral + MLX to unmodified UI callback:

|Acceptance Input|Actual Answer|Evidence Belongs to Current Contract|Seconds|
|---|---|---|---:|
|First input 30-day notice|30 days|Yes|43.66|
|Same instance changed to 60-day notice|30 days|No|40.25|
|New instance input 60-day notice|60 days|Yes|42.43|

Original record: [real.json](acceptance/real.json). To reuse local weights, skipped original UI constructor's CUDA/Transformers initialization; cannot claim this test passed native production startup path.

## Candidate Entry

Candidate creates independent Agent, index, and task state for each request, explicitly using NLI/CUAD protocol. It displays all numbered evidence in final model input, intercepts parsing/grounding errors, and explicitly warns that final semantic review not yet complete. Only shares model weights, processes requests serially; call records limited to most recent.

Under same instance, always asking “Can either party terminate with 30 days notice”, contract changed from 30 days to 60 days:

|Model|Normal Input|Changed Condition Input|
|---|---|---|
|Llama 3|entailment, 28.50 seconds|contradiction, 35.37 seconds|
|Mistral|entailment, format and grounding passed|invalid label, display stopped with error feedback|

Both models used current contract. Mistral's format issue remains unresolved; format pass and valid numbering do not equal semantic correctness. Original records: [candidate_llama3.json](acceptance/candidate_llama3.json), [candidate_real.json](acceptance/candidate_real.json). These 4 requests are not baseline performance estimates, nor can they provide stable latency promises.

Both candidate web launches returned HTTP 200; actual Gradio API empty input requests correctly returned missing input prompts. Normal/changed condition real model requests passed same callback test, not separately repeated via browser clicks. Manual blind review page has checked data, script syntax, and all evidence intervals; browser visual verification at that time not completed due to control connection exception.

## Simulated Failures and Automated Checks

[fixture.json](acceptance/fixture.json) is original entry's fixed output/fault injection record, not real model performance. Empty input and model exceptions can be handled; old entry still displays non-compliant generated text and out-of-bounds citations. That original entry HTTP smoke test failed due to sandbox port access, cannot count as web acceptance passed.

Candidate and blind review tools have 12 checks: current contract isolation, full call token counting, invalid label and out-of-bounds citation stopping, model exception and empty input feedback, blank review metrics null, Unicode interval and adjudication pending handling, duplicate/missing IDs, incorrect citations, missing judgments, rejection of incomplete claims. Synthetic scores only used in test memory, not written to real human blind review files.

Published copy check results see [tests.txt](acceptance/tests.txt) and [tests.xml](acceptance/tests.xml). Machine-specific paths in source files removed, call parameters changed to explicitly pass local model directory.

## Not Yet Completed

- Team official entry integration, final submission package fresh environment installation testing per README.
- Real human blind review, disagreement adjudication, and non-team member user testing as proposed.
- Long-duration load, concurrent service, hard timeout, cross-device speed and cost verification.
- Independent semantic judgment of revised answers again, and audit of server full item-by-item results.

Safe stopping not counted as correct answers, nor internal verifier judgments counted as manual ground truth.
