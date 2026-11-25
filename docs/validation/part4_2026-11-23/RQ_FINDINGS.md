# Part 4 Research Findings and Demo Wording Verification

Existing results support “benefits vary by model and task; retrieval hits, label correctness, and answer groundedness are different verification layers.” At this stage, cannot claim to have proven the system reduced independently measured hallucination rates, nor claim 100% revision success rate. This file for results summary and demo revision use, not the course report.

## Evidence Scope

- Archived main experiments: 2 models × 2 datasets × Direct, RAG, Agent, total 18,666 system outputs. Verified summaries and confusion matrices; server item-by-item output not yet retrieved, so only archived metric comparison here.
- New retrieval evaluation runs: 36 groups complete test set configurations. ContractNLI 2,091 questions, 1,188 with gold evidence; CUAD 1,020 questions, 599 with gold evidence, covering selected 10 classes.
- Component ablation: Fixed 10 questions, 2 models × 3 variants, total 60 outputs. Belongs to paired pilot, cannot claim 60 independent questions or full component ablation.
- Error cases: Existing 9 purposefully selected cases, suitable for explaining mechanisms, not suitable for estimating overall proportions of error types.

Data sources: [Published Experiment Summary](https://github.com/1336516233abc-ship-it/legal-hallucination-auditing/blob/29b1d4643f7b40f654c3d525d0d9cceb3e2c67b3/docs/validation/part4_2026-11-22/RESULTS.md), [Archived Main Experiment Verification](https://github.com/1336516233abc-ship-it/legal-hallucination-auditing/blob/29b1d4643f7b40f654c3d525d0d9cceb3e2c67b3/docs/validation/part4_2026-11-22/archived_result_audit/results_tables.md), [Case-by-Case Analysis](https://github.com/1336516233abc-ship-it/legal-hallucination-auditing/blob/29b1d4643f7b40f654c3d525d0d9cceb3e2c67b3/docs/validation/part4_2026-11-22/CASE_ANALYSIS.md). Input hashes see previous day's case_analysis.json; this code source see SOURCE_PROVENANCE.json.

## RQ1 Does Retrieval Augmentation Reduce Hallucinations

**Currently cannot directly answer hallucination rate changes.** Independent support judgment for each method is missing; label accuracy and evidence recall cannot substitute for hallucination rate.

In archived label accuracy, RAG compared to Direct only improves in Mistral × CUAD, other three groups decline.

|Model and Dataset|Direct|RAG|Change (percentage points)|
|---|---:|---:|---:|
|Mistral / ContractNLI|43.42%|41.89%|-1.53|
|Mistral / CUAD|67.06%|69.71%|+2.65|
|Llama 3 / ContractNLI|47.01%|42.52%|-4.50|
|Llama 3 / CUAD|77.55%|68.63%|-8.92|

Statement available for reporting: “Retrieval augmentation improves label performance in some settings, but not universal benefit. We have not yet completed independent hallucination rate comparison across three methods.”

Regarding retrieval evidence, BM25 clause-level Recall@5 is 98.62% for NLI, 64.24% for CUAD; they only use questions with gold evidence as denominator. CUAD still has room for missed detections; NLI's high recall does not guarantee label and explanation consistency. Current evidence cannot derive causality or optimal retriever from this.

## RQ2 Does Verification and Revision Improve Standard RAG

Archived Agent compared to RAG improves by 13.01 percentage points on Llama 3 × ContractNLI, declines on other three groups.

|Model and Dataset|RAG|Agent|Change (percentage points)|
|---|---:|---:|---:|
|Mistral / ContractNLI|41.89%|37.06%|-4.83|
|Mistral / CUAD|69.71%|63.24%|-6.47|
|Llama 3 / ContractNLI|42.52%|55.52%|+13.01|
|Llama 3 / CUAD|68.63%|62.06%|-6.57|

This is descriptive comparison of label performance; cannot claim hallucination reduction or statistical significance. In archived means, Agent/RAG time ratio is about 4.29–5.36; cannot directly reuse old material's “only 2–4× slower”; these server results also cannot directly mix-compare with local MLX timing.

In paired component pilot, Llama 3 × NLI's Full label correct count is 6/6, other two variants 5/6; other model-task combination label correct counts unchanged. Of 20 Full outputs, 11 revisions occurred: 1 label wrong-to-right, 4 wrong-to-wrong, 6 right-to-right; 2 revisions added out-of-bounds citations. Only label correction case still has evidence pointing to error.

Statement available for reporting: “Benefits from verification and revision depend on model and task. Pilot observed one label correction, also found revision may introduce citation errors; complete revision success rate not yet measured.”

No verification simultaneously disables verification and revision dependent on verification; it is not “removing only one independent module” strict single-factor intervention. No revision retains verification. This design must be noted beside ablation table.

## RQ3 Where Are Errors Concentrated

Existing 9 cases can illustrate following failure modes, cannot yet answer “which clause types most error-prone” overall ranking.

|Observation|Recorded Cases|Supported Conclusions|
|---|---|---|
|After retrieval recovers gold, still answers absent|C03, C04|Retrieval success is not sufficient condition for correct generation|
|Explanation text contradicts label|C02|Need to review both task label and explanation|
|After revision, numbering still exists but points to different content|C01|Checking only citation number range is insufficient|
|Placeholder in original text treated as answer to question|C06|Verbatim extraction pass does not equal task-relevant|
|Output format failure|C05|Must retain failure denominator and stop status|
|New or existing out-of-bounds citations|C07–C09|Correct label may still fail to provide usable display answer|

These are AI-assisted, original-text-based case verifications, not independent manual review. Did not treat internal verifier status as manual ground truth.

## Suggested Replacements for Old Demo Statements

Following are local revision suggestions, preserving original files, for team review before deciding to replace.

|Original Location|Original Statement or Issue|Suggested Replacement|
|---|---|---|
|DEMO_PREP_FINAL.md:77, 112, 138; DEMO_SCRIPT.md:99, 181; DEMO_CHECKLIST.md:161–169|100% hallucination revision success|Independent revision success rate not yet assessed; pilot 11 revisions had only 1 label correction, with citation residual issues|
|DEMO_SCRIPT.md:98, 126–133|Treat verifier detection count as true hallucination rate|Only call “verifier trigger count”, separately list support metrics awaiting manual measurement|
|DEMO_PREP_FINAL.md:82; DEMO_SCRIPT.md:182|Only adds 9–14% tokens|Old statistics have criteria issues; before unified complete call counting, do not make this cost conclusion|
|DEMO_SCRIPT.md:180|Already proven to significantly reduce hallucinations|Currently only observed label benefits in specific settings, independent hallucination rate still missing|
|DEMO_PREP_FINAL.md:63|Llama-3.1-8B|This Part 4 is Meta-Llama-3-8B-Instruct, revision 8afb486c1db24fe5011ec46dfbe5b5dccdb575c2|
|DEMO_SCRIPT.md:3; DEMO_PREP_FINAL.md:148; DEMO_CHECKLIST.md:101|15–20 minute demo|Per verified course notice, control to within 8 minutes, including 2-minute pitch; prepare operations that can change input on-site|
|Illustrated “30-day” successful revision scenario|Readers may mistake for formal dataset actual test|Mark as manually constructed input; actual records separately attach model output and traces, do not fabricate successful revision process|

## Usable Two-Minute Narration Draft

“We provide retrieval evidence, answer generation, claim checking, and at most one revision workflow for users needing to verify contract clauses. We compared two models, two datasets, and three methods. Current results show Agent is not comprehensively superior: Llama 3 on ContractNLI label accuracy improved from RAG's 42.52% to 55.52%, but other three groups declined. Retrieval hits and label correctness also do not guarantee answers have sufficient grounding. We supplemented paired component pilot, recorded cases of revision still citing incorrect evidence, format failures, etc. Independent manual hallucination rate still awaiting review. On-site will demonstrate normal requests, behavior after modifying contract conditions, and situations where erroneous output display is stopped.”

Before use, adjust last sentence per actual demo candidate verification records; cannot claim team official entry already completed fixes.
