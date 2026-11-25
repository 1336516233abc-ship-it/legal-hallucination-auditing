# Part 4 Supplementary Acceptance and Manual Review Materials

Organized by Li Xinru on 2026-11-23. This directory provides independent interface candidate, real execution records, and blind review forms to be filled by humans, for team members to check and decide on integration. Original ui/app.py and Agent module remain as-is.

## See Conclusions First

**Identified issue: After original page changes to a different contract, may still cite previous contract in answer.** In real Mistral acceptance, first input “30-day advance notice”, then changed to “60-day advance notice”, still answered 30 days; only new instance answered 60 days. Reason is old entry reuses retrieval index established on first use.

[local_demo.py](local_demo.py) creates independent Agent/index for each request, calls existing task protocol, and provides stop prompts for format or citation errors. This is candidate entry, did not replace team original page.

|Real Acceptance|Normal Request|After Changing to 60-Day Notice|
|---|---|---|
|Original page callback / Mistral|Answered 30 days|Still answered 30 days, cited old contract|
|Candidate / Llama 3|Correctly judged 30-day claim supported|Correctly judged 30-day claim contradicts current contract|
|Candidate / Mistral|Correctly judged 30-day claim supported|Invalid label appeared, intercepted and displayed stop reason|

These are manually constructed functional acceptance inputs, total 7 real model requests, not added to existing 60 formal pilot items. Failure records preserved, do not write “intercepted” as “correct answer”. Details in [UI_ACCEPTANCE.md](UI_ACCEPTANCE.md).

## This Directory Contents

- [RQ_FINDINGS.md](RQ_FINDINGS.md): Supported conclusions for RQ1–RQ3 and old demo wording revision suggestions. Independent hallucination rate, complete revision success rate not yet measured.
- [local_demo.py](local_demo.py): Local demo candidate; per-request contract isolation, displays all evidence in final input, failure reasons, and complete call token counts.
- [human_review/reviewer/评审说明.md](human_review/reviewer/评审说明.md): Blind review rules for 60 outputs; 2 models × 3 variants × same 10 questions, not 60 independent questions.
- [human_review/reviewer/review.html](human_review/reviewer/review.html): Download and open in browser. GitHub file page will not execute this form; download this branch ZIP then open.
- [score_reviews.py](score_reviews.py): Validates human score IDs, exact original text intervals, and judgment completeness; statistics for single score. Blank metrics remain null, does not generate scores or dual-person adjudication.
- [test_local_delivery.py](test_local_delivery.py): Request isolation, error handling, and review boundary tests. Test uses synthetic data, not counted in real model or manual results.

Blind review identity mapping and gold reference retained at coordinator's local, not placed in this directory. Reviewers only get reviewer subdirectory, should not read existing case analysis and source traces first. Public repository cannot prevent active searching for method identity, so this is identity-hidden material package, does not claim strict reviewer isolation.

## Running Candidate

First configure repository's existing Python and retrieval environment. This candidate's real inference depends on **Apple Silicon and MLX**; CUDA/Linux backends not yet adapted. Need to prepare locally-owned original model weights yourself; this directory does not contain weights, credentials, or virtual environment.

Run in repository root:

```bash
python -m pip install -r requirements-retrieval.txt
python -m pip install -r docs/validation/part4_2026-11-23/requirements-ui.txt
python docs/validation/part4_2026-11-23/local_demo.py \
  --snapshot /path/to/original-model-snapshot --port 7862
```

Replace placeholder path with existing model directory. Open http://127.0.0.1:7862, test with normal and changed-condition examples on page; Ctrl+C to stop. Only listens on localhost, does not generate public share link.

Tested Llama 3 revision: 8afb486c1db24fe5011ec46dfbe5b5dccdb575c2; Mistral revision: 63a8b081895390a26e140280378bc85ec8bce07a. Temperature 0, seed 42, input budget 4096, output budget 512; BM25 fixed_length, 512 whitespace-delimited words per chunk, 128-word overlap, at most one revision. Did not re-verify semantics after revision; page explicitly states this.

At publication, only adjusted repository location, model directory parameters, and cache paths; inference and judgment logic unchanged. Original records and published file source information in [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json). Installation list is readable direct dependencies; complete runtime environment versions also see existing Part 4 manifest; did not claim completed cold-start installation acceptance on a new computer.

## Check Commands

```bash
python -m pytest -q tests docs/validation/part4_2026-11-23/test_local_delivery.py
python docs/validation/part4_2026-11-23/score_reviews.py \
  --packet docs/validation/part4_2026-11-23/human_review/reviewer/packet.json \
  --ratings /path/to/actual-human-ratings.json \
  --output /path/to/new-score-file.json
```

Rating file and output path need replacement with actual locations; script refuses to overwrite existing results. Current manual rating count is 0. Requires two independent human reviews, discussion and recording of disagreements, then confirm official criteria.

## Handoff Boundaries

Please have interface/Agent maintainers review request state isolation, task entry, and error feedback, then decide on merge approach. Server full item-by-item output still needs delivery before conducting formal paired support assessment of Direct/RAG/Agent. Existing archived results cannot support “100% revision success” demo conclusion.

This iteration used Codex assistance to organize code, acceptance, blind review materials, and conclusion explanations; real model call prompts and outputs saved in acceptance JSON. Human ratings and user testing not yet conducted, final report not yet written. This directory is not complete course submission package or complete AI prompt statement.
