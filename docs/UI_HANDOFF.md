# Official UI Deployment and Acceptance

Owner: Li Xinru. As confirmed by the latest division of labor, the UI is included in the mandatory demo deliverables. The entry point is `ui/app.py`, with historical candidates and old acceptance records preserved in their original locations. This is a deployment handoff, not the final course report.

## What the Interface Can Do

- Verify Claims: Returns supported, contradicted, or not_enough_information task conclusions for English statements.
- Find Clauses: Extracts original contract sentences for English questions; failure to retrieve related clauses does not prove the entire contract lacks such clauses.
- Each request rebuilds the Agent and index for the current contract without reusing content from the previous contract. Modifying input immediately clears old results; input and example buttons are disabled during execution.
- Displays all numbered evidence actually used in the final answer, as well as the initial answer check and one revision status. The semantics of the final revision have not been re-verified; the interface provides explicit warnings.
- When model output format, extracted sentence, or citation range checks fail, answer display stops while preserving failure reasons and original records. Failures are not counted as correct or zero hallucination.
- View total call count, total generated tokens, and current trace; does not use the last call to misrepresent total cost. Results belong to the current browser request and do not save a shared “previous contract”.

## Installation and Launch

Use Python 3.12. Model weights and virtual environments are not committed to GitHub. First install common dependencies and the verified UI version:

```bash
python -m pip install -r requirements-retrieval.txt -r requirements-ui.txt
```

Apple Silicon local deployment:

```bash
python -m pip install 'mlx-lm==0.32.2' 'mlx==0.32.4' 'transformers==5.24.0'
python ui/app.py --model llama3 --backend mlx \
  --snapshot /path/to/local/llama3-model --port 7860
```

Mistral likewise uses `--model mistral` and specifies the corresponding model directory. MLX models are loaded and inferred in a dedicated worker thread to avoid Gradio worker threads using MLX streams created by other threads. Model family mismatches, missing directories, or non-Apple Silicon platforms will produce errors at startup.

CUDA servers use the repository's Transformers interface:

```bash
python -m pip install 'torch>=2.0.0' 'transformers==5.24.0' 'accelerate>=0.26.0'
python ui/app.py --model llama3 --backend transformers --device cuda \
  --snapshot /path/to/local/llama3-model --port 7860
```

When `--snapshot` is not provided, Transformers loads according to the model name in `configs/model_config.yaml`; this may require downloading and existing model access permissions. `--device auto` selects CUDA, MPS, CPU in sequence; specifying an unavailable device explicitly will produce an error. CPU inference may be very slow.

Access via browser at `http://127.0.0.1:7860`. Remote servers can be accessed through user-established SSH local port forwarding. By default only listens on localhost and does not create public share links. Exit by pressing Ctrl+C in the launch terminal.

Temperature is 0, seed is 42, input budget defaults to 4096, output maximum 512 tokens; retrieval is BM25 with 512-word chunks and 128-word overlap. The interface supports disabling evidence checking or revision, but disabling checking also disables revision which depends on it. Examples are all manually constructed inputs, not added to the formal test set.

## Demo Operations

1. Click “30-day notice”, then click “Verify current contract”, view conclusions and original text evidence.
2. Click “Change to 60-day notice”: old results should clear immediately. Run again and check whether evidence truly comes from the new 60-day contract.
3. Click “Find liability cap”, run and check extracted sentences.
4. Expand “View checks and execution record”, explaining initial check, whether revised, failure status, and token statistics. Do not interpret “format passed” as “no hallucination”.
5. You can modify English input yourself. The model may answer incorrectly or return invalid format; the interface should provide honest feedback and cannot substitute pre-filled answers for actual execution.

## Validation and Scope

Acceptance entry point: `scripts/validate_ui_service.py`. First launch the official interface, then run:

```bash
python scripts/validate_ui_service.py --url http://127.0.0.1:7860 \
  --output /path/to/new-ui-acceptance.json
python -m pytest -q tests docs/validation/part4_2026-11-23/test_local_delivery.py
```

The acceptance script saves output item by item and refuses to overwrite old files. Model case success count, execution errors, empty input rejection, and browser operations should be explained separately; successful script invocation does not equal correct model conclusions.

This record is placed in [ui_2026-11-23](validation/ui_2026-11-23/). First real service invocation discovered MLX worker thread errors; failure records are still retained; final results see acceptance instructions in that directory. Subsequent [submission package acceptance](validation/final_submission_acceptance.json) has completed extraction, installation, and Mistral service acceptance in a fresh Python 3.11 environment on the same Mac, recording 188 tests passed. Original pilot and subsequent environment versions are separately retained. CUDA/CPU/MPS physical inference and another machine have not yet been accepted; cannot claim cross-platform validation is complete.

## Integration Boundaries and Remaining Items

The official UI reuses existing Agent and task protocols without modifying `agent/`, `retrieval/`, or original data. The entry point preserves the original four return values of `ContractReviewUI.analyze_contract`; new logic is in `ui/service.py` and `ui/backends.py`. The Gradio interface endpoint is `/review`.

Experiment orchestration, main results, error cases, and official UI have been merged into main; course submission package and PDF entry point see [Final Submission Instructions](FINAL_SUBMISSION.md). Server item-by-item output has been obtained, [comprehensive analysis and computable metrics](validation/part4_2026-11-24/README.md) have been completed; independent hallucination rate still requires Yin Fengling and the team to confirm evaluation method and complete annotation; this UI acceptance is not manual blind review. Manual scoring remains incomplete; report and UI instructions have been corrected according to existing acceptance records, see [FINAL_REPORT.md](../FINAL_REPORT.md).
