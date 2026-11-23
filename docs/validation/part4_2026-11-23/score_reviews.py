"""Validate actual human annotations; never fill missing judgments or treat them as zero."""

import argparse
from collections import Counter
import json
from pathlib import Path


def score(packet, reviews):
    items = {item["blind_id"]: item for item in packet["items"]}
    ids = [review.get("blind_id") for review in reviews]
    if len(ids) != len(set(ids)) or set(ids) != set(items):
        raise ValueError("Ratings must contain every packet ID exactly once")
    types, verdicts = Counter(), Counter()
    complete, substantive, unsupported_answers, adjudication = 0, 0, 0, 0
    for review in reviews:
        if not review.get("completed"):
            continue
        item = items[review["blind_id"]]
        if not isinstance(review.get("annotator"), str) or not review["annotator"].strip():
            raise ValueError("Completed annotations require an actual reviewer identifier")
        kind = review.get("response_type")
        if kind not in {"format_failure", "withheld_error", "abstention", "substantive"}:
            raise ValueError("Invalid or missing response type")
        if review.get("needs_adjudication") not in (True, False) or review.get("needs_adjudication") is None:
            raise ValueError("Choose whether adjudication is needed")
        if kind == "format_failure" and item["format_pass"]:
            raise ValueError("Format failure disagrees with the frozen parsing result")
        if kind == "withheld_error" and (not item["format_pass"] or not item["display_suppressed"]):
            raise ValueError("Withheld-error applies to parsed outputs suppressed by a grounding error")
        if kind in {"abstention", "substantive"} and (not item["format_pass"] or item["display_suppressed"]):
            raise ValueError("Do not classify a suppressed response as a displayed substantive answer")
        claims = review.get("claims", [])
        if kind == "substantive" and (not claims or review.get("all_claims_reviewed") is not True):
            raise ValueError("Substantive answers require all atomic claims, including the task conclusion")
        if kind != "substantive" and claims:
            raise ValueError("Non-substantive outputs are counted separately and must not contain scored claims")
        local_verdicts = Counter()
        source = packet["contracts"][item["contract_id"]]
        for claim in claims:
            verdict = claim.get("verdict")
            if verdict not in {"supported", "contradicted", "not_enough_information"}:
                raise ValueError("Every completed claim requires an explicit human verdict")
            if not claim.get("claim", "").strip() or not claim.get("rationale", "").strip():
                raise ValueError("Claim text and justification are required")
            evidence = claim.get("evidence_spans", [])
            if verdict in {"supported", "contradicted"} and not evidence:
                raise ValueError("Supported and contradicted judgments require exact source evidence")
            for span in evidence:
                start, end = span.get("start"), span.get("end")
                if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source):
                    raise ValueError("Invalid Unicode source offsets")
                if span.get("quote") != source[start:end]:
                    raise ValueError("Quoted evidence differs from the source at the supplied offsets")
            local_verdicts[verdict] += 1
        if review["needs_adjudication"]:
            adjudication += 1
            continue
        complete += 1
        types[kind] += 1
        verdicts.update(local_verdicts)
        if kind == "substantive":
            substantive += 1
            unsupported_answers += bool(local_verdicts["contradicted"] + local_verdicts["not_enough_information"])
    claims_n = sum(verdicts.values())
    return {"scope": packet["scope"], "review_target": packet["review_target"],
            "status": "unreviewed" if complete == 0 and adjudication == 0 else "single_reviewer_annotations_not_team_adjudicated",
            "outputs": len(items), "complete_without_pending_adjudication": complete,
            "pending_adjudication": adjudication, "remaining": len(items) - complete - adjudication,
            "review_coverage": complete / len(items), "response_types": dict(types), "claim_verdicts": dict(verdicts),
            "claim_denominator": claims_n, "substantive_answer_denominator": substantive,
            "claim_level_unsupported_rate": (verdicts["contradicted"] + verdicts["not_enough_information"]) / claims_n if claims_n else None,
            "answer_level_unsupported_rate": unsupported_answers / substantive if substantive else None,
            "answer_completeness": None, "abstention_precision": None, "independent_revision_success_rate": None,
            "notes": "No model/method decoding or cross-reviewer agreement is inferred; incomplete and withheld responses are not supported claims."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output file to preserve existing results")
    result = score(json.loads(args.packet.read_text()), json.loads(args.ratings.read_text()))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
