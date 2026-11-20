"""Select a small, document-disjoint development cohort from the official CUAD archive."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from zipfile import ZipFile

from retrieval.common import ROOT, load_examples, write_json


MEMBER = "train_separate_questions.json"
CLAUSE_TYPES = ("Agreement Date", "Notice Period To Terminate Renewal")
SEED = 42


def digest(text):
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def prepare(archive, output, manifest_path):
    from_test = load_examples("cuad")
    test_titles = {row.metadata["document_title"].strip().casefold() for row in from_test}
    test_contexts = {digest(row.context) for row in from_test}
    with ZipFile(archive) as source:
        train = json.load(source.open(MEMBER))["data"]

    pools = defaultdict(list)
    for contract in train:
        title = contract["title"].strip()
        if title.casefold() in test_titles:
            continue
        for paragraph in contract["paragraphs"]:
            context = paragraph["context"]
            if digest(context) in test_contexts:
                continue
            questions = defaultdict(list)
            for question in paragraph["qas"]:
                base_id, _, _ = question["id"].rpartition("_")
                clause_type = base_id.rsplit("__", 1)[-1]
                if clause_type in CLAUSE_TYPES:
                    questions[(base_id, clause_type)].append(question)
            for (base_id, clause_type), rows in questions.items():
                gold = sorted({(answer["answer_start"], answer["text"]) for row in rows for answer in row["answers"]})
                if any(not text.strip() or context[start:start + len(text)] != text for start, text in gold):
                    continue
                label = "present" if gold else "absent"
                pools[clause_type, label].append({
                    "id": f"cuad_dev_{base_id}", "query": rows[0]["question"], "context": context,
                    "answer": "; ".join(text for _, text in gold) if gold else None, "label": label,
                    "evidence_spans": [[start, start + len(text)] for start, text in gold],
                    "metadata": {"dataset": "cuad", "split": "dev", "document_title": title,
                                 "clause_type": clause_type},
                })

    rng = random.Random(SEED)
    selected = []
    selected_titles = set()
    for key in [(kind, label) for kind in CLAUSE_TYPES for label in ("present", "absent")]:
        available = [row for row in sorted(pools[key], key=lambda row: row["id"])
                     if row["metadata"]["document_title"].casefold() not in selected_titles]
        if not available:
            raise ValueError(f"No independent source document for {key}")
        row = rng.choice(available)
        selected.append(row)
        selected_titles.add(row["metadata"]["document_title"].casefold())

    write_json(output, selected)
    manifest = {"archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "source_member": MEMBER,
                "source_contracts": len(train), "seed": SEED, "test_overlap": 0,
                "eligible_rows": {f"{kind} / {label}": len(rows) for (kind, label), rows in sorted(pools.items())},
                "samples": [{"id": row["id"], "gold_label": row["label"],
                             "source_chars": len(row["context"]), "source_sha256": digest(row["context"]),
                             "evidence_spans": row["evidence_spans"]} for row in selected]}
    write_json(manifest_path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/raw/cuad/official-v1.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/cuad_dev.json")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.archive, args.output, args.manifest)


if __name__ == "__main__":
    main()
