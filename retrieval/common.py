"""Shared task schemas, prompt budgeting and processed-data helpers."""

import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"contractnli", "cuad"}
LABELS = {"contractnli": {"entailment", "contradiction", "neutral"}, "cuad": {"present", "absent"}}
_CUAD_QUERY = re.compile(r'^Highlight the parts \(if any\) of this contract related to "([^"\n]+)" '
                         r'that should be reviewed by a lawyer\.\s*(?:Details:\s*(.*))?$', re.DOTALL)


class PromptBudgetError(ValueError):
    """Raised when the question and instructions alone exceed the input budget."""


def validate_dataset(dataset):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    return dataset


def task_question(query, dataset):
    """Remove the annotation boilerplate from CUAD generation prompts."""
    match = _CUAD_QUERY.fullmatch(query.strip()) if dataset == "cuad" else None
    if not match:
        return query
    field, details = match.groups()
    return f"Find the contract text for the field {field!r}. {details or ''}".rstrip()


def task_instruction(dataset, *, numbered_evidence=False):
    validate_dataset(dataset)
    if dataset == "contractnli":
        citation = (" Cite evidence numbers within the explanation, for example [1]."
                    if numbered_evidence else "")
        return ("Use the text above to answer this question. A statement that follows from the text is entailment. "
                "A statement that conflicts with it is contradiction. If the text is insufficient, choose neutral. "
                "Return two lines with these exact field names. Here is a format example with an unrelated answer:\n"
                "Label: neutral\nAnswer: The document does not specify a notice period.\n"
                "Now give the label and answer for the actual question."
                + citation)
    example_citations = "[1]" if numbered_evidence else "[]"
    citation_rule = ("Use only the displayed evidence numbers, as integers, for excerpts you copied. "
                     if numbered_evidence else "Use an empty citations array. ")
    return ('Return exactly one JSON object with keys "label", "extracted_spans", and "citations". '
            'Choose "present" when the supplied text provides the requested information, even without a heading. Copy the shortest '
            'complete excerpt that answers the question, preserving words, punctuation, and spacing. '
            'Do not rewrite the excerpt or add an answer field. '
            + citation_rule
            + 'For example, when the source reads "Payments are due within thirty days." and the question asks '
            'when payment is due, the response is:\n'
            f'{{"label":"present","extracted_spans":["within thirty days"],"citations":{example_citations}}}\n'
            'When the requested clause is missing from the input, return:\n'
            '{"label":"absent","extracted_spans":[],"citations":[]}\n'
            'Answer the actual contract question now. Stop immediately after the closing brace.')


# Match a whole JSON object or one final JSON block preceded by prose.
_JSON_OBJECT_START = re.compile(r"\{|^\[|```")
_FENCE = re.compile(r"```(?:json)?[ \t]*\r?\n", re.IGNORECASE)
_LEGACY_FIELD = re.compile(r"^[ \t]*(Answer|Label)[ \t]*:[ \t]*", re.IGNORECASE | re.MULTILINE)
_JSON_DECODER = json.JSONDecoder()


def completed_json_object(text):
    """Return the emitted prefix once a leading JSON object is syntactically complete."""
    prefix = text.lstrip()
    if not prefix.startswith("{") or "}" not in prefix:
        return None
    try:
        _, end = _JSON_DECODER.raw_decode(prefix)
    except json.JSONDecodeError:
        return None
    return prefix[:end]


def generate_task_response(llm, prompt, dataset):
    """Use a JSON completion boundary on backends that support it."""
    generate_json = getattr(llm, "generate_json_for_baseline", None) if dataset == "cuad" else None
    if generate_json is not None:
        return generate_json(prompt)
    return getattr(llm, "generate_for_baseline", llm.generate)(prompt)


def _unique_json_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _response_object(text):
    start = _JSON_OBJECT_START.search(text)
    if start and not _LEGACY_FIELD.match(text):
        body = text[start.start():]
        fence = _FENCE.match(body)
        if fence:
            body = body[fence.end():]
            if not body.endswith("```"):
                raise ValueError("response JSON fence is not closed")
            body = body[:-3]
        try:
            obj = json.loads(body, object_pairs_hook=_unique_json_fields)
        except json.JSONDecodeError as exc:
            raise ValueError("response must contain one complete JSON object") from exc
        if not isinstance(obj, dict):
            raise ValueError("response JSON must be an object")
        return obj

    matches = list(_LEGACY_FIELD.finditer(text))
    if not matches or matches[0].start() != 0:
        raise ValueError("response has no JSON object or Answer/Label fields")
    fields = {}
    for index, match in enumerate(matches):
        name = match.group(1).lower()
        if name in fields:
            raise ValueError(f"duplicate response field: {name}")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields[name] = text[match.end():end].strip()
    return fields


def _citation_numbers(citations):
    if not isinstance(citations, list):
        raise ValueError("citations must be an array of positive integers")
    numbers = []
    for citation in citations:
        if isinstance(citation, dict):
            citation = citation.get("number")
        if type(citation) is not int or citation <= 0:
            raise ValueError("citations must contain positive integer evidence numbers")
        numbers.append(citation)
    return numbers


def parse_response(response, dataset):
    """Normalize explicit fields and reject malformed or ambiguous responses."""
    validate_dataset(dataset)
    answer, spans, citations = "", [], []
    try:
        if not isinstance(response, str):
            raise ValueError("response must be text")
        parsed = _response_object(response.strip())
        label = parsed.get("label")
        if not isinstance(label, str) or label.strip().lower() not in LABELS[dataset]:
            raise ValueError("missing or invalid task label")
        label = label.strip().lower()
        if "extracted_spans" in parsed:
            spans = parsed["extracted_spans"]
            if not isinstance(spans, list) or not all(isinstance(s, str) and s.strip() for s in spans):
                spans = []
                raise ValueError("extracted_spans must contain nonempty strings")
        answer = parsed.get("answer")
        if dataset == "cuad" and "answer" not in parsed and "extracted_spans" in parsed:
            answer = "; ".join(spans)
        if not isinstance(answer, str):
            answer = ""
            raise ValueError("answer must be a string")
        if dataset == "contractnli" and not answer.strip():
            raise ValueError("NLI responses require a nonempty answer")
        if dataset == "cuad" and label == "absent" and (answer.strip() or spans):
            raise ValueError("absent responses must have an empty answer and no extracted spans")
        if dataset == "cuad" and label == "present" and not answer.strip():
            raise ValueError("present responses require an extracted answer")
        inline_citations = [int(n) for n in re.findall(r"\[(\d+)\]", answer)] if dataset == "contractnli" else []
        citations = _citation_numbers(parsed.get("citations", inline_citations))
        if "abstained" in parsed and not isinstance(parsed["abstained"], bool):
            raise ValueError("abstained must be a boolean")
    except ValueError as exc:
        return {"answer": answer.strip(), "label": "invalid", "extracted_spans": spans,
                "citations": citations, "parse_error": str(exc), "abstained": False}
    return {"answer": answer.strip(), "label": label, "extracted_spans": spans,
            "citations": citations, "parse_error": None,
            "abstained": parsed.get("abstained") is True or label == "neutral"}


def count_tokens(tokenizer, text):
    if tokenizer is None:
        return None
    return len(tokenizer.encode(text, add_special_tokens=True))


def check_cuad_extraction(parsed, evidence):
    """Check literal support in the exact text seen by the generator."""
    if parsed["parse_error"]:
        return {"answer_is_extractive": None, "extraction_error": None}
    spans = parsed["extracted_spans"]
    answer = parsed["answer"]
    candidates = list(spans)
    if answer and answer != "; ".join(spans):
        candidates.append(answer)
    valid = all(any(text in source for source in evidence) for text in candidates)
    return {"answer_is_extractive": valid,
            "extraction_error": None if valid else "CUAD output contains text absent verbatim from the model evidence"}


def fit_evidence(chunks, build, tokenizer, max_input_tokens):
    """Fit ranked evidence into a full prompt, retaining source offsets."""
    full_prompt = build(chunks)
    full_tokens = count_tokens(tokenizer, full_prompt)
    if max_input_tokens is None:
        return chunks, full_prompt, {"input_tokens": full_tokens, "untruncated_input_tokens": full_tokens,
                                     "max_input_tokens": None, "truncated": False}
    fit_text("", lambda _: build([]), tokenizer, max_input_tokens)
    used = []
    for chunk in chunks:
        if count_tokens(tokenizer, build(used + [chunk])) <= max_input_tokens:
            used.append(dict(chunk))
            continue
        if count_tokens(tokenizer, build(used + [{**chunk, "text": ""}])) > max_input_tokens:
            break
        prefix, _, _ = fit_text(chunk["text"], lambda text: build(used + [{**chunk, "text": text}]),
                                 tokenizer, max_input_tokens)
        if prefix.strip():
            used.append({**chunk, "text": prefix, "end": chunk["start"] + len(prefix)})
        break
    prompt = build(used)
    return used, prompt, {"input_tokens": count_tokens(tokenizer, prompt),
                           "untruncated_input_tokens": full_tokens, "max_input_tokens": max_input_tokens,
                           "truncated": full_prompt != prompt}


def fit_text(text, build_prompt, tokenizer=None, max_input_tokens=None):
    """Fit a source prefix into a full-prompt token budget, preserving offsets."""
    prompt = build_prompt(text)
    full_tokens = count_tokens(tokenizer, prompt)
    if max_input_tokens is None:
        return text, prompt, {"input_tokens": full_tokens, "untruncated_input_tokens": full_tokens,
                              "max_input_tokens": None, "truncated": False}
    if type(max_input_tokens) is not int or max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be a positive integer or None")
    if tokenizer is None:
        raise ValueError("A tokenizer is required for a token-based input budget")
    if count_tokens(tokenizer, build_prompt("")) > max_input_tokens:
        raise PromptBudgetError("The question and instructions exceed the input token budget")
    if full_tokens <= max_input_tokens:
        return text, prompt, {"input_tokens": full_tokens, "untruncated_input_tokens": full_tokens,
                              "max_input_tokens": max_input_tokens, "truncated": False}
    lo, hi = 0, len(text)
    while lo < hi:
        middle = (lo + hi + 1) // 2
        if count_tokens(tokenizer, build_prompt(text[:middle])) <= max_input_tokens:
            lo = middle
        else:
            hi = middle - 1
    fitted = text[:lo]
    prompt = build_prompt(fitted)
    return fitted, prompt, {"input_tokens": count_tokens(tokenizer, prompt),
                            "untruncated_input_tokens": full_tokens,
                            "max_input_tokens": max_input_tokens, "truncated": lo < len(text)}


def document_id(context, metadata=None, dataset=None):
    metadata = metadata or {}
    if metadata.get("doc_id") is not None:
        return str(metadata["doc_id"])
    source = metadata.get("document_id", metadata.get("document_title"))
    if source is not None:
        return f"{dataset or metadata.get('dataset', 'document')}:{source}"
    return "sha256:" + hashlib.sha256(context.encode("utf-8")).hexdigest()


def example_dict(example):
    return dict(example) if isinstance(example, dict) else vars(example).copy()


def example_dataset(example, default):
    return validate_dataset(example.get("dataset") or (example.get("metadata") or {}).get("dataset") or default)


def select_examples(examples, sample_size=None, seed=42):
    """Select reproducibly, returning the selected rows in source order."""
    if sample_size is None:
        return list(examples)
    if type(sample_size) is not int or sample_size < 0:
        raise ValueError("sample_size must be a nonnegative integer or None")
    indices = sorted(random.Random(seed).sample(range(len(examples)), min(sample_size, len(examples))))
    return [examples[i] for i in indices]


def load_examples(dataset, data_dir=None, split="test"):
    from data.dataset_loader import DatasetLoader
    validate_dataset(dataset)
    return DatasetLoader(str(data_dir or ROOT / "data/processed")).load_processed(f"{dataset}_{split}.json")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
