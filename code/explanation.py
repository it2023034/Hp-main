import json
import re
from code.prompts import build_explanation_prompt


def load_triples_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    triples = []
    for item in data:
        triple = item.get("triple", {})
        entity = triple.get("entity", "").strip()
        attribute = triple.get("attribute", "").strip()
        value = triple.get("value", "").strip()

        if entity and attribute and value:
            triples.append((entity, attribute, value))

    return triples


def extract_explanation_for_triple(dialogue_text, triple, llm):
    prompt = build_explanation_prompt(dialogue_text, triple)

    raw = llm.invoke(prompt).strip()

    for marker in ["<|assistant|>", "TASK:", "RULES:", "DIALOGUE:", "TRIPLE:", "OUTPUT:"]:
        if marker in raw:
            raw = raw.split(marker)[-1].strip()

    text = " ".join(raw.split()).strip()

    if "Explanation:" in text:
        text = text.split("Explanation:", 1)[-1].strip()

    entity, attribute, value = triple
    triple_text = f"{entity} | {attribute} | {value}"
    if triple_text in text:
        text = text.replace(triple_text, "").strip()

    match = re.search(r"^.*?[.!?](?=\s|$)", text)
    if match:
        text = match.group(0).strip()

    return text


def build_explanations(dialogue_text, triples, llm):
    results = []

    for triple in triples:
        entity, attribute, value = triple

        explanation = extract_explanation_for_triple(dialogue_text, triple, llm)

        results.append({
            "triple": {
                "entity": entity,
                "attribute": attribute,
                "value": value
            },
            "explanation": explanation
        })

    return results