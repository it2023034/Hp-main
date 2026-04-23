from code.model import qwen
from code.prompts import build_triple_extraction_prompt, build_explanation_prompt
from code.explanation import extract_explanation_for_triple
from code.utils import (
    load_file,
    parse_triples,
    extract_allowed_relations,
    filter_triples,
    ttl_to_metapaths,
    infer_types_from_schema,
    is_message_level_triple,
    remove_duplicates
)
import json

def main():
    schema = ttl_to_metapaths("files/GANNDALF-onto.ttl")

    with open("files/schema_from_ttl.txt", "w", encoding="utf-8") as f:
        f.write(schema)

    example_out = load_file("files/example_out.txt")
    example_dialogue = load_file("files/example.txt")
    case_dialogue = load_file("files/case.txt")

    prompt = build_triple_extraction_prompt(
        schema,
        example_dialogue,
        example_out,
        case_dialogue
    )
    result = qwen.invoke(prompt)

    triples = parse_triples(result)

    allowed_relations = extract_allowed_relations(schema)
    filtered_triples = filter_triples(triples, allowed_relations)

    inferred_triples = infer_types_from_schema(filtered_triples, schema)
    all_triples = filtered_triples + inferred_triples
    all_triples = remove_duplicates(all_triples)

    with open("files/triples_ttl.json", "w", encoding="utf-8") as f:
        json.dump(all_triples, f, ensure_ascii=False, indent=2)

    results = []
    for item in all_triples:
        triple = item["triple"]

        if item.get("inferred") or is_message_level_triple(triple):
            explanation = ""
        else:
            explanation = extract_explanation_for_triple(
                case_dialogue,
                (triple["entity"], triple["attribute"], triple["value"]),
                qwen
            )

        entry = {
            "triple": triple,
            "explanation": explanation
        }

        results.append(entry)

    with open("files/triples_explanations.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()