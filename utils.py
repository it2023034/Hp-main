import json
from rdflib import Graph, RDF, RDFS, OWL

def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def parse_triples(raw_output):
    triples = []

    for line in raw_output.splitlines():
        line = line.strip()

        if line.count("|") != 2:
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            continue

        triples.append({
            "triple": {
                "entity": parts[0],
                "attribute": parts[1],
                "value": parts[2]
            }
        })

    return triples

def extract_allowed_relations(schema_text):
    relations = set()

    for line in schema_text.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3:
            relations.add(parts[1])

    return relations

def normalize_relation(r):
    r = r.strip()

    new = ""
    for c in r:
        if c.isupper():
            new += "_" + c.lower()
        else:
            new += c

    return new.lstrip("_")

def filter_triples(triples, allowed_relations):
    filtered = []

    normalized_allowed = {normalize_relation(r) for r in allowed_relations}

    for triple in triples:
        attr = normalize_relation(triple["triple"]["attribute"])

        if attr in normalized_allowed:
            triple["triple"]["attribute"] = attr
            filtered.append(triple)

    return filtered

def infer_types_from_schema(triples, schema_text):
    inferred = []
    seen = set()

    primitive_types = {
        "string", "int", "integer", "float", "double",
        "boolean", "dateTime", "dateTimeStamp", "datetime", "Datetime"
    }

    schema_mappings = []
    for line in schema_text.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            continue

        domain, relation, range_ = parts
        schema_mappings.append((domain, normalize_relation(relation), range_))

    for item in triples:
        t = item["triple"]
        seen.add((t["entity"].strip(), t["attribute"].strip(), t["value"].strip()))

    for item in triples:
        t = item["triple"]
        subj = t["entity"].strip()
        rel = normalize_relation(t["attribute"])
        obj = t["value"].strip()

        for domain, relation, range_ in schema_mappings:
            if rel != relation:
                continue

            key1 = (subj, "rdf:type", domain)
            if key1 not in seen:
                seen.add(key1)
                inferred.append({
                    "triple": {
                        "entity": subj,
                        "attribute": "rdf:type",
                        "value": domain
                    },
                    "inferred": True
                })

            if range_ not in primitive_types:
                key2 = (obj, "rdf:type", range_)
                if key2 not in seen:
                    seen.add(key2)
                    inferred.append({
                        "triple": {
                            "entity": obj,
                            "attribute": "rdf:type",
                            "value": range_
                        },
                        "inferred": True
                    })

    return inferred

def is_message_level_triple(triple):
    message_level_relations = {"sender", "receiver", "timestamp", "message_has_content"}

    entity = triple["entity"].strip()
    attribute = triple["attribute"].strip()

    return entity.startswith("Message") and entity[7:].isdigit() and attribute in message_level_relations

def short(uri):
    uri = str(uri)
    if "#" in uri:
        return uri.split("#")[-1]
    return uri.split("/")[-1]

def ttl_to_metapaths(path):
    g = Graph()
    g.parse(path, format="turtle")

    rows = []

    for prop in g.subjects(RDF.type, OWL.ObjectProperty):
        domain = g.value(prop, RDFS.domain)
        range_ = g.value(prop, RDFS.range)

        if domain and range_:
            rows.append(f"{short(domain)} | {short(prop)} | {short(range_)}")

    for prop in g.subjects(RDF.type, OWL.DatatypeProperty):
        domain = g.value(prop, RDFS.domain)
        range_ = g.value(prop, RDFS.range)

        if domain and range_:
            rows.append(f"{short(domain)} | {short(prop)} | {short(range_)}")

    return "\n".join(rows)

def remove_duplicates(triples):
    seen = set()
    result = []

    for item in triples:
        t = item["triple"]
        key = (t["entity"], t["attribute"], t["value"])

        if key not in seen:
            seen.add(key)
            result.append(item)

    return result