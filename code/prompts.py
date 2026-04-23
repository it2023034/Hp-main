def build_triple_extraction_prompt(schema, example_dialogue, example_out, case_dialogue):
    return f"""
Task:
Extract knowledge graph triples from the input chat history.

Your goal is to extract as many valid schema-compliant triples as possible from the chat.
Be exhaustive.
Do not stop after the most obvious facts.
If multiple distinct facts fit the schema, include all of them.

The input is a plain text chat export.

Input structure:
- A header may appear, such as:
  - Whatsapp chat history
  - Participants: Name1 and Name2
  - Date range: ...
- Each message line has the form:
  [DD/MM/YYYY, HH:MM] Sender: Message

Participant rules:
- Use the sender name exactly as it appears in the message line.
- The participants are listed in the header.
- If the chat contains exactly two participants, infer the receiver of each message as the other participant.
- Use participant names exactly as they appear in the input.
- Do not invent additional participants.

Extraction rules:
- Extract all facts that are explicitly stated or clearly supported by the chat.
- Be exhaustive within the schema.
- Analyze EACH message independently.
- Consider whether each message expresses one or more schema-valid facts.
- If multiple distinct triples can be supported by the same message, include all of them.
- Use ONLY entity names that appear explicitly in the input.
- Do NOT introduce entities from the example.
- Do NOT infer hidden participants.
- Do NOT infer transactions, accounts, balances, or transfers unless explicitly stated.
- Do NOT output duplicate triples.
- Do NOT generate multiple triples that express the same fact with slight variations.
- Do NOT merge multiple distinct facts into one general triple.
- If a fact clearly fits the schema, include it.
- If a fact does not fit the schema, omit it.

Relation selection rules:
- For each fact, choose the MOST SPECIFIC relation from the schema.
- Do NOT use a generic relation if a more precise schema relation exists.
- Avoid overusing broad relations such as provides when a more precise relation is available.
- Different facts should map to different schema relations when appropriate.
- If multiple interpretations are possible, choose the one closest to the schema.

Participant balance rules:
- Analyze facts expressed by BOTH participants, not only the participant providing information.
- Questions, requests, doubts, clarifications, and inquiries from either participant may also produce valid triples if supported by the schema.
- Do not focus only on offers, guidance, links, or provided resources.
- Treat requester-side facts and provider-side facts as equally important.
- If one participant asks for information, details, support, clarification, registration, safety, timing, or similar concepts, extract the corresponding request-related triples when supported by the schema.

Per-message reasoning rules:
- For each message, determine whether it expresses:
  1. a request or inquiry,
  2. a provided resource, offer, or answer,
  3. a communication relation,
  4. any other schema-valid fact.
- Extract all applicable triples.

Message-level triples:
- If the schema includes message-level relations such as sender, receiver, timestamp, or message_has_content, create message entities named Message1, Message2, Message3, ... in chronological order.
- For each message entity, use the exact timestamp from the chat line.
- Use the exact sender and inferred receiver.
- Preserve the original message order from the input.

Communication triples:
- Use sender/receiver information to infer communication relationships when supported by the schema.

Output requirements:
- Output ONLY triples.
- Do not output explanations.
- Do not output introductory text.
- Do not output bullets.
- Do not output labels other than the final "Output:" line.

Output format:
subject | relation | object

Formatting constraints:
- Exactly one triple per line
- Use exactly one space before and after |
- No empty lines

Allowed schema:

{schema}

Example Input:
{example_dialogue}

Example Output:
{example_out}

Now process this input.

Input:
{case_dialogue}

Output:
""".strip()


def build_explanation_prompt(dialogue_text, triple):
    entity, attribute, value = triple

    return f"""
ROLE:
You are an explanation generator for knowledge graph triples derived from a chat history.
The triple already exists.
Your task is NOT to explain how the triple was extracted,
but to explain why this fact appears in the chat.

CONTEXT:
The data below contains a chat history.
Each triple represents a fact that is explicitly stated or clearly supported by the chat.
You must explain the factual reason why this information is present.

CHAT HISTORY:
{dialogue_text}

TRIPLE:
{entity} | {attribute} | {value}

TASK:
Write ONLY ONE concise sentence explaining why this triple appears in the chat,
based strictly on the provided context.

RULES:
- Explain WHY this fact appears in the chat.
- Focus strictly on the given triple.
- The explanation MUST explicitly mention the given attribute and the given value.
- Base the explanation on the chat context only when necessary.
- Do NOT repeat or summarize the entire chat.
- Do NOT introduce additional events or actions not directly tied to the triple.
- Ensure the explanation matches the exact attribute and value in the triple.
- Do NOT introduce unrelated attributes or values.
- Do NOT add new facts or assumptions.
- Do NOT include unrelated information.
- Do NOT explain the extraction process.
- Do NOT mention models, systems, or analysis steps.
- Output exactly ONE complete sentence ending with a period.
- Maximum ONE sentence.
- Stop immediately after the first sentence.
- Natural language only.
- Keep the explanation concise and specific.

OUTPUT:
""".strip()