from typing import List, Tuple
import dspy
from pydantic import BaseModel


def extraction_sig(
    Relation: BaseModel, is_conversation: bool, context: str = ""
) -> dspy.Signature:
    if not is_conversation:

        class ExtractTextRelations(dspy.Signature):
            __doc__ = f"""Extract subject-predicate-object triples from the source text. 
      Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
      This is for an extraction task, please be thorough, accurate, and faithful to the reference text. {context}"""

            source_text: str = dspy.InputField()
            entities: list[str] = dspy.InputField()
            relations: list[Relation] = dspy.OutputField(
                desc="List of subject-predicate-object tuples. Be thorough."
            )

        return ExtractTextRelations
    else:

        class ExtractConversationRelations(dspy.Signature):
            __doc__ = f"""Extract subject-predicate-object triples from the conversation, including:
      1. Relations between concepts discussed
      2. Relations between speakers and concepts (e.g. user asks about X)
      3. Relations between speakers (e.g. assistant responds to user)
      Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
      This is for an extraction task, please be thorough, accurate, and faithful to the reference text. {context}"""

            source_text: str = dspy.InputField()
            entities: list[str] = dspy.InputField()
            relations: list[Relation] = dspy.OutputField(
                desc="List of subject-predicate-object tuples where subject and object are exact matches to items in entities list. Be thorough"
            )

        return ExtractConversationRelations


def fallback_extraction_sig(
    entities, is_conversation, context: str = ""
) -> dspy.Signature:
    """This fallback extraction does not strictly type the subject and object strings."""

    entities_str = "\n- ".join(entities)

    class Relation(BaseModel):
        # TODO: should use literal's here instead.
        __doc__ = f"""Knowledge graph subject-predicate-object tuple. Subject and object entities must be one of: {entities_str}"""

        subject: str = dspy.InputField(desc="Subject entity", examples=["Kevin"])
        predicate: str = dspy.InputField(desc="Predicate", examples=["is brother of"])
        object: str = dspy.InputField(desc="Object entity", examples=["Vicky"])

    return Relation, extraction_sig(Relation, is_conversation, context)


def get_relations(
    input_data: str,
    entities: list[str],
    is_conversation: bool = False,
    context: str = "",
) -> List[Tuple[str, str, str]]:
    class Relation(BaseModel):
        """Knowledge graph subject-predicate-object tuple."""

        subject: str = dspy.InputField(desc="Subject entity", examples=["Kevin"])
        predicate: str = dspy.InputField(desc="Predicate", examples=["is brother of"])
        object: str = dspy.InputField(desc="Object entity", examples=["Vicky"])

    ExtractRelations = extraction_sig(Relation, is_conversation, context)

    try:
        extract = dspy.Predict(ExtractRelations)
        result = extract(source_text=input_data, entities=entities)
        return [(r.subject, r.predicate, r.object) for r in result.relations]

    except Exception:
        return get_relations_string_fallback(
            input_data=input_data,
            entities=entities,
            is_conversation=is_conversation,
            context=context,
        )


def parse_relation_line(line: str) -> Tuple[str, str, str] | None:
    line = str(line).strip().strip("-").strip()
    if not line:
        return None

    for separator in (" | ", "\t", " -> "):
        parts = [part.strip() for part in line.split(separator)]
        if len(parts) == 3 and all(parts):
            return parts[0], parts[1], parts[2]

    return None


def get_relations_string_fallback(
    input_data: str,
    entities: list[str],
    is_conversation: bool = False,
    context: str = "",
) -> List[Tuple[str, str, str]]:
    entities_str = "\n- ".join(entities)

    class FallbackRelations(dspy.Signature):
        __doc__ = f"""Extract subject-predicate-object triples from the source text.
Subject and object should be exact matches from the provided entities when possible.
Return each relation as one string in this exact format:
subject | predicate | object
Use only faithful relations from the source text. {context}"""

        source_text: str = dspy.InputField()
        entities: list[str] = dspy.InputField(desc=f"Allowed entities:\n- {entities_str}")
        relation_lines: list[str] = dspy.OutputField(
            desc="List of relation lines formatted exactly as: subject | predicate | object"
        )

    extract = dspy.Predict(FallbackRelations)
    result = extract(source_text=input_data, entities=entities)

    relations = []
    entity_set = set(entities)
    for line in result.relation_lines:
        parsed = parse_relation_line(line)
        if parsed is None:
            continue
        subject, predicate, object_value = parsed
        if subject in entity_set and object_value in entity_set:
            relations.append((subject, predicate, object_value))

    return relations
