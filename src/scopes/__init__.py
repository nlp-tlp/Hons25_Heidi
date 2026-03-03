from .property_text.property_text_scope import PropertyTextScopeGraph, PropertyTextScopeRetriever
from .concept_text.concept_text_scope import ConceptTextScopeGraph, ConceptTextScopeRetriever
from .row_text.row_text_scope import RowTextScopeGraph, RowTextScopeRetriever
from .row_all.row_all_scope import RowAllScopeGraph, RowAllScopeRetriever

retriever_choices = [
    {"name": "baseline_text2cypher", "allow_linking": True},
    {"name": "baseline_text2cypher", "allow_linking": False},
    {"name": "property_descriptive", "allow_linking": True},
    {"name": "property_descriptive", "allow_linking": False},
    {"name": "property_text", "allow_linking": True},
    {"name": "property_text", "allow_linking": False},
    {"name": "concept_descriptive", "allow_linking": False},
    {"name": "concept_text", "allow_linking": False},
    {"name": "row_descriptive", "allow_linking": False},
    {"name": "row_text", "allow_linking": False},
    {"name": "baseline_vectorsearch", "allow_linking": False},
]

def retriever_factory(name: str, allow_linking: bool = False):
    match name:
        case "baseline_text2cypher":
            return PropertyTextScopeRetriever(
                prompt_path="scopes/property_text/t2c_prompt.txt",
                allow_linking=allow_linking,
                allow_extended=False,
                allow_descriptive_only=False,
            )
        case "property_descriptive":
            return PropertyTextScopeRetriever(
                prompt_path="scopes/property_text/exc_descriptive_prompt.txt",
                allow_linking=allow_linking,
                allow_extended=True,
                allow_descriptive_only=True,
            )
        case "property_text":
            return PropertyTextScopeRetriever(
                prompt_path="scopes/property_text/exc_text_prompt.txt",
                allow_linking=allow_linking,
                allow_extended=True,
                allow_descriptive_only=False,
            )
        case "concept_descriptive":
            return ConceptTextScopeRetriever(
                prompt_path="scopes/concept_text/exc_descriptive_prompt.txt",
                allow_descriptive_only=True
            )
        case "concept_text":
            return ConceptTextScopeRetriever(
                prompt_path="scopes/concept_text/exc_text_prompt.txt",
                allow_descriptive_only=False
            )
        case "row_descriptive":
            return RowTextScopeRetriever(
                prompt_path="scopes/row_text/exc_descriptive_prompt.txt",
                allow_descriptive_only=True
            )
        case "row_text":
            return RowTextScopeRetriever(
                prompt_path="scopes/row_text/exc_text_prompt.txt",
                allow_descriptive_only=False
            )
        case "baseline_vectorsearch":
            return RowAllScopeRetriever()
        case _:
            print("Error: Not a valid retriever name.")
            return
