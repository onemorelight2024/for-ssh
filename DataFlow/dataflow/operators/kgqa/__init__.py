from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # eval
    from .eval.kgqa_validity_evaluate import KGQAValidityEvaluate
    from .eval.kgqa_semantic_diversity_evaluate import KGQASemanticDiversityEvaluate
    from .eval.kgqa_coverage_evaluate import KGQACoverageEvaluate

    # filter
    from .filter.kg_sparql_reverse_generate import KGSparqlReverseGenerate
    from .filter.kg_sparql_keyword_filter import KGSparqlKeywordFilter
    from .filter.kg_sparql_validate import KGSparqlValidate

    # generate
    from .generate.clinic_kg_abbreviation_expansion import ClinicKGAbbreviationExpansion
    from .generate.clinic_kg_negation_resolution import ClinicKGNegationResolution
    from .generate.kg_sparql_path_sampler import KGSparqlPathSampler
    from .generate.kg_sparql_qa_generate import KGSparqlQAGenerate
    from .generate.kg_triple_format_converter import KGTripleFormatConverter
    from .generate.kg_sparql_select import KGSparqlSelect

    # refine
    from .refine.kg_question_rewriter import KGQuestionRewriter

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    cur_path = "dataflow/operators/kgqa/"
    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, cur_path, _import_structure)
