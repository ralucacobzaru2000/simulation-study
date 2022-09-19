class JsonConfigConstants:
    ADJ_METHOD: str = "adjustment_method"
    LEARNER: str = "learner"
    SUPER_LEARNER: str = "super_learner"
    D_ROBUST_MODEL: str = "double_robust_models"
    EFF_MEASURE: str = "effect_measure"
    ADVANCED: str = "advanced"

    COMMENT: str = "comment"
    LEARNER_PARAMS: str = "learner_params"
    IPW: str = "IPW"
    STANDARDIZATION: str = "standardization"

    BINARY: str = "binary"
    CONTINUOUS: str = "continuous"

    D_ROBUST: str = "double_robust"
    TREATMENT_MODEL: str = "treatment_model"
    OUTCOME_MODEL: str = "outcome_model"

    PROP_MODEL: str = "propensity_model"
    EVENT_MODEL: str = "hazard_model"
    CENSOR_MODEL: str = "censor_model"

    SAMPLE_SPLIT: str = "sample_split"

    OUTCOME_TYPE: str = "outcome_type"
    RELATION: str = "relation"

    DATA_PROCESS: str = "data_process"
    CHECK_DECOMPRESS: str = "decompress"
    MULTI_TREAT: str = "multi_treat"
    MULTI_TREAT_ITERS: str = "multi_treat_iters"
    ONLY_SPARSE_COVAR: str = "only_sparse_covar"
    BOOTSTRAP_SEED: str = "bootstrap_seed"

    STD_ERR: str = "std_error"
    BOOTSTRAP: str = "bootstrap"
    ANALYTIC: str = "analytic"
    WEIGHT_STABILIZE: str = "weight_stabilize"
    TRUNCATE: str = "truncate"
    STABILIZE: str = "stabilize"
    STABILITY_CHECK: str = "stability_checks"
