

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3"
LLM_MODEL = "llama3.2"

RAW_FOLDER = "data/raw"
SQLITE_PATH = "data/rag.db"
CHROMA_PATH = "data/chroma_db"


CHUNK_SIZES = [200, 500]
CHUNK_OVERLAP = 40

EMBED_BATCH_SIZE = 16


RAG_CHUNK_SIZE = 200
REFUSE_THRESHOLD = 0.5  # from calibrate_threshold.py


CLASSIFIER_CANDIDATES_PATH = "data/classifier_candidates.csv"
CLASSIFIER_DATA_PATH = "data/classifier_data.csv"
CLASSIFIER_PATH = "models/scope_classifier.joblib"
CLASSIFIER_RESULTS_PATH = "results/classifier_results.json"
CLASSIFIER_THRESHOLD = 0.5
RANDOM_SEED = 42  # fixed so the split is the same every run


CACHE_THRESHOLD = 0.95
CACHE_GUARD_MODE = "topic"
CACHE_PAIRS_CANDIDATES_PATH = "data/cache_pairs_candidates.csv"
CACHE_PAIRS_PATH = "data/cache_pairs.csv"
CACHE_RESULTS_PATH = "results/cache_threshold.csv"


HARD_TEST_PATH = "data/hard_test.csv"
CLASSIFIER_CHECKS_PATH = "results/classifier_checks.json"


JUDGE_MODEL = None
ANSWER_EVAL_PATH = "results/answer_eval.csv"
ANSWER_METRICS_PATH = "results/answer_metrics.json"
HUMAN_SHEET_PATH = "results/human_label_sheet.csv"
WORST_PATH = "results/worst5.md"
JUDGE_AGREEMENT_PATH = "results/judge_agreement.json"
JUDGE_DISAGREEMENTS_PATH = "results/judge_disagreements.csv"


TOP_K = 5
EVAL_CANDIDATES_PATH = "data/eval_candidates.csv"
EVAL_PATH = "data/eval_dataset.csv"
RESULTS_PATH = "results/experiments.csv"
