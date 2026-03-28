"""
配置文件 - 修改这里的连接信息
"""

# ── Elasticsearch 配置 ──
ES_HOST = "http://localhost:9200"
ES_INDEX = "your_index_name"
ES_USER = ""  # 如果不需要认证，留空
ES_PASSWORD = ""

# ── Embedding 模型配置 ──
EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_API_KEY = "YOUR_API_KEY_HERE"
EMBEDDING_MODEL = "text-embedding-v3"
EMBEDDING_DIMS = 1024

# ── LLM 配置 ──
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_API_KEY = "YOUR_API_KEY_HERE"
LLM_MODEL = "qwen-plus"

# ── 工作流参数 ──
KNN_TOP1_THRESHOLD = 0.78       # 阶段1 knn快筛阈值
SEMANTIC_TOP_K = 5              # 语义召回数量
BM25_TOP_K = 5                  # BM25召回数量
EXACT_MATCH_TOP_K = 20          # 精确/模糊召回数量
LLM_CONCURRENCY = 50            # LLM 并发数
ES_CONCURRENCY = 50             # ES 查询并发数
GROUP_SIZE_REVIEW_THRESHOLD = 10  # 超过此数量标记为待人工审核

# ── 重试配置 ──
RETRY_BASE_DELAY = 1.0            # 首次重试等待秒数
RETRY_MAX_DELAY = 60.0            # 最大等待秒数（指数退避上限）

# ── 输入输出 ──
INPUT_EXCEL = "input.xlsx"      # 输入Excel文件路径
OUTPUT_DIR = "output"           # 输出目录

# ── 中间结果分批写入（CSV） ──
CHECKPOINT_DIR = "output/checkpoints"
PHASE2_WRITE_BATCH_SIZE = 5000
PHASE3_WRITE_BATCH_SIZE = 1000
PHASE4_WRITE_BATCH_SIZE = 1000
ENABLE_CHECKPOINT_RESUME = True
CHECKPOINT_RESET_ON_START = False
