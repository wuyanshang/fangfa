"""
入口文件 - 运行资产相似度工作流

使用方式:
    python -m asset_similarity.main

前置依赖:
    pip install langgraph httpx openpyxl
"""
import sys
import time

from .config import INPUT_EXCEL, CHECKPOINT_RESET_ON_START
from .checkpoint_io import prepare_checkpoint_outputs
from .excel_io import read_input_excel
from .graph import build_graph


def main():
    print("=" * 60)
    print("  资产相似度分析工作流 (LangGraph)")
    print("=" * 60)

    start = time.time()

    # 1. 读取输入
    asset_rows = read_input_excel(INPUT_EXCEL)
    if not asset_rows:
        print("未读取到数据，请检查输入文件。")
        sys.exit(1)

    # 2. 构建并运行工作流
    prepare_checkpoint_outputs(force_reset=CHECKPOINT_RESET_ON_START)
    graph = build_graph()
    initial_state = {
        "asset_rows": asset_rows,
        "normalized_groups": {},
        "unique_names": [],
        "phase0_pairs": [],
        "candidate_names": [],
        "all_pairs": [],
        "judge_results": [],
        "groups": [],
        "edges": [],
    }

    print("\n开始执行工作流...\n")
    final_state = graph.invoke(initial_state)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  完成! 总耗时: {elapsed:.1f}s")
    print(f"  最终分组数: {len(final_state.get('groups', []))}")
    print(f"  输出文件在 output/ 目录下")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
