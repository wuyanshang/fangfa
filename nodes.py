"""
LangGraph workflow nodes implemented with ThreadPoolExecutor."""
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import (
    KNN_TOP1_THRESHOLD, SEMANTIC_TOP_K, BM25_TOP_K, EXACT_MATCH_TOP_K,
    LLM_CONCURRENCY, ES_CONCURRENCY, GROUP_SIZE_REVIEW_THRESHOLD,
    PHASE3_WRITE_BATCH_SIZE, PHASE4_WRITE_BATCH_SIZE, ENABLE_CHECKPOINT_RESUME,
)
from .models import AssetRow, CandidatePair, JudgeResult, AssetGroup, WorkflowState
from .checkpoint_io import (
    write_phase2_pairs_batched,
    append_phase3_results,
    append_phase4_results,
    has_phase2_pairs,
    has_phase3_results,
    has_phase4_results,
    is_stage_completed,
    mark_stage_completed,
    reset_stage,
    load_phase2_pairs,
    load_phase3_results,
    load_phase4_results,
)
from .utils import (
    normalize,
    es_knn_search, es_bm25_search, es_exact_fuzzy_search,
    get_embedding_from_es, llm_judge,
)

logger = logging.getLogger(__name__)


def _connected_components_from_edges(edges: set[tuple[str, str]]) -> list[set[str]]:
    """Compute connected components for an undirected graph."""
    parent: dict[str, str] = {}
    rank: dict[str, int] = {}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while x != root:
            p = parent[x]
            parent[x] = root
            x = p
        return root

    def union(a: str, b: str):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for a, b in edges:
        if a not in parent:
            parent[a] = a
            rank[a] = 0
        if b not in parent:
            parent[b] = b
            rank[b] = 0
        union(a, b)

    groups: dict[str, set[str]] = defaultdict(set)
    for node in parent:
        groups[find(node)].add(node)

    return list(groups.values())

def _find_bridges_and_articulation_points(
    edges: set[tuple[str, str]]
) -> tuple[set[tuple[str, str]], set[str]]:
    """Find bridge edges and articulation points via Tarjan algorithm."""
    adjacency: dict[str, set[str]] = defaultdict(set)
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    discovery_time: dict[str, int] = {}
    low_link: dict[str, int] = {}
    parent: dict[str, str | None] = {}
    bridges: set[tuple[str, str]] = set()
    articulation_points: set[str] = set()
    timer = 0

    def dfs(node: str):
        nonlocal timer
        timer += 1
        discovery_time[node] = timer
        low_link[node] = timer
        child_count = 0

        for neighbor in adjacency[node]:
            if neighbor not in discovery_time:
                parent[neighbor] = node
                child_count += 1
                dfs(neighbor)
                low_link[node] = min(low_link[node], low_link[neighbor])

                if low_link[neighbor] > discovery_time[node]:
                    bridges.add(tuple(sorted([node, neighbor])))

                if parent[node] is None and child_count > 1:
                    articulation_points.add(node)
                if parent[node] is not None and low_link[neighbor] >= discovery_time[node]:
                    articulation_points.add(node)
            elif neighbor != parent[node]:
                low_link[node] = min(low_link[node], discovery_time[neighbor])

    for node in adjacency:
        if node not in discovery_time:
            parent[node] = None
            dfs(node)

    return bridges, articulation_points


# 
# Stage 0: normalize and exact-name grouping
def phase0_normalize(state: WorkflowState) -> dict:
    """Normalize names and generate exact-match pairs from normalized buckets."""
    rows: list[AssetRow] = state["asset_rows"]

    groups = defaultdict(list)
    for row in rows:
        row.normalized_asset = normalize(row.asset)
        groups[row.normalized_asset].append(row)

    unique_names = []
    for norm_name, group_rows in groups.items():
        unique_names.append(group_rows[0].asset)

    phase0_pairs = []
    for norm_name, group_rows in groups.items():
        distinct_names = list(set(r.asset for r in group_rows))
        if len(distinct_names) > 1:
            for i in range(len(distinct_names)):
                for j in range(i + 1, len(distinct_names)):
                    phase0_pairs.append(CandidatePair(
                        name_a=distinct_names[i],
                        name_b=distinct_names[j],
                        source="phase0_exact",
                        score=1.0,
                    ))

    print(f"[phase0] rows: {len(rows)}, unique names: {len(unique_names)}, "
          f"exact-match pairs: {len(phase0_pairs)}")

    return {
        "normalized_groups": dict(groups),
        "unique_names": unique_names,
        "phase0_pairs": phase0_pairs,
    }


# 
# Stage 1: KNN filter
def phase1_knn_filter(state: WorkflowState) -> dict:
    """Run KNN top-1 filter and keep names that pass the threshold."""
    unique_names = state["unique_names"]
    candidate_names = []
    total = len(unique_names)

    def check_one(name: str) -> str | None:
        embedding = get_embedding_from_es(name)
        if embedding is None:
            return None
        results = es_knn_search(embedding, k=1, exclude_names=[name])
        if results and results[0]["score"] >= KNN_TOP1_THRESHOLD:
            return name
        return None

    done_count = 0
    with ThreadPoolExecutor(max_workers=ES_CONCURRENCY) as pool:
        futures = {pool.submit(check_one, n): n for n in unique_names}
        for future in as_completed(futures):
            done_count += 1
            if done_count % 500 == 0 or done_count == total:
                print(f"[1] : {done_count}/{total}")
            result = future.result()  # retry_forever guarantees no exception here
            if result is not None:
                candidate_names.append(result)

    print(f"[phase1] passed candidates: {len(candidate_names)} / {total}")
    return {"candidate_names": candidate_names}


# 
# Stage 2: recall and deduplicate
def phase2_recall(state: WorkflowState) -> dict:
    """Recall candidate pairs from semantic/BM25/fuzzy routes and deduplicate."""
    if ENABLE_CHECKPOINT_RESUME and is_stage_completed("phase2") and has_phase2_pairs():
        cached_pairs = load_phase2_pairs()
        print(f"[2] :  {len(cached_pairs)}")
        return {"all_pairs": cached_pairs}

    reset_stage("phase2")
    candidate_names = state["candidate_names"]
    total = len(candidate_names)
    pair_set: set[tuple] = set()
    pair_set_lock = __import__("threading").Lock()

    def recall_one(name: str) -> list[str]:
        """Recall candidate names for a single asset name."""
        exclude = [name]
        recalled = set()

 # R1: 
        embedding = get_embedding_from_es(name)
        if embedding:
            r1 = es_knn_search(embedding, k=SEMANTIC_TOP_K, exclude_names=exclude)
            for item in r1:
                recalled.add(item["asset"])

 # R2: BM25 
        r2 = es_bm25_search(name, top_k=BM25_TOP_K, exclude_names=exclude)
        for item in r2:
            recalled.add(item["asset"])

 # R3: /
        r3 = es_exact_fuzzy_search(name, top_k=EXACT_MATCH_TOP_K, exclude_names=exclude)
        for item in r3:
            recalled.add(item["asset"])

        return list(recalled)

    done_count = 0
    with ThreadPoolExecutor(max_workers=ES_CONCURRENCY) as pool:
        futures = {pool.submit(recall_one, n): n for n in candidate_names}
        for future in as_completed(futures):
            done_count += 1
            if done_count % 200 == 0 or done_count == total:
                print(f"[2] : {done_count}/{total}")
            name_a = futures[future]
            recalled_names = future.result()
            with pair_set_lock:
                for name_b in recalled_names:
                    pair_key = tuple(sorted([name_a, name_b]))
                    pair_set.add(pair_key)

    all_pairs = [
        CandidatePair(name_a=a, name_b=b, source="recall")
        for a, b in pair_set
    ]
    write_phase2_pairs_batched(all_pairs)
    mark_stage_completed("phase2")
    print(f"[2] : {len(all_pairs)}")
    return {"all_pairs": all_pairs}


# 
# Stage 3: batched LLM judgment
def phase3_llm_judge(state: WorkflowState) -> dict:
    """Run batched LLM YES/NO judgments for candidate pairs."""
    if ENABLE_CHECKPOINT_RESUME and is_stage_completed("phase3") and has_phase3_results():
        cached_results = load_phase3_results()
        print(f"[phase3] resume from checkpoint: {len(cached_results)}")
        return {"judge_results": cached_results}

    reset_stage("phase3")
    all_pairs: list[CandidatePair] = state["all_pairs"]

    phase0_keys = set(p.key for p in state["phase0_pairs"])
    existing_results = load_phase3_results() if (ENABLE_CHECKPOINT_RESUME and has_phase3_results()) else []
    existing_map = {tuple(sorted([r.name_a, r.name_b])): r for r in existing_results}

    pairs_to_judge = []
    for p in all_pairs:
        if p.key in phase0_keys:
            continue
        if p.key in existing_map:
            continue
        pairs_to_judge.append(p)
    total = len(pairs_to_judge)
    print(
        f"[phase3] pairs to judge: {total} "
        f"(exclude phase0 exact: {len(phase0_keys)}, resumed: {len(existing_map)})"
    )

    judge_results = list(existing_results)
    results_lock = __import__("threading").Lock()
    batch_buffer = []

    def judge_one(pair: CandidatePair) -> JudgeResult:
        answer = llm_judge(pair.name_a, pair.name_b)
        return JudgeResult(
            name_a=pair.name_a, name_b=pair.name_b,
            result=answer, source="recall",
        )

    done_count = 0
    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as pool:
        futures = {pool.submit(judge_one, p): p for p in pairs_to_judge}
        for future in as_completed(futures):
            done_count += 1
            if done_count % 500 == 0 or done_count == total:
                print(f"[3] LLM: {done_count}/{total}")
            result = future.result()
            with results_lock:
                judge_results.append(result)
                batch_buffer.append(result)
                if len(batch_buffer) >= PHASE3_WRITE_BATCH_SIZE:
                    append_phase3_results(batch_buffer)
                    batch_buffer = []

    if batch_buffer:
        append_phase3_results(batch_buffer)
        batch_buffer = []

    # Append phase0 exact-match pairs as YES
    for p in state["phase0_pairs"]:
        key = p.key
        if key in existing_map:
            continue
        r = JudgeResult(name_a=p.name_a, name_b=p.name_b, result="YES", source="phase0_exact")
        judge_results.append(r)
        phase0_results_to_append.append(r)
    append_phase3_results(phase0_results_to_append)

    mark_stage_completed("phase3")

    yes_count = sum(1 for r in judge_results if r.result == "YES")
    no_count = sum(1 for r in judge_results if r.result == "NO")
    print(f"[phase3] finished. YES: {yes_count}, NO: {no_count}")

    return {"judge_results": judge_results}


# 
# Stage 4: graph build and bridge recheck
def phase4_build_graph(state: WorkflowState) -> dict:
    """Recheck bridge edges to split mistakenly merged large components."""
    judge_results: list[JudgeResult] = state["judge_results"]

    judged_pairs: dict[tuple[str, str], str] = {}
    for r in judge_results:
        judged_pairs[tuple(sorted([r.name_a, r.name_b]))] = r.result

    yes_edges = {pair_key for pair_key, result in judged_pairs.items() if result == "YES"}
    components = _connected_components_from_edges(yes_edges)
    print(f"[phase4] connected components: {len(components)}")

    bridge_edges, articulation_points = _find_bridges_and_articulation_points(yes_edges)
    bridge_pairs = [
        CandidatePair(name_a=a, name_b=b, source="bridge_recheck")
        for a, b in bridge_edges
    ]
    print(
        f"[phase4] bridge recheck pairs: {len(bridge_pairs)}, "
        f"articulation points: {len(articulation_points)}"
    )

    reset_stage("phase4")
    existing_phase4 = load_phase4_results() if (ENABLE_CHECKPOINT_RESUME and has_phase4_results()) else []
    existing_bridge_results = [r for r in existing_phase4 if r.source == "bridge_recheck"]
    existing_bridge_map = {
        tuple(sorted([r.name_a, r.name_b])): r
        for r in existing_bridge_results
    }
    bridge_pairs_to_run = [p for p in bridge_pairs if p.key not in existing_bridge_map]
    if existing_bridge_results:
        print(f"[phase4] resumed bridge recheck results: {len(existing_bridge_results)}")

    bridge_results = list(existing_bridge_results)
    if bridge_pairs_to_run:
        def judge_one(pair: CandidatePair) -> JudgeResult:
            answer = llm_judge(pair.name_a, pair.name_b)
            return JudgeResult(
                name_a=pair.name_a,
                name_b=pair.name_b,
                result=answer,
                source="bridge_recheck",
            )

        with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as pool:
            futures = [pool.submit(judge_one, p) for p in bridge_pairs_to_run]
            batch_buffer = []
            for f in as_completed(futures):
                result = f.result()
                bridge_results.append(result)
                batch_buffer.append(result)
                if len(batch_buffer) >= PHASE4_WRITE_BATCH_SIZE:
                    append_phase4_results(batch_buffer)
                    batch_buffer = []
            if batch_buffer:
                append_phase4_results(batch_buffer)
    mark_stage_completed("phase4")

    final_pair_results = dict(judged_pairs)
    for r in bridge_results:
        final_pair_results[tuple(sorted([r.name_a, r.name_b]))] = r.result

    all_edges = [
        JudgeResult(name_a=a, name_b=b, result=result, source="final")
        for (a, b), result in final_pair_results.items()
    ]

    final_yes_edges = {
        pair_key
        for pair_key, result in final_pair_results.items()
        if result == "YES"
    }
    final_components = _connected_components_from_edges(final_yes_edges)
    node_to_component = {}
    for idx, comp in enumerate(final_components):
        for node in comp:
            node_to_component[node] = idx

    component_edge_count = [0] * len(final_components)
    for a, b in final_yes_edges:
        comp_idx = node_to_component.get(a)
        if comp_idx is not None and comp_idx == node_to_component.get(b):
            component_edge_count[comp_idx] += 1

    groups = []
    for idx, comp in enumerate(final_components):
        members = sorted(comp)
        n = len(members)
        expected_edges = n * (n - 1) // 2
        actual_edges = component_edge_count[idx]
        is_fully_connected = (actual_edges == expected_edges)

        groups.append(AssetGroup(
            group_id=idx + 1,
            asset_names=members,
            is_fully_connected=is_fully_connected,
            needs_review=(n > GROUP_SIZE_REVIEW_THRESHOLD or not is_fully_connected),
        ))

    print(
        f"[phase4] final groups: {len(groups)}, "
        f"needs review: {sum(1 for g in groups if g.needs_review)}"
    )

    return {"groups": groups, "edges": all_edges}

