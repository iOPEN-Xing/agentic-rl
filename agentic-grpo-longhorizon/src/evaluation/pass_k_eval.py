"""Repeated-sampling evaluation with mathematically distinct pass@k/pass^k."""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import json
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Optional
from tqdm import tqdm

if TYPE_CHECKING:
    from src.envs.tau_bench_wrapper import TauBenchWrapper, TrajectoryResult


def _validate_pass_inputs(n: int, c: int, k: int) -> None:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= c <= n:
        raise ValueError(f"c must be in [0, n], got c={c}, n={n}")
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, n], got k={k}, n={n}")


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that at least one of k samples succeeds, without replacement."""
    _validate_pass_inputs(n, c, k)
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def estimate_pass_power_k(n: int, c: int, k: int) -> float:
    """Probability that all k samples succeed, without replacement."""
    _validate_pass_inputs(n, c, k)
    if c < k:
        return 0.0
    return math.comb(c, k) / math.comb(n, k)


def _get_tokenizer(policy_factory):
    """尝试从 policy 获取 model_name 并加载 tokenizer；失败则返回 None。"""
    try:
        policy = policy_factory()
        model_name = getattr(policy, "model_name", None)
        if model_name:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            print(f"[pass_k_eval] Loaded tokenizer for {model_name}")
            return tok
    except Exception as e:
        print(f"[pass_k_eval] Failed to load tokenizer: {e}, falling back to char length")
    return None


def _make_token_counter(tokenizer):
    """返回一个 callable: text -> token count"""
    if tokenizer is not None:
        def _count(text: str) -> int:
            try:
                return len(tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                return len(text)
        return _count
    return len


@dataclass
class EvalReport:
    env_name: str
    num_tasks: int
    num_samples_per_task: int
    any_success_rate: float
    pass_at_1: float
    pass_at_4: Optional[float]
    pass_at_8: Optional[float]
    pass_power_1: float
    pass_power_4: Optional[float]
    pass_power_8: Optional[float]
    # Backward-compatible aliases. pass_hat_* now consistently means pass^k.
    pass_hat_1: float
    pass_hat_4: Optional[float]
    pass_hat_8: Optional[float]
    legacy_pass_at_1_any_success: float
    legacy_pass_hat_4_as_pass_at_4: Optional[float]
    legacy_pass_hat_8_as_pass_at_8: Optional[float]
    avg_turns: float
    avg_tool_calls: float
    error_rate: float          # trajectory 异常中止的比例
    per_task_results: list[dict]


def run_eval(
    wrapper: TauBenchWrapper,
    policy_factory,                    # callable -> policy instance (thread-safe)
    num_tasks: Optional[int] = None,
    num_samples_per_task: int = 4,
    max_turns: int = 30,
    num_workers: int = 4,
    output_dir: str = "experiments/baseline_airline_7B_user",
) -> EvalReport:
    """
    policy_factory: 每个 worker 线程自己 new 一个 policy,避免并发问题
    """
    if num_tasks is None:
        num_tasks = wrapper.get_num_tasks()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 每个 (task_idx, sample_idx) 是一个独立 job
    jobs = [(t, s) for t in range(num_tasks) for s in range(num_samples_per_task)]
    results: dict[int, list[TrajectoryResult]] = {t: [] for t in range(num_tasks)}
    
    def _run_one(task_idx: int, sample_idx: int) -> tuple[int, TrajectoryResult]:
        policy = policy_factory()
        # 给同一个 task 不同 sample 设不同 temperature seed
        # (vLLM server 端已经有采样随机性,这里主要是逻辑标记)
        traj = wrapper.run_single_task(task_idx, policy, max_turns=max_turns)
        return task_idx, traj
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_run_one, t, s) for t, s in jobs]
        for fut in tqdm(as_completed(futures), total=len(jobs), desc="Eval"):
            task_idx, traj = fut.result()
            results[task_idx].append(traj)
    
    import numpy as np
    
    # 尝试加载 tokenizer（用于精确统计 assistant content tokens）
    tokenizer = _get_tokenizer(policy_factory)
    count_tokens = _make_token_counter(tokenizer)

    per_task = []
    pass_at_1_list, pass_at_4_list, pass_at_8_list = [], [], []
    pass_power_1_list, pass_power_4_list, pass_power_8_list = [], [], []
    any_success_list = []
    all_turns, all_tool_calls, all_errors = [], [], []

    for t in range(num_tasks):
        trajs = results[t]
        n = len(trajs)
        c = sum(1 for tr in trajs if tr.success)

        pass_at_1 = estimate_pass_at_k(n, c, 1)
        pass_power_1 = estimate_pass_power_k(n, c, 1)
        pass_at_4 = estimate_pass_at_k(n, c, 4) if n >= 4 else None
        pass_power_4 = estimate_pass_power_k(n, c, 4) if n >= 4 else None
        pass_at_8 = estimate_pass_at_k(n, c, 8) if n >= 8 else None
        pass_power_8 = estimate_pass_power_k(n, c, 8) if n >= 8 else None

        pass_at_1_list.append(pass_at_1)
        pass_power_1_list.append(pass_power_1)
        any_success_list.append(1.0 if c > 0 else 0.0)
        if pass_at_4 is not None:
            pass_at_4_list.append(pass_at_4)
            pass_power_4_list.append(pass_power_4)
        if pass_at_8 is not None:
            pass_at_8_list.append(pass_at_8)
            pass_power_8_list.append(pass_power_8)

        traj_dicts = []
        for tr in trajs:
            all_turns.append(tr.num_turns)
            all_tool_calls.append(tr.num_tool_calls)
            all_errors.append(1.0 if tr.error else 0.0)

            # 计算每轮 assistant turn 的 content token 数
            per_turn_assistant_content_tokens = []
            for msg in tr.raw_messages:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content", "") or ""
                    per_turn_assistant_content_tokens.append(count_tokens(content))

            traj_dict = tr.to_dict()
            traj_dict["per_turn_assistant_content_tokens"] = per_turn_assistant_content_tokens
            traj_dicts.append(traj_dict)

        per_task.append({
            "task_id": t,
            "success_count": c,
            "total_samples": n,
            "any_success": bool(c > 0),
            "pass@1": pass_at_1,
            "pass@4": pass_at_4,
            "pass@8": pass_at_8,
            "pass^1": pass_power_1,
            "pass^4": pass_power_4,
            "pass^8": pass_power_8,
            "avg_turns": np.mean([tr.num_turns for tr in trajs]),
            "trajectories": traj_dicts,
        })

    mean_pass_at_4 = float(np.mean(pass_at_4_list)) if pass_at_4_list else None
    mean_pass_at_8 = float(np.mean(pass_at_8_list)) if pass_at_8_list else None
    mean_pass_power_4 = float(np.mean(pass_power_4_list)) if pass_power_4_list else None
    mean_pass_power_8 = float(np.mean(pass_power_8_list)) if pass_power_8_list else None
    report = EvalReport(
        env_name=wrapper.env_name,
        num_tasks=num_tasks,
        num_samples_per_task=num_samples_per_task,
        any_success_rate=float(np.mean(any_success_list)),
        pass_at_1=float(np.mean(pass_at_1_list)),
        pass_at_4=mean_pass_at_4,
        pass_at_8=mean_pass_at_8,
        pass_power_1=float(np.mean(pass_power_1_list)),
        pass_power_4=mean_pass_power_4,
        pass_power_8=mean_pass_power_8,
        pass_hat_1=float(np.mean(pass_power_1_list)),
        pass_hat_4=mean_pass_power_4,
        pass_hat_8=mean_pass_power_8,
        legacy_pass_at_1_any_success=float(np.mean(any_success_list)),
        legacy_pass_hat_4_as_pass_at_4=mean_pass_at_4,
        legacy_pass_hat_8_as_pass_at_8=mean_pass_at_8,
        avg_turns=float(np.mean(all_turns)),
        avg_tool_calls=float(np.mean(all_tool_calls)),
        error_rate=float(np.mean(all_errors)),
        per_task_results=per_task,
    )
    
    # 保存
    with open(output_dir / "eval_report.json", "w") as f:
        #json.dump(asdict(report), f, indent=2, ensure_ascii=False)
        json.dump(asdict(report), f, indent=2, ensure_ascii=False, default=str)
    
    # 打印摘要
    print(f"\n=== Eval Report: {wrapper.env_name} ===")
    print(f"Tasks: {num_tasks} × Samples: {num_samples_per_task}")
    print(f"Any-success rate: {report.any_success_rate:.3f}")
    print(f"pass@1:          {report.pass_at_1:.3f}")
    print(f"pass@4:          {report.pass_at_4 if report.pass_at_4 is not None else 'n/a'}")
    print(f"pass@8:          {report.pass_at_8 if report.pass_at_8 is not None else 'n/a'}")
    print(f"pass^1:          {report.pass_power_1:.3f}")
    print(f"pass^4:          {report.pass_power_4 if report.pass_power_4 is not None else 'n/a'}")
    print(f"pass^8:          {report.pass_power_8 if report.pass_power_8 is not None else 'n/a'}")
    print(f"Avg turns:       {report.avg_turns:.2f}")
    print(f"Avg tool calls:  {report.avg_tool_calls:.2f}")
    print(f"Error rate:      {report.error_rate:.3f}")
    
    return report
