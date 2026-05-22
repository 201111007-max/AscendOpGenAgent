#!/usr/bin/env python3
"""Single-subprocess orchestrator: verify + profile_gen (+ profile_base).

Verification logic lives in the kernel-verifier skill — this script does
NOT reimplement `run_single_case`:

    skills/triton/kernel-verifier/scripts/verify.py     run_single_case

Timing uses an inline NPU-event helper (`_measure_npu_event_ms` below),
NOT the skill's `measure_single` — the skill's profiler-based path pays
~3 sec/case in CANN parse overhead, which on 51-case multi-shape ops
(pad / interpolate / repeat_interleave) dominates wall-clock 10× over
actual measurement work. The skill code stays untouched; only the
in-engine helper changed.

Why a single-subprocess orchestrator instead of just spawning the two
skill CLIs separately?

  - Triton JIT cache populated during verify is reused by profile_gen
    in the SAME process (warm start). Splitting into two subprocesses
    means cold compilation each time — minutes on triton-ascend kernels
    with constexpr specializations.

  - autoresearch's eval_client._assemble_eval_result expects ONE JSON
    sidecar with {verify, profile_base, profile_gen} keys; this script
    glues the three skill calls into that schema. The skill CLIs each
    produce their own JSON shape and don't know about autoresearch's
    contract.

  - File naming: autoresearch uses `reference.py` + `kernel.py` per
    task_dir; the skill CLIs use `<op>_torch.py` + `<op>_<dsl>_impl.py`.
    Loading via importlib.util sidesteps that mismatch — we hand
    already-loaded module objects + cases to the skill functions.

Standalone reproducer:
    python .autoresearch/scripts/engine/eval_kernel.py \\
        --task-dir <task_dir> --op-name <op> \\
        --kernel-file kernel --ref-file reference \\
        --device-id 0 \\
        --warmup 10 --repeats 100 --phases verify,profile_gen,profile_base

Precision: per-dtype rtol+atol; fp16 / bf16 / fp32 thresholds in
skill verify.compare(). autoresearch's metric is on-device elapsed time
from `torch.npu.Event(enable_timing=True)` — same device-time semantics
the profiler reports, just measured without the profiler context.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback


# ---------------------------------------------------------------------------
# Module + skill loaders
# ---------------------------------------------------------------------------

def _load_module(name: str, path: str):
    """Load a Python file as a module under `name`."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_skill_modules() -> tuple:
    """Load skill verify.py + benchmark.py under unique names so the
    generic basenames `verify` / `benchmark` can't shadow (or be shadowed
    by) other modules with the same names elsewhere on sys.path
    (e.g. .autoresearch/scripts/batch/verify.py)."""
    here = os.path.dirname(os.path.abspath(__file__))
    # engine/ -> scripts/ -> .autoresearch/ -> repo root
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    skill_scripts = os.path.join(
        repo_root, "skills", "triton", "kernel-verifier", "scripts",
    )
    if not os.path.isdir(skill_scripts):
        raise RuntimeError(
            f"kernel-verifier skill not found at {skill_scripts!r} — "
            f"eval_kernel orchestrator depends on it. Layout assumption: "
            f"skills/ is a sibling of .autoresearch/."
        )
    verify_mod = _load_module(
        "_skill_verify", os.path.join(skill_scripts, "verify.py"))
    bench_mod = _load_module(
        "_skill_benchmark", os.path.join(skill_scripts, "benchmark.py"))
    return verify_mod, bench_mod


# ---------------------------------------------------------------------------
# Per-target build helpers
# ---------------------------------------------------------------------------

def _build_target(target_cls, init_inputs, device):
    m = target_cls(*init_inputs)
    if hasattr(m, "to"):
        m = m.to(device)
    if hasattr(m, "eval"):
        m = m.eval()
    return m


def _seed_npu():
    import torch
    torch.manual_seed(0)
    try:
        torch.npu.manual_seed(0)
    except Exception:
        pass


def _measure_npu_event_ms(model, inputs, warmup: int, repeats: int) -> float:
    """Mean per-call device time in milliseconds via NPU events.

    Why inline here (and not in the skill): the skill's profiler-based
    `measure_single` pays ~3 sec/case in CANN parse overhead — on
    multi-shape ops (51 cases for pad / interpolate) that dominates
    wall-clock 10× over the actual kernel work. NPU events give the same
    device-time metric with none of that overhead.

    Per-operator breakdown is not available from this path; the skill's
    `measure_single` is still the right call when an agent wants per-op
    timing standalone. autoresearch's eval_client only consumes the
    aggregate latency so this trade is free here.
    """
    import torch
    import torch_npu  # noqa: F401

    torch.npu.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*inputs)
    torch.npu.synchronize()

    ev_start = torch.npu.Event(enable_timing=True)
    ev_end = torch.npu.Event(enable_timing=True)
    ev_start.record()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(*inputs)
    ev_end.record()
    torch.npu.synchronize()

    elapsed_ms = ev_start.elapsed_time(ev_end)
    return round(elapsed_ms / max(1, repeats), 4)


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def run_verify(ref_mod, kernel_mod, cases_cpu, init_inputs, device,
               skill_verify) -> dict:
    """Per-case correctness check with ref-side vs kernel-side blame.

    The verify loop is split into three independently-caught phases per
    case so failures can be attributed (error_source = "ref" | "kernel"):

      1. Build ref model + forward         → error_source="ref"
      2. Build kernel model + forward      → error_source="kernel"
      3. Compare ref vs kernel outputs     → error_source="kernel"
         (a comparison failure is a kernel numerical issue; the ref
         outputs from phase 1 are the ground truth.)

    Scaffold reads the resulting `error_source` on the verify dict and
    refuses to activate the task when it's "ref" — the user must fix
    the source --ref file. "kernel" failures take the normal seed-fail
    recovery path (PLAN -> EDIT rewrite).

    Schema mirrors the legacy shape `eval_client._assemble_eval_result`
    consumes: {correctness, error_source, num_cases, failed_indices,
    per_case, diagnostics}.
    """
    from utils.input_groups import describe_case as _describe_case
    from utils.correctness import compare_outputs

    Model = ref_mod.Model
    ModelNew = kernel_mod.ModelNew

    num_cases = len(cases_cpu)
    per_case: list = []
    failed_indices: list[int] = []
    diagnostics: list[str] = []
    overall_correct = True
    # Track which side broke. "ref" takes priority over "kernel" because
    # a ref failure invalidates the whole eval regardless of what the
    # kernel does. None when verify passes.
    overall_error_source: str | None = None

    def _tag_ref_fail() -> None:
        nonlocal overall_error_source
        # ref-side errors override kernel-side ones — see docstring.
        overall_error_source = "ref"

    def _tag_kernel_fail() -> None:
        nonlocal overall_error_source
        if overall_error_source != "ref":
            overall_error_source = "kernel"

    def _to_cpu_list(out):
        import torch
        if isinstance(out, torch.Tensor):
            return [out.detach().cpu()]
        if isinstance(out, (list, tuple)):
            return [o.detach().cpu() if hasattr(o, "detach") else o
                    for o in out]
        return [out]

    import torch  # noqa: F401  — used by _to_cpu_list above

    for idx, inputs in enumerate(cases_cpu):
        framework_model = None
        impl_model = None
        case_desc = _describe_case(inputs, None)
        case_entry = {
            "idx": idx,
            "case_desc": case_desc,
            "correctness": False,
            "error": None,
            "error_source": None,
        }

        # --- Phase 1: ref build + forward ---
        out_ref_cpu = None
        try:
            _seed_npu()
            framework_model = _build_target(Model, init_inputs, device)
            ref_inputs_dev = [x.to(device) if hasattr(x, "to") else x
                              for x in inputs]
            with torch.no_grad():
                out_ref = framework_model(*ref_inputs_dev)
            out_ref_cpu = _to_cpu_list(out_ref)
            del ref_inputs_dev, out_ref
        except Exception as e:
            overall_correct = False
            failed_indices.append(idx)
            _tag_ref_fail()
            err = f"{type(e).__name__}: {e}"
            case_entry["error"] = f"ref-side: {err}"
            case_entry["error_source"] = "ref"
            diagnostics.append(skill_verify.truncate_error(
                f"[case {idx}] ref-side: {err}\n{traceback.format_exc()}"
            ))
            del framework_model
            skill_verify.cleanup_npu_memory()
            per_case.append(case_entry)
            continue

        # Free ref model before building kernel; HBM doesn't fit both at
        # once for BatchNorm-scale tensors.
        del framework_model
        framework_model = None
        skill_verify.cleanup_npu_memory()

        # --- Phase 2: kernel build + forward ---
        out_new_cpu = None
        try:
            _seed_npu()
            impl_model = _build_target(ModelNew, init_inputs, device)
            new_inputs_dev = [x.to(device) if hasattr(x, "to") else x
                              for x in inputs]
            with torch.no_grad():
                out_new = impl_model(*new_inputs_dev)
            out_new_cpu = _to_cpu_list(out_new)
            del new_inputs_dev, out_new
        except Exception as e:
            overall_correct = False
            failed_indices.append(idx)
            _tag_kernel_fail()
            err = f"{type(e).__name__}: {e}"
            case_entry["error"] = f"kernel-side: {err}"
            case_entry["error_source"] = "kernel"
            diagnostics.append(skill_verify.truncate_error(
                f"[case {idx}] kernel-side: {err}\n{traceback.format_exc()}"
            ))
            del impl_model
            skill_verify.cleanup_npu_memory()
            per_case.append(case_entry)
            continue

        # --- Phase 3: compare ---
        # compare_outputs (utils/correctness.py) delegates to the skill's
        # _check_accuracy_allclose for the actual element-wise check so
        # per-dtype tolerances stay in lockstep with batch verify.
        try:
            cmp = compare_outputs(list(out_ref_cpu), list(out_new_cpu))
            if cmp["correctness"]:
                case_entry["correctness"] = True
            else:
                overall_correct = False
                failed_indices.append(idx)
                _tag_kernel_fail()
                case_entry["error_source"] = "kernel"
                case_entry["error"] = "kernel output != reference"
                for d in cmp["diagnostics"]:
                    diagnostics.append(f"[case {idx}] {d}")
        except Exception as e:
            overall_correct = False
            failed_indices.append(idx)
            _tag_kernel_fail()
            err = f"{type(e).__name__}: {e}"
            case_entry["error"] = f"compare: {err}"
            case_entry["error_source"] = "kernel"
            diagnostics.append(skill_verify.truncate_error(
                f"[case {idx}] compare: {err}\n{traceback.format_exc()}"
            ))
        finally:
            del impl_model
            skill_verify.cleanup_npu_memory()

        per_case.append(case_entry)

    return {
        "correctness": overall_correct,
        "error_source": overall_error_source,  # "ref" | "kernel" | None
        "ref_source": "computed",
        "num_cases": num_cases,
        "failed_indices": failed_indices,
        "per_case": per_case,
        "diagnostics": diagnostics,
        # Worst-case detail comes from compare_outputs in utils.correctness;
        # skill's allclose path does not surface them. Keep keys present so
        # eval_client doesn't trip KeyError on the failure-detail path.
        "worst_idx": None,
        "worst_max_abs_diff": None,
    }


def run_profile(target_cls, ref_mod, cases_cpu, init_inputs, device,
                warmup: int, repeats: int, mode: str,
                skill_benchmark) -> dict:
    """Per-case latency via inline `_measure_npu_event_ms`.

    `skill_benchmark` is still passed in so we can reuse its
    `cleanup_npu_memory` between cases — that helper lives in the skill
    and we don't want to duplicate it here. Timing itself does not go
    through the skill (see `_measure_npu_event_ms` for the why).

    `mode` is "gen" (kernel) or "base" (PyTorch reference); kept on the
    per-shape dict so downstream readers can tell which side a timing
    came from.
    """
    import math
    from utils.input_groups import describe_case as _describe_case

    per_shape: list = []
    for idx, case in enumerate(cases_cpu):
        model = None
        try:
            _seed_npu()
            model = _build_target(target_cls, init_inputs, device)
            inputs_dev = [x.to(device) if hasattr(x, "to") else x
                          for x in case]
            latency_ms = _measure_npu_event_ms(
                model, inputs_dev, warmup, repeats,
            )
            if (latency_ms is None or latency_ms <= 0
                    or latency_ms == float("inf")):
                raise RuntimeError(
                    f"_measure_npu_event_ms returned invalid "
                    f"ms={latency_ms!r}")
            avg_us = float(latency_ms) * 1000.0
            method = "npu_event"
        except Exception as e:
            print(f"[profile {mode}] case {idx} benchmark failed: {e} "
                  f"(case marked inf so it doesn't poison the aggregate)",
                  file=sys.stderr)
            traceback.print_exc()
            avg_us = float("inf")
            method = None
        per_shape.append({
            "idx": idx,
            "case_desc": _describe_case(case, model),
            "avg_time_us": avg_us,
            "method": method,
        })
        del model
        skill_benchmark.cleanup_npu_memory()

    # Aggregate over cases that produced finite timing — `eval_client`
    # already filters per-shape speedups the same way for the geomean.
    finites = [s["avg_time_us"] for s in per_shape
               if isinstance(s["avg_time_us"], (int, float))
               and math.isfinite(s["avg_time_us"])]
    agg_us = sum(finites) / len(finites) if finites else float("inf")
    return {
        "avg_time_us": agg_us,
        "execution_time_us": agg_us,
        "execution_time_ms": (agg_us / 1000.0) if finites else None,
        "warmup_times": warmup,
        "run_times": repeats,
        "num_cases": len(per_shape),
        "per_shape": per_shape,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _wrap_phase_error(phase: str, e: Exception) -> dict:
    return {
        "phase": phase,
        "type": type(e).__name__,
        "msg": str(e),
        "trace": traceback.format_exc(),
    }


def main():
    ap = argparse.ArgumentParser(
        description="autoresearch eval orchestrator (verify + profile)")
    ap.add_argument("--task-dir", required=True,
                    help="task directory containing reference.py + kernel.py")
    ap.add_argument("--op-name", required=True,
                    help="operator name (recorded in result; not used for "
                         "file naming since we load by path)")
    ap.add_argument("--kernel-file", required=True,
                    help="kernel module name without .py (default convention: kernel)")
    ap.add_argument("--ref-file", required=True,
                    help="reference module name without .py (default convention: reference)")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--phases", default="verify,profile_gen,profile_base",
                    help="comma-separated subset of {verify, profile_gen, "
                         "profile_base}; run order matters because verify "
                         "warms the JIT cache profile_gen reuses")
    ap.add_argument("--output", default=None,
                    help="JSON sidecar path (default: <task_dir>/.eval_result.json)")
    args = ap.parse_args()

    requested = {p.strip() for p in args.phases.split(",") if p.strip()}
    valid = {"verify", "profile_gen", "profile_base"}
    bad = requested - valid
    if bad:
        print(f"unknown phase(s): {sorted(bad)}; valid: {sorted(valid)}",
              file=sys.stderr)
        sys.exit(2)

    task_dir = os.path.abspath(args.task_dir)
    # __file__ is scripts/engine/eval_kernel.py — climb one to scripts/ so
    # `from utils.input_groups import describe_case` resolves at runtime.
    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in (scripts_dir, task_dir):
        if p and p not in sys.path:
            sys.path.insert(0, p)

    # Load skill core functions (verify.run_single_case + benchmark.cleanup_npu_memory)
    # before touching torch — surfaces a missing-skill setup error early.
    try:
        skill_verify, skill_benchmark = _load_skill_modules()
    except Exception as e:
        print(f"[eval_kernel] failed to load kernel-verifier skill: {e}",
              file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    # Device id has to be set before torch_npu init.
    device_id = int(os.environ.get("DEVICE_ID", args.device_id))
    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", str(device_id))
    import torch
    import torch_npu  # noqa: F401
    import triton  # noqa: F401
    import triton.language  # noqa: F401

    device = torch.device("npu:0")

    ref_path = os.path.join(task_dir, args.ref_file + ".py")
    kernel_path = os.path.join(task_dir, args.kernel_file + ".py")

    result: dict = {
        "verify": None,
        "profile_base": None,
        "profile_gen": None,
        "ok": True,
        "errors": [],
    }
    out_path = args.output or os.path.join(task_dir, ".eval_result.json")

    def _write_and_exit(rc: int) -> None:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, default=str)
        print(f"[eval_kernel] result -> {out_path}", file=sys.stderr)
        sys.exit(rc)

    try:
        ref_mod = _load_module("_eval_ref", ref_path)
    except Exception as e:
        # ref-side: source file broken. Scaffold rejects + asks user to
        # fix the source --ref. Surface the verify block too so eval_client
        # can read error_source through the normal path.
        result["ok"] = False
        result["errors"].append(_wrap_phase_error("import_ref", e))
        result["verify"] = {
            "correctness": False,
            "error_source": "ref",
            "error": f"import reference failed: {type(e).__name__}: {e}",
            "num_cases": 0,
            "per_case": [],
            "diagnostics": [],
            "failed_indices": [],
        }
        _write_and_exit(1)

    try:
        from utils.input_groups import resolve as _resolve_groups
        cases_cpu = _resolve_groups(ref_mod)
        init_inputs = ref_mod.get_init_inputs()
        if not cases_cpu:
            raise RuntimeError("reference module returned 0 input cases")
    except Exception as e:
        # ref-side: get_input_groups / get_inputs / get_init_inputs crashed
        # or returned nothing. Same recovery as import_ref above.
        result["ok"] = False
        result["errors"].append(_wrap_phase_error("resolve_cases", e))
        result["verify"] = {
            "correctness": False,
            "error_source": "ref",
            "error": f"resolve cases failed: {type(e).__name__}: {e}",
            "num_cases": 0,
            "per_case": [],
            "diagnostics": [],
            "failed_indices": [],
        }
        _write_and_exit(1)

    kernel_mod = None
    if "verify" in requested or "profile_gen" in requested:
        try:
            kernel_mod = _load_module("_eval_kernel", kernel_path)
        except Exception as e:
            # kernel-side: ModelNew import / module-level code crashed.
            # eval_client treats this as a correctness=False round; PLAN
            # takes over to rewrite the kernel. profile_gen is impossible
            # without a kernel module.
            result["errors"].append(_wrap_phase_error("import_kernel", e))
            if "verify" in requested:
                result["verify"] = {
                    "correctness": False,
                    "error_source": "kernel",
                    "error": f"import kernel failed: {type(e).__name__}: {e}",
                    "num_cases": len(cases_cpu),
                    "per_case": [],
                    "diagnostics": [],
                    "failed_indices": [],
                }
            requested.discard("verify")
            requested.discard("profile_gen")

    # ---- verify (warms JIT cache for profile_gen) ----
    if "verify" in requested:
        try:
            result["verify"] = run_verify(
                ref_mod, kernel_mod, cases_cpu, init_inputs, device,
                skill_verify=skill_verify)
        except Exception as e:
            # run_verify itself crashed (not a per-case failure caught
            # internally). Without more info we conservatively tag this
            # as kernel-side; ref-only crashes would be caught by Phase 1
            # of the per-case loop and surface with error_source="ref".
            result["ok"] = False
            result["errors"].append(_wrap_phase_error("verify", e))
            result["verify"] = {
                "correctness": False,
                "error_source": "kernel",
                "error": f"{type(e).__name__}: {e}",
                "num_cases": len(cases_cpu),
                "per_case": [],
                "diagnostics": [],
                "failed_indices": [],
            }

    # ---- profile_gen (uses warm JIT cache from verify above) ----
    if "profile_gen" in requested:
        try:
            result["profile_gen"] = run_profile(
                kernel_mod.ModelNew, ref_mod, cases_cpu, init_inputs, device,
                warmup=args.warmup, repeats=args.repeats, mode="gen",
                skill_benchmark=skill_benchmark)
        except Exception as e:
            result["ok"] = False
            result["errors"].append(_wrap_phase_error("profile_gen", e))

    # ---- profile_base (PyTorch reference timing) ----
    if "profile_base" in requested:
        try:
            result["profile_base"] = run_profile(
                ref_mod.Model, ref_mod, cases_cpu, init_inputs, device,
                warmup=args.warmup, repeats=args.repeats, mode="base",
                skill_benchmark=skill_benchmark)
        except Exception as e:
            result["ok"] = False
            result["errors"].append(_wrap_phase_error("profile_base", e))

    _write_and_exit(0)


if __name__ == "__main__":
    main()
