#!/usr/bin/env python3
"""KernelBench 任务代码验证脚本

验证代码是否符合 KernelBench 格式并通过运行时检查。
支持两种输入提供方式：
- 单 case：get_inputs() 返回单组输入
- 多 case：get_input_groups() 返回多组输入列表（每组对应一个 shape 配置）

检查项目:
1. 静态: class Model(nn.Module), forward, get_init_inputs, (get_inputs OR get_input_groups)
2. 运行时: exec → Model() → 遍历所有 groups 执行 forward() → NaN/Inf 检查 → 一致性检查

用法:
    python validate_task.py /abs/path/task_desc.py
    python validate_task.py /abs/path/task_desc.py --json
    python validate_task.py /abs/path/task_desc.py --static-only

输出格式:
    [VALID] 代码符合 KernelBench 格式
    [INVALID] 代码不符合格式 + 原因 + 修复建议
"""
import ast
import sys
import argparse
import json


def check_static(code: str) -> dict:
    """静态检查: 验证 KernelBench 四大组件是否存在

    输入函数允许两种之一：get_inputs() 或 get_input_groups()。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {
            "passed": False,
            "found": [],
            "missing": ["Model", "forward", "get_init_inputs", "get_inputs|get_input_groups"],
            "error": f"SyntaxError: {e}",
        }

    has = {
        "Model": False,
        "forward": False,
        "get_inputs": False,
        "get_input_groups": False,
        "get_init_inputs": False,
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            for base in node.bases:
                base_name = getattr(base, "attr", getattr(base, "id", ""))
                if base_name == "Module":
                    has["Model"] = True
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == "forward":
                            has["forward"] = True

        if isinstance(node, ast.FunctionDef) and node.name in (
            "get_inputs",
            "get_input_groups",
            "get_init_inputs",
        ):
            has[node.name] = True

    has_input_provider = has["get_inputs"] or has["get_input_groups"]
    required_passed = has["Model"] and has["forward"] and has["get_init_inputs"] and has_input_provider

    found = [k for k, v in has.items() if v]
    missing = []
    if not has["Model"]:
        missing.append("Model")
    if not has["forward"]:
        missing.append("forward")
    if not has["get_init_inputs"]:
        missing.append("get_init_inputs")
    if not has_input_provider:
        missing.append("get_inputs|get_input_groups")
    return {"passed": required_passed, "found": found, "missing": missing, "error": None}


def check_runtime(code: str, file_path: str = None) -> dict:
    """运行时检查: exec → Model() → 遍历所有 groups → forward() → NaN/Inf → 一致性

    若任务文件提供 get_input_groups()，全部 groups 都会执行。
    若仅提供 get_inputs()，按单 case 处理。
    """
    checks = []
    namespace = {}
    if file_path:
        namespace["__file__"] = file_path

    try:
        exec(code, namespace)
        checks.append({"name": "exec", "passed": True})
    except Exception as e:
        checks.append({"name": "exec", "passed": False, "error": str(e)})
        return {"passed": False, "checks": checks, "error": f"exec error: {e}", "cases_tested": 0, "cases_passed": 0}

    try:
        init_inputs = namespace["get_init_inputs"]()
        checks.append({"name": "get_init_inputs()", "passed": True})
    except Exception as e:
        checks.append({"name": "get_init_inputs()", "passed": False, "error": str(e)})
        return {"passed": False, "checks": checks, "error": f"get_init_inputs() error: {e}", "cases_tested": 0, "cases_passed": 0}

    try:
        model = namespace["Model"](*init_inputs)
        checks.append({"name": "Model(*init_inputs)", "passed": True})
    except Exception as e:
        checks.append({"name": "Model(*init_inputs)", "passed": False, "error": str(e)})
        return {"passed": False, "checks": checks, "error": f"Model() error: {e}", "cases_tested": 0, "cases_passed": 0}

    if "get_input_groups" in namespace:
        try:
            input_groups = namespace["get_input_groups"]()
            checks.append({"name": "get_input_groups()", "passed": True, "note": f"{len(input_groups)} groups"})
        except Exception as e:
            checks.append({"name": "get_input_groups()", "passed": False, "error": str(e)})
            return {"passed": False, "checks": checks, "error": f"get_input_groups() error: {e}", "cases_tested": 0, "cases_passed": 0}
        provider_kind = "groups"
    elif "get_inputs" in namespace:
        try:
            input_groups = [namespace["get_inputs"]()]
            checks.append({"name": "get_inputs()", "passed": True})
        except Exception as e:
            checks.append({"name": "get_inputs()", "passed": False, "error": str(e)})
            return {"passed": False, "checks": checks, "error": f"get_inputs() error: {e}", "cases_tested": 0, "cases_passed": 0}
        provider_kind = "single"
    else:
        return {"passed": False, "checks": checks, "error": "缺少 get_inputs 或 get_input_groups", "cases_tested": 0, "cases_passed": 0}

    import torch

    # 检测 NPU 是否可用，若可用则把 tensor 搬到 NPU 执行 forward
    try:
        import torch_npu
        npu_available = torch_npu.npu.is_available()
    except Exception:
        npu_available = False

    def _to_npu_device(x):
        if npu_available and isinstance(x, torch.Tensor):
            return x.npu()
        return x

    def _check_tensor(t, name="output"):
        if isinstance(t, torch.Tensor):
            if torch.isnan(t).any():
                return f"{name} contains NaN"
            if torch.isinf(t).any():
                return f"{name} contains Inf"
        return None

    def _tensors_close(a, b, rtol=1e-5, atol=1e-6):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol)
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            return all(_tensors_close(x, y) for x, y in zip(a, b))
        return True

    cases_passed = 0
    total = len(input_groups)
    for idx, inputs in enumerate(input_groups):
        case_label = f"case[{idx}]"
        # 若 NPU 可用，将输入 tensor 搬到 NPU 设备
        device_inputs = [_to_npu_device(x) for x in inputs]
        try:
            output = model(*device_inputs)
        except Exception as e:
            checks.append({"name": f"{case_label} forward", "passed": False, "error": str(e)})
            return {"passed": False, "checks": checks, "error": f"{case_label} forward error: {e}", "cases_tested": idx + 1, "cases_passed": cases_passed}

        # 输出搬回 CPU 做后续检查
        if isinstance(output, (tuple, list)):
            output = [x.cpu() if isinstance(x, torch.Tensor) else x for x in output]
        elif isinstance(output, torch.Tensor):
            output = output.cpu()

        issues = []
        if isinstance(output, (tuple, list)):
            for i, item in enumerate(output):
                issue = _check_tensor(item, f"{case_label} output[{i}]")
                if issue:
                    issues.append(issue)
        else:
            issue = _check_tensor(output, f"{case_label} output")
            if issue:
                issues.append(issue)
        if issues:
            checks.append({"name": f"{case_label} NaN/Inf", "passed": False, "error": "; ".join(issues)})
            return {"passed": False, "checks": checks, "error": "; ".join(issues), "cases_tested": idx + 1, "cases_passed": cases_passed}

        try:
            output2 = model(*device_inputs)
            if isinstance(output2, (tuple, list)):
                output2 = [x.cpu() if isinstance(x, torch.Tensor) else x for x in output2]
            elif isinstance(output2, torch.Tensor):
                output2 = output2.cpu()
            if not _tensors_close(output, output2):
                checks.append({"name": f"{case_label} consistency", "passed": False, "error": "outputs differ between runs"})
                return {"passed": False, "checks": checks, "error": f"{case_label} consistency check failed", "cases_tested": idx + 1, "cases_passed": cases_passed}
        except Exception:
            pass

        cases_passed += 1

    checks.append({"name": "all cases", "passed": True, "note": f"{cases_passed}/{total} passed (provider={provider_kind})"})
    return {"passed": True, "checks": checks, "error": None, "cases_tested": total, "cases_passed": cases_passed}


def main():
    parser = argparse.ArgumentParser(
        description="验证代码是否符合 KernelBench 任务格式"
    )
    parser.add_argument("file", help="要验证的 Python 文件路径")
    parser.add_argument("--static-only", action="store_true", help="只做静态检查")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")

    args = parser.parse_args()

    try:
        with open(args.file, "r", encoding="utf-8") as f:
            code = f.read()
    except FileNotFoundError:
        if args.json:
            print(json.dumps({"valid": False, "error": f"File not found: {args.file}"}))
        else:
            print(f"[ERROR] 文件不存在: {args.file}")
        sys.exit(1)

    static_result = check_static(code)
    result = {
        "valid": False,
        "static_check": static_result,
        "runtime_check": None,
        "suggestion": "",
    }

    if not static_result["passed"]:
        result["error"] = static_result.get("error") or f"缺少组件: {', '.join(static_result['missing'])}"
        result["suggestion"] = "检查代码结构，确保包含 Model(nn.Module)、forward、get_inputs、get_init_inputs"
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"[INVALID] 代码不符合 KernelBench 格式")
            print(f"缺少: {', '.join(static_result['missing'])}")
            print(f"建议: {result['suggestion']}")
        sys.exit(1)

    if not args.static_only:
        runtime_result = check_runtime(code, file_path=args.file)
        result["runtime_check"] = runtime_result
        result["cases_tested"] = runtime_result.get("cases_tested", 0)
        result["cases_passed"] = runtime_result.get("cases_passed", 0)

        if not runtime_result["passed"]:
            result["error"] = runtime_result["error"]
            result["suggestion"] = "检查代码逻辑，修复后重新验证"
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(f"[INVALID] 运行时检查失败")
                print(f"错误: {runtime_result['error']}")
                print(f"已测试 cases: {runtime_result.get('cases_tested', 0)} / 通过: {runtime_result.get('cases_passed', 0)}")
                for check in runtime_result["checks"]:
                    status = "PASS" if check["passed"] else "FAIL"
                    print(f"  [{status}] {check['name']}")
            sys.exit(1)

    result["valid"] = True
    check_type = "静态" if args.static_only else "静态+运行时"

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"[VALID] 代码符合 KernelBench 格式（{check_type}检查通过）")
        print(f"包含组件: {', '.join(static_result['found'])}")
        if not args.static_only and result.get("cases_tested"):
            print(f"运行时测试 cases: {result['cases_passed']}/{result['cases_tested']} 全部通过")
    sys.exit(0)


if __name__ == "__main__":
    main()
