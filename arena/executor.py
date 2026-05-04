"""
Code Arena – Safe Code Executor
=================================
Runs student code in a sandboxed subprocess with timeout and import restrictions.
Supports Python, Java, C++, C, and JavaScript.
"""

import subprocess
import tempfile
import os
import re
from typing import Dict, List, Any


BLOCKED_IMPORTS_PYTHON = {
    "os", "sys", "subprocess", "socket", "shutil",
    "pathlib", "importlib", "ctypes", "signal",
    "multiprocessing", "threading", "http", "urllib",
    "requests", "pickle", "shelve", "sqlite3",
    "webbrowser", "tkinter", "code", "codeop",
}


def _check_blocked_imports_python(code: str) -> str | None:
    """Return name of blocked import if found, else None."""
    for mod in BLOCKED_IMPORTS_PYTHON:
        patterns = [
            rf'\bimport\s+{mod}\b',
            rf'\bfrom\s+{mod}\b',
            rf'__import__\s*\(\s*["\']({mod})',
        ]
        for pat in patterns:
            if re.search(pat, code):
                return mod
    return None


def _check_dangerous_code(code: str, language: str) -> str | None:
    """Check for dangerous operations in any language."""
    dangerous = [
        r'system\s*\(', r'exec\s*\(', r'popen\s*\(',
        r'Runtime\.getRuntime', r'ProcessBuilder',
        r'File\s*\(', r'FileWriter', r'FileReader',
        r'fopen\s*\(', r'fwrite\s*\(', r'fread\s*\(',
        r'require\s*\(\s*["\']child_process',
        r'require\s*\(\s*["\']fs',
    ]
    for pat in dangerous:
        if re.search(pat, code):
            return pat
    return None


def _get_safe_env():
    """Minimal environment for subprocess."""
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/usr/local/bin"),
        "HOME": tempfile.gettempdir(),
    }


def _is_cloud_environment() -> bool:
    """Detect if running on Streamlit Cloud (sandboxed, no subprocess)."""
    # Streamlit Cloud sets HOSTNAME to a container ID and has limited /proc
    return (
        os.environ.get("STREAMLIT_SHARING_MODE") is not None
        or os.environ.get("IS_STREAMLIT_CLOUD") is not None
        or (os.path.exists("/mount/src") and os.environ.get("HOME", "").startswith("/home/appuser"))
    )


CLOUD_ERROR = {
    "stdout": "",
    "stderr": "⚠️ Code execution is not available on the cloud deployment. "
              "Please run this app locally to use the Code Arena.",
    "passed": False,
    "timed_out": False,
    "blocked_import": None,
}


# ── Python Executor ─────────────────────────────────────────────────────────

def _execute_python(code: str, test_input: str, expected: str, timeout: int) -> Dict[str, Any]:
    blocked = _check_blocked_imports_python(code)
    if blocked:
        return {"stdout": "", "stderr": f"SecurityError: Import of '{blocked}' is not allowed.",
                "passed": False, "timed_out": False, "blocked_import": blocked}

    harness = f"""{code}

# --- Test Harness ---
try:
    result = eval("solution({test_input})")
    print(repr(result))
except Exception as _e:
    print(f"ERROR: {{_e}}")
"""
    return _run_file(harness, ".py", ["python3"], expected, timeout)


# ── JavaScript Executor ─────────────────────────────────────────────────────

def _execute_javascript(code: str, test_input: str, expected: str, timeout: int) -> Dict[str, Any]:
    harness = f"""{code}

// --- Test Harness ---
try {{
    const result = solution({test_input});
    console.log(typeof result === 'object' ? JSON.stringify(result) : String(result));
}} catch(e) {{
    console.log("ERROR: " + e.message);
}}
"""
    return _run_file(harness, ".js", ["node"], expected, timeout)


# ── Common Runner ────────────────────────────────────────────────────────────

def _run_file(code: str, ext: str, cmd: List[str], expected: str, timeout: int) -> Dict[str, Any]:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False, dir=tempfile.gettempdir()) as f:
            f.write(code)
            tmp_path = f.name

        proc = subprocess.run(
            cmd + [tmp_path], capture_output=True, text=True,
            timeout=timeout, env=_get_safe_env(),
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        return {"stdout": stdout, "stderr": stderr,
                "passed": _compare_output(stdout, expected.strip()),
                "timed_out": False, "blocked_import": None}

    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Time Limit Exceeded (>{timeout}s)",
                "passed": False, "timed_out": True, "blocked_import": None}
    except FileNotFoundError:
        return {"stdout": "", "stderr": f"Runtime ({cmd[0]}) not found.",
                "passed": False, "timed_out": False, "blocked_import": None}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "passed": False,
                "timed_out": False, "blocked_import": None}
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ── Output Comparison ────────────────────────────────────────────────────────

def _compare_output(actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    try:
        return eval(actual) == eval(expected)
    except Exception:
        pass
    return actual.strip("'\"") == expected.strip("'\"")


# ── Public API ───────────────────────────────────────────────────────────────

def execute_code(
    code: str,
    test_input: str,
    expected_output: str,
    timeout: int = 5,
    language: str = "Python",
) -> Dict[str, Any]:
    """Execute student code against a single test case.

    Supports: Python, JavaScript.
    """
    # Graceful degradation on Streamlit Cloud (subprocess is sandboxed)
    if _is_cloud_environment():
        return CLOUD_ERROR

    dangerous = _check_dangerous_code(code, language)
    if dangerous:
        return {"stdout": "", "stderr": "SecurityError: Dangerous operation detected.",
                "passed": False, "timed_out": False, "blocked_import": dangerous}

    if language == "Python":
        return _execute_python(code, test_input, expected_output, timeout)
    elif language == "JavaScript":
        return _execute_javascript(code, test_input, expected_output, timeout)
    else:
        return _execute_python(code, test_input, expected_output, timeout)


def run_all_tests(
    code: str,
    visible_examples: List[dict],
    hidden_tests: List[dict],
    timeout: int = 5,
    language: str = "Python",
) -> Dict[str, Any]:
    """Run code against all test cases (visible + hidden).

    Returns:
        dict with: results (list), total_passed, total_tests, all_passed
    """
    results = []

    for i, ex in enumerate(visible_examples):
        inp = ex.get("input", "")
        exp = ex.get("output", ex.get("expected", ""))
        res = execute_code(code, inp, exp, timeout, language)
        res["test_num"] = i + 1
        res["test_type"] = "visible"
        res["input"] = inp
        res["expected"] = exp
        results.append(res)

    for i, tc in enumerate(hidden_tests):
        inp = tc.get("input", "")
        exp = tc.get("expected", tc.get("output", ""))
        res = execute_code(code, inp, exp, timeout, language)
        res["test_num"] = len(visible_examples) + i + 1
        res["test_type"] = "hidden"
        res["input"] = inp
        res["expected"] = exp
        results.append(res)

    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    return {
        "results": results,
        "total_passed": passed,
        "total_tests": total,
        "all_passed": passed == total,
    }
