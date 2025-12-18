import re
from pathlib import Path
from typing import Any

import sympy as smp

julia_functions = [
    "gaussian_2d",
    "gaussian_2d_rotated",
    "phase_modulation",
    "square_wave",
    "resonant_polarization_modulation",
    "sawtooth_wave",
    "variable_on_off",
    "multipass_2d_intensity",
    "rabi_from_intensity",
    "multipass_2d_rabi",
    "gaussian_beam_rabi",
    "variable_on_off_duty",
]


def _parse_julia_function_signature(julia_code_text: str, func_name: str) -> list[str]:
    """Parse a Julia function signature to extract positional parameter names."""
    # 1) Locate the start of the signature
    pat = rf"(?:@inline\s+)?function\s+{re.escape(func_name)}\s*\("
    m = re.search(pat, julia_code_text)
    if not m:
        return []
    # 2) Walk forward to find the matching ')' (handles nested/bracketed defaults)
    start = m.end() - 1
    depth = 0
    for idx, ch in enumerate(julia_code_text[start:], start):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = idx
                break
    else:
        return []
    raw = julia_code_text[start + 1 : end].strip()
    if not raw:
        return []
    # 3) Drop anything after ';' (keyword args) or a 'where' clause
    raw = re.split(r";|where\b", raw, maxsplit=1)[0].strip()
    # 4) Split at top‐level commas
    parts = _split_params(raw)
    # 5) Clean up each part (drop ::types, =defaults)
    names = []
    for p in parts:
        p = p.split("::", 1)[0].split("=", 1)[0].strip()
        if p:
            names.append(p)
    return names


def _split_params(s: str) -> list[str]:
    """Split a comma‐separated param string at top‐level commas only."""
    params: list[str] = []
    buf: list[str] = []
    counts = {"(": 0, "[": 0, "{": 0, "<": 0}
    pairs = {")": "(", "]": "[", "}": "{", ">": "<"}
    for ch in s:
        if ch in counts:
            counts[ch] += 1
        elif ch in pairs:
            counts[pairs[ch]] -= 1
        if ch == "," and all(v == 0 for v in counts.values()):
            params.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf and "".join(buf).strip():
        params.append("".join(buf).strip())
    return params


def _create_sympy_functions() -> dict[str, smp.Function]:
    """Create SymPy functions based on Julia function signatures.

    Returns:
        Dictionary mapping function names to SymPy Function objects
    """
    # Read the Julia functions file
    julia_file = Path(__file__).parent / "julia_common.jl"

    if not julia_file.exists():
        return {}

    julia_code_text = julia_file.read_text(encoding="utf-8")
    sympy_functions = {}

    for func_name in julia_functions:
        param_names = _parse_julia_function_signature(julia_code_text, func_name)

        if param_names:
            # Create SymPy symbols for parameters
            symbols = [smp.Symbol(name) for name in param_names]
            # Create the SymPy function
            sympy_functions[func_name] = smp.Function(func_name)(*symbols)
        else:
            # If we can't parse parameters, create a generic function
            sympy_functions[func_name] = smp.Function(func_name)

    return sympy_functions


# Create the SymPy functions dictionary on import
sympy_julia_functions = _create_sympy_functions()


def get_julia_function(name: str) -> Any:
    """Get a SymPy function representation of a Julia function.

    Args:
        name: Name of the Julia function

    Returns:
        SymPy Function object with appropriate arguments

    Raises:
        KeyError: If the function name is not found

    Example:
        >>> gauss = get_julia_function("gaussian_peak")
        >>> print(gauss)  # gaussian_peak(x, μ, σ)
    """
    if name not in sympy_julia_functions:
        raise KeyError(
            f"Julia function '{name}' not found. Available functions: {list(sympy_julia_functions.keys())}"
        )
    return sympy_julia_functions[name]


def list_julia_functions() -> dict[str, Any]:
    """Get all available Julia functions as SymPy Function objects.

    Returns:
        Dictionary mapping function names to SymPy Function objects
    """
    return sympy_julia_functions.copy()
