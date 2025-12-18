from pathlib import Path
from typing import Any, Optional

import sympy as smp

from .parse_julia_functions import (
    _parse_julia_function_signature,
    get_julia_function,
    julia_functions,
)


def create_julia_function_with_postfix(func_name: str, param_postfix: str) -> Any:
    """Create a Julia function with postfixed parameter names.

    Args:
        func_name: Name of the base Julia function (e.g., "phase_modulation_sinusoidal")
        param_postfix: Prefix for parameter names (e.g., "rabi1", "laser2")

    Returns:
        SymPy Function with postfixed parameter names

    Example:
        >>> phase_mod_rabi1 = create_julia_function_with_postfix("phase_modulation_sinusoidal", "rabi1")
        >>> print(phase_mod_rabi1)  # phase_modulation_sinusoidal(t, Omega_rabi1, phi_rabi1, freq_rabi1)
    """
    if func_name not in julia_functions:
        raise KeyError(
            f"Julia function '{func_name}' not found. Available functions: {julia_functions}"
        )

    # Get the base function to extract its parameter structure
    base_func = get_julia_function(func_name)

    # Extract parameter names from the base function
    if hasattr(base_func, "args") and base_func.args:
        param_names = [str(arg) for arg in base_func.args]
    else:
        # Fallback: parse from Julia file if function has no args
        julia_file = Path(__file__).parent / "julia_common.jl"
        if julia_file.exists():
            julia_code_text = julia_file.read_text(encoding="utf-8")
            param_names = _parse_julia_function_signature(julia_code_text, func_name)
        else:
            param_names = []

    if not param_names:
        # Return function with just 't' if no parameters found
        return smp.Function(func_name)(smp.Symbol("t"))

    # Create new parameter symbols, keeping 't' unchanged and adding postfix to others
    new_params = []
    for param_name in param_names:
        if param_name == "t":
            new_params.append(smp.Symbol("t"))
        else:
            new_params.append(smp.Symbol(f"{param_name}_{param_postfix}"))

    # Create and return the new function
    return smp.Function(func_name)(*new_params)


def substitute_julia_function_param(
    julia_func: Any, param_name: str, substitution: smp.Basic
) -> Any:
    """Substitute a parameter in a Julia function with a variable or expression.

    Args:
        julia_func: The Julia function to modify (e.g., from create_julia_function_with_postfix)
        param_name: Name of the parameter to substitute (with or without postfix)
        substitution: Variable or expression to substitute (can be Symbol, number, or expression)

    Returns:
        New SymPy Function with the specified parameter substituted

    Example:
        >>> phase_mod = create_julia_function_with_postfix("phase_modulation_sinusoidal", "rabi1")
        >>> # Substitute Omega_rabi1 with a specific value or variable
        >>> phase_mod_custom = substitute_julia_function_param(phase_mod, "Omega", smp.Symbol("my_omega"))
        >>> print(phase_mod_custom)  # phase_modulation_sinusoidal(t, my_omega, phi_rabi1, freq_rabi1)
        >>>
        >>> # Or substitute exact parameter name
        >>> phase_mod_exact = substitute_julia_function_param(phase_mod, "Omega_rabi1", smp.Symbol("my_omega"))
    """
    if not hasattr(julia_func, "args") or not julia_func.args:
        return julia_func

    # Get function name
    func_name = str(julia_func.func)

    # Create substitution mapping
    subs_dict = {}

    # Look for parameters that match either exact name or pattern param_name_*
    for arg in julia_func.args:
        arg_str = str(arg)
        # Check if this argument matches exactly or follows the postfix pattern
        if arg_str == param_name or arg_str.startswith(f"{param_name}_"):
            subs_dict[arg] = substitution

    # Apply substitution to get new arguments
    new_args = []
    for arg in julia_func.args:
        if arg in subs_dict:
            new_args.append(subs_dict[arg])
        else:
            new_args.append(arg)

    # Create new function with substituted arguments
    return smp.Function(func_name)(*new_args)


def create_julia_function_substituted(
    func_name: str,
    param_postfix: Optional[str] = None,
    substitutions: Optional[dict[smp.Symbol, smp.Symbol | float | int]] = None,
) -> Any:
    """Create a Julia function with optional postfixed parameters and optional substitutions.

    Args:
        func_name: Name of the base Julia function
        param_postfix: Postfix for parameter names (optional, if None uses original names)
        substitutions: Dict mapping parameter names (with or without postfix) to substitutions

    Returns:
        SymPy Function with optionally postfixed and substituted parameters

    Example:
        >>> # Create phase modulation with custom omega and postfix
        >>> phase_mod = create_julia_function_substituted(
        ...     "phase_modulation_sinusoidal",
        ...     "rabi1",
        ...     {"Omega": smp.Symbol("custom_omega")}
        ... )
        >>> print(phase_mod)  # phase_modulation_sinusoidal(t, custom_omega, phi_rabi1, freq_rabi1)
        >>>
        >>> # Create without postfix but with substitutions
        >>> phase_mod_no_postfix = create_julia_function_substituted(
        ...     "phase_modulation_sinusoidal",
        ...     None,
        ...     {"Omega": smp.Symbol("custom_omega")}
        ... )
        >>> print(phase_mod_no_postfix)  # phase_modulation_sinusoidal(t, custom_omega, phi, freq)
    """
    # Create the base function with or without postfix
    if param_postfix is not None:
        julia_func = create_julia_function_with_postfix(func_name, param_postfix)
    else:
        julia_func = get_julia_function(func_name)

    # Apply substitutions if provided
    if substitutions:
        for param_name, substitution in substitutions.items():
            julia_func = substitute_julia_function_param(
                julia_func, param_name, substitution
            )

    return julia_func
