from typing import Any, cast

import sympy as smp
from sympy import Indexed
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.printing.julia import JuliaCodePrinter

from .parse_julia_functions import julia_functions


def shift_sympy_indices(expr: smp.Basic, shift: int = 1) -> smp.Basic:
    """Shift literal-integer indices in Indexed(...) and MatrixElement(...) by `shift`.

    - Only shifts indices that are SymPy Integers (e.g., 0, 15).
    - Leaves symbolic indices (i, j) unchanged.
    """

    def shift_idx(i: smp.Basic) -> smp.Basic:
        if isinstance(i, smp.Integer):
            return i + shift
        return i

    # SymPy stubs for .replace can confuse type checkers, so we cast after each call.
    expr2 = cast(
        smp.Basic,
        expr.replace(
            lambda e: isinstance(e, Indexed),
            lambda e: Indexed(e.base, *(shift_idx(i) for i in e.indices)),
        ),
    )

    expr3 = cast(
        smp.Basic,
        expr2.replace(
            lambda e: isinstance(e, MatrixElement),
            lambda e: MatrixElement(e.parent, shift_idx(e.i), shift_idx(e.j)),
        ),
    )

    return expr3


class CustomJuliaCodePrinter(JuliaCodePrinter):
    """Custom Julia code printer that handles our library's Julia functions.

    This printer also shifts literal integer indices in Indexed/MatrixElement
    by `index_shift` once per top-level doprint call.
    """

    def __init__(self, settings: dict[str, Any] | None = None):
        super().__init__(settings)
        self._known_functions = getattr(self, "_known_functions", {}).copy()
        for func_name in julia_functions:
            self._known_functions[func_name] = func_name

        # Default shift can be overridden via settings={"index_shift": N}
        self._index_shift: int = int(
            getattr(self, "settings", {}).get("index_shift", 1)
        )

        # Guard to ensure we shift only once per doprint call.
        self._shifting_enabled: bool = True

    def doprint(
        self,
        expr: smp.Basic,
        assign_to: Any = None,
    ) -> str:
        if self._shifting_enabled:
            self._shifting_enabled = False
            try:
                expr = shift_sympy_indices(expr, shift=self._index_shift)
                return cast(str, super().doprint(expr, assign_to))
            finally:
                self._shifting_enabled = True

        return cast(str, super().doprint(expr, assign_to))

    def _print_Function(self, expr: smp.Expr) -> str:
        """Override to handle our custom Julia functions."""
        func_name = str(expr.func)

        if func_name in julia_functions:
            args = [self._print(arg) for arg in expr.args]  # type: ignore[attr-defined]
            return f"{func_name}({', '.join(args)})"

        return cast(str, super()._print_Function(expr))  # type: ignore[misc]


def custom_julia_code(expr: smp.Basic, **settings: Any) -> str:
    """Generate Julia code using our custom printer."""
    printer = CustomJuliaCodePrinter(settings)
    return cast(str, printer.doprint(expr))
