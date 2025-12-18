from typing import Any, Optional, cast

import sympy as smp
from sympy import Indexed
from sympy.matrices.expressions.matexpr import MatrixElement

from .julia_code_printer import custom_julia_code
from .parse_julia_functions import julia_functions, sympy_julia_functions
from .utils_julia_matrix import generate_code_matrix_method


def _get_triangle_indices(rows: int, uplo: str) -> list[tuple[int, int]]:
    """Generate indices for triangle elements.

    Args:
        rows: Number of rows (assumes square matrix)
        uplo: 'U' for upper triangle, 'L' for lower triangle

    Returns:
        List of (i, j) tuples for the specified triangle
    """
    if uplo == "U":
        return [(i, j) for i in range(rows) for j in range(i, rows)]
    else:  # uplo == "L"
        return [(i, j) for i in range(rows) for j in range(0, i + 1)]


def _build_function_signature(
    func_name: str, output_name: str, matrix_var_name: str | None, args: str
) -> str:
    """Build Julia function signature.

    Args:
        func_name: Name of the function
        output_name: Name of the output parameter
        matrix_var_name: Name of the matrix variable (if any)
        args: Additional arguments string

    Returns:
        Julia function signature line
    """
    if matrix_var_name and args:
        return f"function {func_name}({output_name}, {matrix_var_name}, {args})"
    elif matrix_var_name:
        return f"function {func_name}({output_name}, {matrix_var_name})"
    elif args:
        return f"function {func_name}({output_name}, {args})"
    else:
        return f"function {func_name}({output_name})"


def _transform_matrix_references_for_triangle(
    expr: smp.Expr, matrix_var_name: str | None, uplo: str, mirror: bool
) -> smp.Expr:
    """Transform matrix references in expressions to use the correct triangle.

    When filling only one triangle (mirror=False), matrix references to the opposite
    triangle should be replaced with conj() of the symmetric position from the
    computed triangle.

    For example, if uplo='U' and expression contains u[5,1], it should become conj(u[1,5]).

    Args:
        expr: The SymPy expression to transform
        matrix_var_name: Name of the matrix variable (e.g., 'u'), or None if no matrix refs
        uplo: Which triangle is being computed ('U' or 'L')
        mirror: If True, no transformation needed (mirroring handles it)

    Returns:
        Transformed expression with conjugate references to opposite triangle
    """
    if mirror or matrix_var_name is None:
        # If mirroring, both triangles will be filled, no transformation needed
        return expr

    # Find all Indexed symbols in the expression that match the matrix variable
    def transform_indexed(e):  # type: ignore[no-untyped-def]
        if isinstance(e, Indexed):
            base = e.base
            if hasattr(base, "name") and base.name == matrix_var_name:  # type: ignore[attr-defined]
                indices = e.indices
                if len(indices) == 2:
                    i, j = indices
                    # Check if this reference is in the opposite triangle
                    if uplo == "U":
                        # We're computing upper (i <= j), check if reference is lower (i > j)
                        # For symbolic indices, we check the numeric relationship if possible
                        try:
                            i_val = int(i) if hasattr(i, "__int__") else None  # type: ignore[arg-type]
                            j_val = int(j) if hasattr(j, "__int__") else None  # type: ignore[arg-type]
                            if (
                                i_val is not None
                                and j_val is not None
                                and i_val > j_val
                            ):
                                # This is a lower triangle reference, swap to upper and conjugate
                                return smp.conjugate(Indexed(base, j, i))
                        except (TypeError, ValueError):
                            # Can't determine statically, leave as is
                            pass
                    else:  # uplo == 'L'
                        # We're computing lower (i >= j), check if reference is upper (i < j)
                        try:
                            i_val = int(i) if hasattr(i, "__int__") else None  # type: ignore[arg-type]
                            j_val = int(j) if hasattr(j, "__int__") else None  # type: ignore[arg-type]
                            if (
                                i_val is not None
                                and j_val is not None
                                and i_val < j_val
                            ):
                                # This is an upper triangle reference, swap to lower and conjugate
                                return smp.conjugate(Indexed(base, j, i))
                        except (TypeError, ValueError):
                            pass
        return e

    # Apply transformation recursively
    result = expr.replace(lambda e: isinstance(e, Indexed), transform_indexed)
    return result[0] if isinstance(result, tuple) else result  # type: ignore[return-value]


def _get_triangle_elements(matrix: smp.Matrix, uplo: str) -> list[Any]:
    """Extract triangle elements from a matrix.

    Args:
        matrix: SymPy matrix
        uplo: 'U' for upper triangle, 'L' for lower triangle

    Returns:
        List of expressions from the specified triangle
    """
    rows, cols = matrix.rows, matrix.cols
    if uplo == "U":
        return [matrix[i, j] for i in range(rows) for j in range(i, cols)]  # type: ignore[misc]
    else:  # uplo == "L"
        return [matrix[i, j] for i in range(rows) for j in range(0, i + 1)]  # type: ignore[misc]


def _extract_matrix_variable_and_symbols(
    matrix: smp.Matrix,
) -> tuple[str | None, list[smp.Symbol], str]:
    """Extract matrix variable name and other symbols from a matrix.

    Args:
        matrix: SymPy matrix to analyze

    Returns:
        Tuple of (matrix_var_name, other_syms, args_string)
    """
    from sympy import Indexed

    syms = sorted(matrix.free_symbols, key=lambda s: s.name)
    indexed_syms = [s for s in syms if isinstance(s, Indexed)]

    if indexed_syms:
        first_indexed = indexed_syms[0]
        matrix_var_name = first_indexed.base.name  # type: ignore[attr-defined]
        other_syms = [
            s for s in syms if not isinstance(s, Indexed) and s.name != matrix_var_name
        ]
    else:
        matrix_var_name = None
        other_syms = list(syms)

    # Deduplicate symbols by name (SymPy can have multiple symbols with same name but different assumptions)
    seen_names = set()
    unique_syms = []
    for sym in other_syms:
        if sym.name not in seen_names:
            seen_names.add(sym.name)
            unique_syms.append(sym)
    other_syms = unique_syms

    args = ", ".join(s.name for s in other_syms)
    return matrix_var_name, other_syms, args


def sympy_matrix_to_julia_fill_hermitian(
    matrix: smp.Matrix,
    func_name: str = "fill_matrix!",
    zero_input: bool = True,
    inplace_add: bool = False,
    output_name: str = "matrix",
    uplo: str = "U",
    mirror: bool = True,
) -> tuple[str, smp.Function]:
    """Generate a Julia function that efficiently fills a Hermitian matrix in-place.

    This function converts a SymPy matrix to optimized Julia code that fills a
    Hermitian matrix in-place. The generated Julia function employs several
    optimization techniques:

    1. Common subexpression elimination (CSE) applied only to the specified triangle
    2. Pruning of temporary variables not used in the computed triangle
    3. Inlining of temporary variables used only once
    4. Bulk zero-initialization using SIMD loops with type hoisting
    5. Computing only one triangle (upper or lower) and optionally mirroring
    6. Emitting only non-zero assignments
    7. Automatic transformation of matrix references to use the correct triangle
       (when mirror=False, references to the opposite triangle are transformed to
       use conj() of the symmetric position from the computed triangle)

    Args:
        matrix (smp.Matrix): A SymPy matrix (assumed Hermitian) with symbolic expressions
        func_name (str, optional): Name for the generated Julia function. Defaults to "fill_matrix!".
        zero_input (bool, optional): Whether to zero the input matrix first. Defaults to True.
        inplace_add (bool, optional): Whether to add to existing values instead of replacing. Defaults to False.
        output_name (str, optional): Name of the output matrix parameter. Defaults to "matrix".
        uplo (str, optional): Which triangle to compute: "U" for upper, "L" for lower. Defaults to "U".
        mirror (bool, optional): Whether to fill the other triangle by mirroring. If False,
            only the specified triangle is filled and the other triangle is left untouched.
            This is useful for BLAS operations that only use one triangle. When mirror=False,
            any matrix references in expressions that refer to the opposite triangle are
            automatically transformed to use conj() of the symmetric position. For example,
            if computing the upper triangle (uplo='U') and an expression contains u[5,1],
            it will be transformed to conj(u[1,5]). Defaults to True.

    Returns:
        tuple[str, smp.FunctionClass]:
            str: Julia code string for the optimized matrix-filling function
            smp.FunctionClass: SymPy function object representing the call signature

    Raises:
        ValueError: If the matrix is empty or not square, or if uplo is not "U" or "L"

    Example:
        >>> import sympy as smp
        >>> u = smp.MatrixSymbol('u', 3, 3)
        >>> # Create a matrix where [2,0] references the lower triangle element [2,0]
        >>> matrix = smp.Matrix([[u[0,0], u[0,1], u[0,2]],
        ...                      [smp.conjugate(u[0,1]), u[1,1], u[1,2]],
        ...                      [u[2,0], smp.conjugate(u[1,2]), u[2,2]]])
        >>> code, _ = sympy_matrix_to_julia_fill_hermitian(
        ...     matrix, uplo='U', mirror=False)
        >>> # For element [2,0], the expression u[2,0] is in the lower triangle,
        >>> # so when computing only upper triangle, it won't be computed.
        >>> # But if an upper triangle element references it (e.g., in a different matrix),
        >>> # the reference would be transformed to conj(u[0,2])
    """
    rows, cols = matrix.rows, matrix.cols

    # Input validation
    if rows == 0 or cols == 0:
        raise ValueError("Matrix cannot be empty")
    if rows != cols:
        raise ValueError(
            f"Matrix must be square for Hermitian operations, got {rows}x{cols}"
        )
    if uplo not in ("U", "L"):
        raise ValueError(f"uplo must be 'U' or 'L', got '{uplo}'")

    # Extract matrix variable and symbols
    matrix_var_name, other_syms, args = _extract_matrix_variable_and_symbols(matrix)

    # Get triangle elements and run CSE
    triangle = _get_triangle_elements(matrix, uplo)
    cse_result = smp.cse(triangle, symbols=smp.numbered_symbols("_t"))
    subs_all: list[tuple[smp.Symbol, smp.Expr]] = cse_result[0]  # type: ignore[assignment]
    reduced_triangle: list[smp.Expr] = cse_result[1]  # type: ignore[assignment]

    # Reconstruct full reduced list with Hermitian mirror or conjugate reference
    reduced_all: list[Optional[smp.Expr]] = [None] * (rows * cols)
    idx = 0

    if uplo == "U":
        for i in range(rows):
            for j in range(i, cols):
                expr = reduced_triangle[idx]
                reduced_all[i * cols + j] = expr
                if mirror and i != j:
                    reduced_all[j * cols + i] = expr.conjugate()  # type: ignore[assignment]
                idx += 1
    else:  # uplo == "L"
        for i in range(rows):
            for j in range(0, i + 1):
                expr = reduced_triangle[idx]
                reduced_all[i * cols + j] = expr
                if mirror and i != j:
                    reduced_all[j * cols + i] = expr.conjugate()  # type: ignore[assignment]
                idx += 1

    # 3) Determine which temporary variables are needed for the specified triangle
    tmp_syms = [t for t, _ in subs_all]
    # Collect expressions used in the specified triangle
    if uplo == "U":
        used_exprs = [
            reduced_all[i * cols + j] for i in range(rows) for j in range(i, cols)
        ]
    else:  # uplo == "L"
        used_exprs = [
            reduced_all[i * cols + j] for i in range(rows) for j in range(0, i + 1)
        ]
    needed = set()  # Set to track required temporary variables
    lookup = {t: expr for t, expr in subs_all}  # Map from temp var to its expression

    # Recursively mark all temporary variables that are needed
    def mark(expr):
        for s in expr.free_symbols:
            if s in tmp_syms and s not in needed:
                needed.add(s)
                # Recursively mark any temporaries used in this temporary's definition
                mark(lookup[s])

    # Find all temporary variables needed for the upper triangle expressions
    for expr in used_exprs:
        mark(expr)
    # Keep only the temporary variables that are actually needed
    pruned_subs = [(t, e) for t, e in subs_all if t in needed]

    # 4) Perform final inlining optimization
    usage = {t: 0 for t, _ in pruned_subs}
    all_exprs = [e for _, e in pruned_subs] + used_exprs
    for expr in all_exprs:
        assert expr is not None
        for t in list(usage):
            if t in expr.free_symbols:  # type: ignore[operator]
                usage[t] += 1
    inline = {t for t, cnt in usage.items() if cnt == 1}
    keep = [(t, e) for t, e in pruned_subs if t not in inline]
    inline_map = {t: lookup[t] for t in inline}

    # Set assignment operator and get triangle indices
    assignment = "+=" if inplace_add else "="
    triangle_indices = _get_triangle_indices(rows, uplo)

    # Generate Julia code
    lines = []
    sig = _build_function_signature(func_name, output_name, matrix_var_name, args)
    lines.append(sig)
    lines.append("    @inbounds begin")

    # Initialize matrix if needed
    if zero_input:
        lines.append(f"        fill!({output_name}, .0im)")

    # Add temporary variable declarations
    if keep:
        lines.append("        # Pre-computed expressions")
        for t, expr in keep:
            expr_sub = expr.subs(inline_map)
            lines.append(f"        {t} = {custom_julia_code(expr_sub)}")

    # Fill matrix with non-zero elements
    for i, j in triangle_indices:
        expr_opt = reduced_all[i * cols + j]
        assert expr_opt is not None, f"Expression at ({i},{j}) should not be None"
        expr = expr_opt.subs(inline_map)
        # Transform matrix references to use the correct triangle
        expr = _transform_matrix_references_for_triangle(  # type: ignore[arg-type, assignment]
            expr, matrix_var_name, uplo, mirror
        )
        # Skip zero elements
        is_zero = expr == 0 or (hasattr(expr, "is_zero") and expr.is_zero)
        if not is_zero:
            code = custom_julia_code(expr)
            lines.append(f"        {output_name}[{i + 1},{j + 1}] {assignment} {code}")
            # Mirror for off-diagonal elements (Hermitian property) if requested
            if mirror and i != j:
                lines.append(
                    f"        {output_name}[{j + 1},{i + 1}] {assignment} conj({output_name}[{i + 1},{j + 1}])"
                )

    lines.append("    end")
    lines.append("    nothing")
    lines.append("end")

    # Build the argument list for the SymPy function signature to match Julia signature
    # Order: output_name, matrix_var_name (if present), other_syms
    func_args: list[smp.Symbol] = [smp.Symbol(output_name)]
    if matrix_var_name:
        func_args.append(smp.Symbol(matrix_var_name))
    func_args.extend(other_syms)

    return "\n".join(lines), smp.Function(func_name)(*func_args)  # type: ignore[return-value]


def sympy_matrix_to_julia_add_hermitian(
    matrix: smp.Matrix,
    func_name: str = "add_matrix!",
    output_name: str = "matrix",
    uplo: str = "U",
    mirror: bool = True,
) -> tuple[str, smp.Function]:
    """Generate a Julia function that adds to a Hermitian matrix in-place using temporary variables.

    Unlike sympy_matrix_to_julia_fill_hermitian with inplace_add=True, this function
    uses temporary variables to store computed values before adding them. This is necessary
    when adding to potentially non-zero existing values, because we can't use
    conj(matrix[i,j]) directly - we need to compute the value, add it, then add its
    conjugate to the symmetric position.

    The generated Julia function employs several optimization techniques:
    1. Common subexpression elimination (CSE) applied only to the specified triangle
    2. Inlining of temporary variables used only once
    3. Computing only one triangle (upper or lower) and optionally mirroring
    4. Using temporary variables to store computed values before adding
    5. Automatic transformation of matrix references to use the correct triangle
       (when mirror=False, references to the opposite triangle are transformed to
       use conj() of the symmetric position from the computed triangle)

    Args:
        matrix: A SymPy matrix (assumed Hermitian) with symbolic expressions
        func_name: Name for the generated Julia function. Defaults to "add_matrix!".
        output_name: Name of the output matrix parameter. Defaults to "matrix".
        uplo: Which triangle to compute: "U" for upper, "L" for lower. Defaults to "U".
        mirror: Whether to add to the other triangle by mirroring. If False,
            only the specified triangle is updated and the other triangle is left untouched.
            This is useful for BLAS operations that only use one triangle. When mirror=False,
            any matrix references in expressions that refer to the opposite triangle are
            automatically transformed to use conj() of the symmetric position. Defaults to True.

    Returns:
        Tuple containing:
            - Julia code string for the optimized matrix-adding function
            - SymPy function object representing the call signature

    Raises:
        ValueError: If the matrix is empty or not square, or if uplo is not "U" or "L"
    """
    rows, cols = matrix.rows, matrix.cols

    # Input validation
    if rows == 0 or cols == 0:
        raise ValueError("Matrix cannot be empty")
    if rows != cols:
        raise ValueError(
            f"Matrix must be square for Hermitian operations, got {rows}x{cols}"
        )
    if uplo not in ("U", "L"):
        raise ValueError(f"uplo must be 'U' or 'L', got '{uplo}'")

    # Extract matrix variable and symbols
    matrix_var_name, other_syms, args = _extract_matrix_variable_and_symbols(matrix)

    # Get triangle elements and perform CSE with inlining
    triangle = _get_triangle_elements(matrix, uplo)
    lookup, inline_map, keep = _perform_cse_and_inlining(triangle)

    # Get reduced triangle for final code generation
    cse_result = smp.cse(triangle, symbols=smp.numbered_symbols("_t"))
    reduced_triangle: list[smp.Expr] = cse_result[1]  # type: ignore[assignment]

    # Get triangle indices
    triangle_indices = _get_triangle_indices(rows, uplo)

    # Generate Julia code
    lines = []
    sig = _build_function_signature(func_name, output_name, matrix_var_name, args)
    lines.append(sig)
    lines.append("    @inbounds begin")

    # Add temporary variable declarations for CSE
    if keep:
        lines.append("        # Pre-computed expressions")
        for t, expr in keep:
            expr_sub = expr.subs(inline_map)
            lines.append(f"        {t} = {custom_julia_code(expr_sub)}")

    # Add to matrix with temporary variables for each element

    temp_counter = 0
    for idx, (i, j) in enumerate(triangle_indices):
        expr = reduced_triangle[idx].subs(inline_map)
        # Transform matrix references to use the correct triangle
        expr = _transform_matrix_references_for_triangle(  # type: ignore[arg-type, assignment]
            expr, matrix_var_name, uplo, mirror
        )

        # Skip zero elements
        is_zero = expr == 0 or (hasattr(expr, "is_zero") and expr.is_zero)
        if not is_zero:
            code = custom_julia_code(expr)
            temp_name = f"_tmp{temp_counter}"
            temp_counter += 1

            # Create temporary variable with the computed value
            lines.append(f"        {temp_name} = {code}")
            # Add to the triangle position
            lines.append(f"        {output_name}[{i + 1},{j + 1}] += {temp_name}")

            # Mirror for off-diagonal elements (Hermitian property) if requested
            if mirror and i != j:
                lines.append(
                    f"        {output_name}[{j + 1},{i + 1}] += conj({temp_name})"
                )

    lines.append("    end")
    lines.append("    nothing")
    lines.append("end")

    # Build the argument list for the SymPy function signature
    func_args = [smp.Symbol(output_name)]
    if matrix_var_name:
        func_args.append(smp.Symbol(matrix_var_name))
    func_args.extend(other_syms)

    return "\n".join(lines), smp.Function(func_name)(*func_args)


def _perform_cse_and_inlining(
    triangle: list[smp.Expr],
) -> tuple[
    dict[smp.Symbol, smp.Expr],
    dict[smp.Symbol, smp.Expr],
    list[tuple[smp.Symbol, smp.Expr]],
]:
    """Perform CSE and compute inlining map.

    Args:
        triangle: List of expressions from triangle

    Returns:
        Tuple of (lookup dict, inline_map dict, keep list of (symbol, expr))
    """
    cse_result = smp.cse(triangle, symbols=smp.numbered_symbols("_t"))
    subs_all: list[tuple[smp.Symbol, smp.Expr]] = cse_result[0]  # type: ignore[assignment]
    reduced_triangle: list[smp.Expr] = cse_result[1]  # type: ignore[assignment]

    # Count usage frequency for inlining
    lookup = {t: expr for t, expr in subs_all}
    usage = {t: 0 for t, _ in subs_all}

    all_exprs = [e for _, e in subs_all] + reduced_triangle
    for expr in all_exprs:
        for t in list(usage):
            if t in expr.free_symbols:
                usage[t] += 1

    # Identify temporaries used only once (candidates for inlining)
    inline = {t for t, cnt in usage.items() if cnt == 1}
    keep = [(t, e) for t, e in subs_all if t not in inline]
    inline_map = {t: lookup[t] for t in inline}

    return lookup, inline_map, keep


def generate_hamiltonian_code(hamiltonian: smp.Matrix) -> tuple[str, smp.Function]:
    """Generate Julia code to fill a Hamiltonian matrix.

    Args:
        hamiltonian: SymPy matrix representing the Hamiltonian

    Returns:
        Julia code string for the Hamiltonian-filling function
    """
    code, sig = sympy_matrix_to_julia_fill_hermitian(
        hamiltonian,
        func_name="hamiltonian!",
        zero_input=True,
        output_name="du",
        uplo="U",
        mirror=True,
    )
    return code, sig


def generate_dissipator_code(dissipator: smp.Matrix) -> tuple[str, smp.Function]:
    """Generate Julia code to fill a Dissipator matrix.

    Args:
        dissipator: SymPy matrix representing the Dissipator

    Returns:
        Julia code string for the Dissipator-filling function
    """
    code, sig = sympy_matrix_to_julia_add_hermitian(
        dissipator,
        func_name="dissipator!",
        output_name="du",
        uplo="U",
        mirror=True,
    )
    return code, sig
