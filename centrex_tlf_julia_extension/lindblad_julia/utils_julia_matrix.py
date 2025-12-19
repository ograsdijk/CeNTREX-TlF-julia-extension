import sympy as smp
from sympy.parsing import sympy_parser

from .ode_parameters import odeParameters

liouville_commutator_functions = [
    "liouvillian_commutator!",
    "liouvillian_commutator_her2k!",
]

# Mapping of commutator function names to their number of arguments
commutator_nargs: dict[str, int] = {
    "liouvillian_commutator!": 3,  # (C, A, B)
    "liouvillian_commutator_her2k!": 3,  # (C, A, B)
}


def hamiltonian_functor(hamiltonian_signature: smp.Function) -> str:
    """Generate Julia code for a Hamiltonian functor struct.

    Creates a Julia callable struct that wraps the Hamiltonian function,
    storing parameters as struct fields for efficient repeated evaluation.
    The functor is compatible with the ODE solver interface (du, t) signature.

    Args:
        hamiltonian_signature: SymPy function representing the Hamiltonian
            signature. Should have arguments including 'du' (output), 't' (time),
            and any parameters.

    Returns:
        Julia code string defining:
            - HamFunctor{T} struct with parameter fields
            - Callable (h::HamFunctor)(du, t) method

    Example:
        >>> # For hamiltonian!(du, Omega, Delta, t)
        >>> sig = smp.Function('hamiltonian!')(du, Omega, Delta, t)
        >>> code = ham_functor(sig)
        >>> print(code)
        struct HamFunctor{T}
            Omega::T
            Delta::T
        end

        @inline function (h::HamFunctor)(du, t)
            hamiltonian!(du, h.Omega, h.Delta, t)
            return nothing
        end
    """
    args = hamiltonian_signature.args
    # Extract parameter names (excluding du and t which are function arguments)
    args_struct = "\n".join(
        [f"    {arg.name}::T" for arg in args if arg.name not in ["du", "t"]]
    )
    # Build function call with struct fields (h.param) for parameters
    args_func = ", ".join(
        [f"h.{arg.name}" if arg.name not in ["du", "t"] else arg.name for arg in args]
    )
    ham_functor_code = f"""
struct HamFunctor{{T}}
{args_struct}
end

@inline function (h::HamFunctor)(du, t)
    hamiltonian!({args_func})
    return nothing
end
"""
    return ham_functor_code


def lindblad_function_and_parameters(commutator_name: str) -> str:
    """Generate Julia code for Lindblad equation struct and function.

    Creates Julia code that defines:
    1. LindbladParameters struct to hold function pointers and buffer matrices
    2. lindblad! function that implements the Lindblad master equation using
       the specified commutator method

    The Lindblad equation is: dρ/dt = -i[H, ρ] + D(ρ)
    where H is the Hamiltonian, ρ is the density matrix, and D is the dissipator.

    Args:
        commutator_name: Name of the commutator function to use.
            Must be one of: 'liouvillian_commutator!',
            'liouvillian_commutator_onegemm!', 'liouvillian_commutator_her2k!'

    Returns:
        Julia code string defining:
            - LindbladParameters{HamFunc, DissFunc, T} struct
            - lindblad!(du, u, p, t) function compatible with ODE solvers

    Raises:
        AssertionError: If commutator_name is not a valid commutator function

    Example:
        >>> code = lindblad_function_and_parameters('liouvillian_commutator!')
        >>> print(code)
        struct LindbladParameters{HamFunc, DissFunc, T<:AbstractArray}
            hamiltonian!::HamFunc
            dissipator!::DissFunc
            buffer0::T
        end

        function lindblad!(du, u, p::LindbladParameters, t)
            p.hamiltonian!(p.buffer0, t)
            liouvillian_commutator!(du, p.buffer0, u)
            p.dissipator!(du, u)
            nothing
        end
    """
    assert commutator_name in liouville_commutator_functions, (
        f"Invalid commutator '{commutator_name}'. "
        f"Must be one of: {liouville_commutator_functions}"
    )

    # Calculate number of buffer matrices needed
    # Each commutator needs (nargs - 2) buffers since du and u are always passed
    nargs = commutator_nargs[commutator_name]
    nbuffers = nargs - 2

    # Generate buffer field declarations for the struct
    args_buffer = "\n".join([f"    buffer{i}::T" for i in range(nbuffers)])

    # Build commutator function call arguments
    args_commutator = "du, p.buffer0, u"
    if nbuffers > 1:
        # Add additional buffers if needed (e.g., for onegemm which needs extra buffer)
        args_commutator += ", " + ", ".join(
            [f"p.buffer{i}" for i in range(1, nbuffers)]
        )

    lindblad_function_par_code = f"""
struct LindbladParameters{{HamFunc, DissFunc, T<:AbstractArray}}
    hamiltonian!::HamFunc
    dissipator!::DissFunc
{args_buffer}
end

function Lindblad_rhs!(du, u, p::LindbladParameters, t)
    p.hamiltonian!(p.buffer0, t)
    {commutator_name}({args_commutator})
    p.dissipator!(du, u)
    nothing
end
"""
    return lindblad_function_par_code


def dissipator_functor() -> str:
    """Generate Julia code for a dissipator functor struct.

    Creates a simple callable struct that wraps the dissipator! function.
    This provides a consistent interface for the Lindblad parameters struct.

    Returns:
        Julia code string defining:
            - DissFunctor struct (empty, used only for dispatch)
            - Callable (d::DissFunctor)(du, u) method

    Example:
        >>> code = dissipator_functor()
        >>> print(code)
        struct DissFunctor end

        @inline function (d::DissFunctor)(du, u)
            dissipator!(du, u)
            return nothing
        end
    """
    return """
struct DissFunctor end

@inline function (d::DissFunctor)(du, u)
    dissipator!(du, u)
    return nothing
end
"""


def generate_code_matrix_method(
    commutator_name: str,
    ham_signature: smp.Function,
) -> str:
    """Generate complete Julia code for matrix-based Lindblad equation.

    Generates all the Julia structs and functions needed to solve the Lindblad
    master equation using a specific commutator optimization method. This includes:

    1. HamFunctor - Callable struct for Hamiltonian evaluation
    2. DissFunctor - Callable struct for dissipator evaluation
    3. LindbladParameters - Struct holding functions and buffers
    4. lindblad! - Main function implementing the Lindblad equation

    Args:
        commutator_name: Name of the commutator function to use.
            Options: 'liouvillian_commutator!', 'liouvillian_commutator_onegemm!',
            'liouvillian_commutator_her2k!'
        ham_signature: SymPy function representing the Hamiltonian signature,
            including all parameters (e.g., Omega, Delta, etc.)

    Returns:
        Complete Julia code string with all necessary struct and function
        definitions for solving the Lindblad equation.

    Example:
        >>> import sympy as smp
        >>> du, Omega, Delta, t = smp.symbols('du Omega Delta t')
        >>> ham_sig = smp.Function('hamiltonian!')(du, Omega, Delta, t)
        >>> code = generate_code_matrix_method('liouvillian_commutator!', ham_sig)
        >>> # Returns complete Julia code with HamFunctor, DissFunctor,
        >>> # LindbladParameters, and lindblad! function
    """
    return (
        hamiltonian_functor(ham_signature)
        + "\n"
        + dissipator_functor()
        + "\n"
        + lindblad_function_and_parameters(commutator_name)
        + "\n"
    )


def parse_compound_vars_odepars(odepars: odeParameters) -> dict[str, smp.Expr]:
    expressions = []
    for par in odepars._compound_vars:
        sympy_expr = sympy_parser.parse_expr(getattr(odepars, par))
        expressions.append((par, sympy_expr))
    return dict(expressions[::-1])


def substitute_odepars_hamiltonian(
    hamiltonian: smp.Matrix,
    odepars: odeParameters,
) -> smp.Matrix:
    hamiltonian = hamiltonian.copy()

    parsed_compound_vars = parse_compound_vars_odepars(odepars)
    free_symbols: list[smp.Symbol] = list(hamiltonian.free_symbols)
    free_symbols_str = [s.name for s in free_symbols]

    for par, expr in parsed_compound_vars.items():
        if par in free_symbols_str:
            sym = free_symbols[free_symbols_str.index(par)]
            hamiltonian = hamiltonian.subs(sym, expr)
        else:
            sym = smp.Symbol(par)
            hamiltonian = hamiltonian.subs(sym, expr)
    return hamiltonian
