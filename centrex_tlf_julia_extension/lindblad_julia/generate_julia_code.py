from collections import OrderedDict
from typing import List, Sequence

import sympy as smp
from centrex_tlf import couplings
from sympy import MutableDenseMatrix
from sympy.parsing import sympy_parser
from sympy.printing.julia import julia_code

from .ode_parameters import odeParameters

__all__ = ["system_of_equations_to_lines", "generate_preamble"]


def generate_preamble(
    odepars: odeParameters, transition_selectors: Sequence[couplings.TransitionSelector]
) -> str:
    odepars.check_transition_symbols(transition_selectors)

    preamble = """function Lindblad_rhs!(du, ρ, p, t)
\t@inbounds begin
"""

    # Bind parameters from p with explicit types
    for idp, (par, par_type) in enumerate(
        zip(odepars._parameters, odepars._parameter_types), start=1
    ):
        preamble += f"\t\t{par}::{par_type} = p[{idp}]\n"

    # Compound vars
    for par in odepars._compound_vars:
        sympy_expr = sympy_parser.parse_expr(getattr(odepars, par))
        julia_expr = julia_code(sympy_expr, strict=False)
        clean = "\n".join(
            line for line in julia_expr.splitlines() if not line.strip().startswith("#")
        )
        preamble += f"\t\t{par} = {clean}\n"

    # Remove duplicate lines (same as before)
    preamble = "\n".join(list(OrderedDict.fromkeys(preamble.split("\n"))))

    # If you still need to force Ω symbols to ComplexF64, do it directly
    # (only if those Ω variables appear as bare identifiers in the generated lines)
    for transition in transition_selectors:
        preamble = preamble.replace(f"{transition.Ω} ", f"{transition.Ω}::ComplexF64 ")

    return preamble


def system_of_equations_to_lines(
    system: MutableDenseMatrix,
    transition_selectors: Sequence[couplings.TransitionSelector],
) -> List[str]:
    n_states = system.shape[0]
    rho = smp.IndexedBase("\u03c1")
    ρ = smp.Matrix(n_states, n_states, lambda i, j: rho[i, j])

    cse_temps, [system_opt] = smp.cse(system, optimizations="basic")

    code_lines: list[str] = []

    for val, temp in cse_temps:
        cline = str(temp)
        cline = cline.replace("conjugate", "conj")
        cline = cline.replace("(t)", "")
        cline = cline.replace("I", "1im")
        cline += "\n"
        for i in range(system.shape[0]):
            for j in range(system.shape[1]):
                _ = str(ρ[i, j])
                cline = cline.replace(_ + "*", f"ρ[{i + 1},{j + 1}]*")
                cline = cline.replace(_ + " ", f"ρ[{i + 1},{j + 1}] ")
                cline = cline.replace(_ + "\n", f"ρ[{i + 1},{j + 1}]")
                cline = cline.replace(_ + ")", f"ρ[{i + 1},{j + 1}])")
        cline = cline.strip()
        # replace ρ[i,j] with conj(ρ[j,i])
        for i in range(n_states):
            for j in range(0, i):
                cline = cline.replace(
                    f"ρ[{i + 1},{j + 1}]", f"conj(ρ[{j + 1},{i + 1}])"
                )

        code_lines.append(f"{val} = {cline}")

    # only calculating the upper triangle and diagonal
    for idx in range(n_states):
        for idy in range(idx, n_states):
            if system_opt[idx, idy] != 0:
                cline = str(system_opt[idx, idy])
                cline = cline.replace("conjugate", "conj")
                cline = f"du[{idx + 1},{idy + 1}] = " + cline
                cline = cline.replace("(t)", "")
                cline = cline.replace("I", "1im")

                # replace pol*rabi symbols
                # for repl in pol_rabi_replacements:
                #     cline = cline.replace(*repl)
                cline += "\n"
                for i in range(system.shape[0]):
                    for j in range(system.shape[1]):
                        _ = str(ρ[i, j])
                        cline = cline.replace(_ + "*", f"ρ[{i + 1},{j + 1}]*")
                        cline = cline.replace(_ + " ", f"ρ[{i + 1},{j + 1}] ")
                        cline = cline.replace(_ + "\n", f"ρ[{i + 1},{j + 1}]")
                        cline = cline.replace(_ + ")", f"ρ[{i + 1},{j + 1}])")
                cline = cline.strip()
                # replace ρ[i,j] with conj(ρ[j,i])
                for i in range(n_states):
                    for j in range(0, i):
                        cline = cline.replace(
                            f"ρ[{i + 1},{j + 1}]", f"conj(ρ[{j + 1},{i + 1}])"
                        )

                code_lines.append(cline)
    return code_lines
