from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, cast

import numpy as np
import numpy.typing as npt
import psutil
import sympy as smp
from centrex_tlf import couplings, hamiltonian, states
from centrex_tlf.couplings import CouplingFields
from centrex_tlf.lindblad import OBESystem, utils_decay
from sympy import MutableDenseMatrix

from .generate_julia_code import generate_preamble, system_of_equations_to_lines
from .ode_parameters import odeParameters
from .utils_julia import generate_ode_fun_julia, initialize_julia, jl
from .utils_julia_matrix import (
    dissipator_functor,
    hamiltonian_functor,
    lindblad_function_and_parameters,
    substitute_odepars_hamiltonian,
)
from .utils_julia_matrix_assemble import (
    generate_dissipator_code,
    generate_hamiltonian_code,
)

__all__ = ["OBESystemJulia", "generate_OBE_system_julia", "setup_OBE_system_julia"]


@dataclass
class CodeExpanded:
    preamble: str
    code_lines: List[str]


@dataclass
class CodeMatrix:
    hamiltonian: str
    dissipator: str
    lindblad: str
    support: str
    hamiltonian_signature: smp.Function
    dissipator_signature: smp.Function


@dataclass
class OBESystemJulia:
    ground: Sequence[states.State]
    excited: Sequence[states.State]
    QN: Sequence[states.State]
    H_int: npt.NDArray[np.complex128]
    V_ref_int: npt.NDArray[np.complex128]
    couplings: List[Any]
    H_symbolic: MutableDenseMatrix
    dissipator: MutableDenseMatrix
    C_array: npt.NDArray[np.floating]
    system: MutableDenseMatrix | None
    code: CodeExpanded | CodeMatrix
    full_output: bool = False
    QN_original: Optional[Sequence[states.State]] = None
    decay_channels: Optional[Sequence[utils_decay.DecayChannel]] = None
    couplings_original: Optional[Sequence[CouplingFields]] = None

    def __repr__(self) -> str:
        ground = [s.largest for s in self.ground]
        ground = list(
            np.unique(
                [
                    f"|{s.electronic_state.name}, J = {s.J}, "  # type: ignore
                    f"P = {'+' if s.P == 1 else '-'}>"  # type: ignore
                    for s in ground
                ]
            )
        )
        ground_str: str = ", ".join(ground)  # type: ignore
        excited = [s.largest for s in self.excited]
        excited = list(
            np.unique(
                [
                    str(
                        f"|{s.electronic_state.name}, J = {s.J}, "  # type: ignore
                        f"F₁ = {smp.S(str(s.F1), rational=True)}, "  # type: ignore
                        f"F = {s.F}, "  # type: ignore
                        f"P = {'+' if s.P == 1 else '-'}>"  # type: ignore
                    )
                    for s in excited
                ]
            )
        )
        excited_str: str = ", ".join(excited)  # type: ignore
        return f"OBESystem(ground=[{ground_str}], excited=[{excited_str}])"


def generate_OBE_system_julia(
    obe_system: OBESystem,
    transition_selectors: Sequence[couplings.TransitionSelector],
    ode_parameters: odeParameters,
    method: str,
) -> OBESystemJulia:
    if obe_system.dissipator is None:
        raise ValueError("obe_system.dissipator is None, cannot generate code.")

    if method == "expanded":
        preamble = generate_preamble(ode_parameters, transition_selectors)
        code_lines = system_of_equations_to_lines(
            obe_system.system, transition_selectors
        )
        if obe_system.system is None:
            raise ValueError(
                "obe_system.system is None, cannot generate expanded code."
            )
        return OBESystemJulia(
            QN=obe_system.QN,
            ground=obe_system.ground,
            excited=obe_system.excited,
            couplings=obe_system.couplings,
            H_symbolic=obe_system.H_symbolic,
            dissipator=obe_system.dissipator,
            H_int=obe_system.H_int,
            V_ref_int=obe_system.V_ref_int,
            C_array=obe_system.C_array,
            system=obe_system.system,
            code=CodeExpanded(preamble=preamble, code_lines=code_lines),
            QN_original=obe_system.QN_original,
            decay_channels=obe_system.decay_channels,
            couplings_original=obe_system.couplings_original,
        )
    elif method == "matrix":
        hamiltonian_subbed = substitute_odepars_hamiltonian(
            obe_system.H_symbolic, ode_parameters
        )
        hamiltonian_code, hamiltonian_signature = generate_hamiltonian_code(
            hamiltonian_subbed
        )
        dissipator_code, dissipator_signature = generate_dissipator_code(
            obe_system.dissipator
        )
        lindblad = lindblad_function_and_parameters("liouvillian_commutator_her2k!")

        ham_functor_code = hamiltonian_functor(hamiltonian_signature, ode_parameters)
        diss_functor_code = dissipator_functor()

        # reorder ode parameters to match Hamiltonian signature
        new_order = [
            str(v) for v in hamiltonian_signature.args if str(v) not in ["du", "t"]
        ]
        ode_parameters.reorder(new_order)
        ode_parameters._method = "matrix"

        other_code = "DissFun = DissFunctor()\n"

        nstates = obe_system.H_symbolic.shape[0]
        other_code += f"nstates = {nstates}\n"
        other_code += "buf = zeros(ComplexF64, nstates, nstates)\n"

        return OBESystemJulia(
            QN=obe_system.QN,
            ground=obe_system.ground,
            excited=obe_system.excited,
            couplings=obe_system.couplings,
            H_symbolic=obe_system.H_symbolic,
            dissipator=obe_system.dissipator,
            H_int=obe_system.H_int,
            V_ref_int=obe_system.V_ref_int,
            C_array=obe_system.C_array,
            system=obe_system.system,
            code=CodeMatrix(
                hamiltonian_code,
                dissipator_code,
                lindblad,
                ham_functor_code + diss_functor_code + other_code,
                hamiltonian_signature,
                dissipator_signature,
            ),
            QN_original=obe_system.QN_original,
            decay_channels=obe_system.decay_channels,
            couplings_original=obe_system.couplings_original,
        )
    else:
        raise ValueError(f"Unknown method '{method}' for generating ODE system.")


def setup_OBE_system_julia(
    obe_system: OBESystem,
    transition_selectors: Sequence[couplings.TransitionSelector],
    ode_parameters: odeParameters,
    n_procs: None | int = None,
    method: str = "expanded",
    Γ: float = hamiltonian.Γ,
    verbose: bool = False,
) -> OBESystemJulia:
    if n_procs is None:
        core_count = psutil.cpu_count(logical=False)
        if core_count is None:
            raise RuntimeError("Could not determine number of CPU cores.")
        n_procs = cast(int, core_count + 1)
    if verbose:
        print(f"setup_OBE_system_julia: 1/3 -> Initializing Julia on {n_procs} cores")
    initialize_julia(nprocs=n_procs, verbose=verbose)
    if verbose:
        print("setup_OBE_system_julia: 2/3 -> generating OBESystemJulia")
    obe_system_julia = generate_OBE_system_julia(
        obe_system, transition_selectors, ode_parameters, method=method
    )
    if verbose:
        print(
            "setup_OBE_system_julia: 3/3 -> Defining the ODE equation and"
            " parameters in Julia"
        )
    if isinstance(obe_system_julia.code, CodeExpanded):
        generate_ode_fun_julia(
            obe_system_julia.code.preamble, obe_system_julia.code.code_lines
        )
    elif isinstance(obe_system_julia.code, CodeMatrix):
        jl.seval(
            f"@everywhere begin\n{obe_system_julia.code.hamiltonian}\n{obe_system_julia.code.dissipator}\n{obe_system_julia.code.lindblad}\n{obe_system_julia.code.support}\nend"
        )
        # raise NotImplementedError("Matrix method not yet implemented in setup.")
    jl.seval(f"@everywhere Γ = {Γ}")
    ode_parameters.generate_p_julia()
    return obe_system_julia
