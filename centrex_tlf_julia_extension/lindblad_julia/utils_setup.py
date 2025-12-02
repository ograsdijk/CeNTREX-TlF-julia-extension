from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, cast

import numpy as np
import numpy.typing as npt
import psutil
import sympy as smp
from centrex_tlf import couplings, hamiltonian, states
from centrex_tlf.couplings import CouplingFields
from centrex_tlf.lindblad import OBESystem, utils_decay

from .generate_julia_code import generate_preamble, system_of_equations_to_lines
from .ode_parameters import odeParameters
from .utils_julia import generate_ode_fun_julia, initialize_julia, jl

__all__ = ["OBESystemJulia", "generate_OBE_system_julia", "setup_OBE_system_julia"]


@dataclass
class OBESystemJulia:
    ground: Sequence[states.State]
    excited: Sequence[states.State]
    QN: Sequence[states.State]
    H_int: npt.NDArray[np.complex128]
    V_ref_int: npt.NDArray[np.complex128]
    couplings: List[Any]
    H_symbolic: smp.matrices.dense.MutableDenseMatrix
    C_array: npt.NDArray[np.floating]
    system: smp.matrices.dense.MutableDenseMatrix
    code_lines: List[str]
    full_output: bool = False
    preamble: str = ""
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
) -> OBESystemJulia:
    preamble = generate_preamble(ode_parameters, transition_selectors)
    code_lines = system_of_equations_to_lines(obe_system.system, transition_selectors)

    return OBESystemJulia(
        QN=obe_system.QN,
        ground=obe_system.ground,
        excited=obe_system.excited,
        couplings=obe_system.couplings,
        H_symbolic=obe_system.H_symbolic,
        H_int=obe_system.H_int,
        V_ref_int=obe_system.V_ref_int,
        C_array=obe_system.C_array,
        system=obe_system.system,
        code_lines=code_lines,
        preamble=preamble,
        QN_original=obe_system.QN_original,
        decay_channels=obe_system.decay_channels,
        couplings_original=obe_system.couplings_original,
    )


def setup_OBE_system_julia(
    obe_system: OBESystem,
    transition_selectors: Sequence[couplings.TransitionSelector],
    ode_parameters: odeParameters,
    n_procs: Optional[int] = None,
    Γ: float = hamiltonian.Γ,
    verbose: bool = False,
) -> OBESystemJulia:
    if n_procs is None:
        n_procs = cast(int, psutil.cpu_count(logical=False) + 1)
    if verbose:
        print(f"setup_OBE_system_julia: 1/3 -> Initializing Julia on {n_procs} cores")
    initialize_julia(nprocs=n_procs, verbose=verbose)
    if verbose:
        print("setup_OBE_system_julia: 2/3 -> generating OBESystemJulia")
    obe_system_julia = generate_OBE_system_julia(
        obe_system, transition_selectors, ode_parameters
    )
    if verbose:
        print(
            "setup_OBE_system_julia: 3/3 -> Defining the ODE equation and"
            " parameters in Julia"
        )
    generate_ode_fun_julia(obe_system_julia.preamble, obe_system_julia.code_lines)
    jl.seval(f"@everywhere Γ = {Γ}")
    ode_parameters.generate_p_julia()
    return obe_system_julia
