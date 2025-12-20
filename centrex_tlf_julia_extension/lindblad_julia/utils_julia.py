from pathlib import Path
from typing import List

import juliacall

jl = juliacall.Main  # type: ignore[attr-defined]

__all__ = ["initialize_julia", "generate_ode_fun_julia"]

# jl = juliacall.newmodule("centrex-tlf-julia-extension")

julia_dependency_packages = [
    "TerminalLoggers",
    "ProgressMeter",
    "Waveforms",
    "Trapz",
    "DifferentialEquations",
]


def install_packages() -> None:
    jl.seval("using Pkg")
    for pkg in julia_dependency_packages:
        if not bool(jl.seval(f'isnothing(Base.find_package("{pkg}")) ? false : true')):
            print(f"Installing Julia package: {pkg}")
            jl.Pkg.add(pkg)


def initialize_julia(nprocs: int, blas_threads: int = 1, verbose: bool = True) -> None:
    """
    Function to initialize Julia over nprocs processes.
    Creates nprocs processes and loads the necessary Julia
    packages.

    Args:
        nprocs (int): number of Julia processes to initialize.
    """
    install_packages()
    jl.seval(
        """
        using Logging: global_logger
        using TerminalLoggers: TerminalLogger
        global_logger(TerminalLogger())

        using Distributed
        using ProgressMeter
    """
    )

    if jl.seval("nprocs()") < nprocs:
        jl.seval(f"addprocs({nprocs}-nprocs())")

    if jl.seval("nprocs()") > nprocs:
        procs = jl.seval("procs()")
        procs = procs[nprocs:]
        jl.seval(f"rmprocs({procs})")

    jl.seval(
        f"""
        @everywhere begin
            using LinearAlgebra
            using LinearAlgebra.BLAS
            using Trapz
            using DifferentialEquations
            using Waveforms
            LinearAlgebra.BLAS.set_num_threads({blas_threads})
        end
    """
    )
    # loading common julia functions from julia_common.jl
    path = Path(__file__).parent / "julia_common.jl"
    jl.seval(f'include(raw"{path}")')

    if verbose:
        print(f"Initialized Julia with {nprocs} processes")


def generate_ode_fun_julia(preamble: str, code_lines: List[str]) -> str:
    """
    Generate the ODE function from the preamble and code lines
    generated in Python.

    Args:
        preamble (str): preamble of the ODE function initializing the
                        function variable definitions.
        code_lines (list): list of strings, each line is a generated
                            line of Julia code for part of the ODE.

    Returns:
        str : function definition of the ODE
    """
    ode_fun = preamble
    for cline in code_lines:
        ode_fun += "\t\t" + cline + "\n"
    ode_fun += "\t end \n \t nothing \n end"
    jl.seval(f"@everywhere {ode_fun}")
    return ode_fun
