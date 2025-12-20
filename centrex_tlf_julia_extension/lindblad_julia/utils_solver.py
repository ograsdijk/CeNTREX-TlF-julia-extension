from dataclasses import dataclass, field
from numbers import Number
from typing import Sequence

import numpy as np
import numpy.typing as npt
from sympy.parsing import sympy_parser

from .ode_parameters import julia_literal, odeParameters
from .utils_julia import jl

numeric = int | float | complex

__all__ = [
    "get_diagonal_indices_flattened",
    "setup_initial_condition_scan",
    "setup_ratio_calculation_state_idxs",
    "setup_ratio_calculation",
    "setup_state_integral_calculation_state_idxs",
    "setup_state_integral_calculation",
    "setup_parameter_scan_ND",
    "setup_discrete_callback_terminate",
    "setup_problem",
    "solve_problem",
    "get_results_single",
    "do_simulation_single",
    "setup_problem_parameter_scan",
    "solve_problem_parameter_scan",
    "get_results_parameter_scan",
    "OBEProblem",
    "OBEEnsembleProblem",
    "OBEProblemConfig",
    "OBEEnsembleProblemConfig",
    "OBEResult",
    "OBEResultParameterScan",
]


@dataclass
class OBEProblem:
    odepars: odeParameters
    ρ: npt.NDArray[np.complex128]
    tspan: list[float] | tuple[float, ...]
    name: str = "prob"


@dataclass
class CallbackFunction:
    name: str
    function: str


@dataclass
class ProblemFunction:
    name: str
    function: str


@dataclass
class OutputFunction:
    name: str
    function: str


@dataclass
class OBEEnsembleProblem:
    problem: OBEProblem
    parameters: list[str]
    scan_values: list[npt.NDArray[np.number]]
    name: str = "ens_prob"
    output_func: None | OutputFunction = None
    zipped: bool = False
    method: str = "expanded"


@dataclass
class OBEProblemConfig:
    method: str = "Tsit5()"
    abstol: float = 1e-7
    reltol: float = 1e-4
    dt: float = 1e-8
    callback: None | CallbackFunction = None
    dtmin: float = 0
    maxiters: int = 100_000
    saveat: list[float] | npt.NDArray[np.floating] | float | int = field(
        default_factory=list
    )
    save_everystep: bool = True
    save_idxs: None | list[int] = None
    progress: bool = False
    dense: bool = False
    save_start: bool = True


@dataclass
class OBEEnsembleProblemConfig(OBEProblemConfig):
    distributed_method: str = "EnsembleDistributed()"
    trajectories: None | int = None


@dataclass
class OBEResult:
    t: npt.NDArray[np.float64]
    y: npt.NDArray[np.complex128]


@dataclass
class OBEResultParameterScan:
    parameters: list[str]
    scan_values: Sequence[npt.NDArray[np.generic]]
    results: npt.NDArray[np.complex128]
    zipped: bool


def remove_leading_spaces_to_align(multiline_string: str) -> str:
    lines = multiline_string.splitlines()

    # Find the minimum number of leading spaces in all non-empty lines
    min_spaces = min(
        len(line) - len(line.lstrip(" ")) for line in lines if line.strip()
    )

    # Remove exactly 'min_spaces' number of leading spaces from each line
    aligned_lines = [
        line[min_spaces:] if line.startswith(" " * min_spaces) else line
        for line in lines
    ]

    return "\n".join(aligned_lines)


def get_diagonal_indices_flattened(
    size: int, states: None | Sequence[int] = None, mode: str = "python"
) -> list[int]:
    if states is None:
        indices = [i + size * i for i in range(size)]
    else:
        indices = [i + size * i for i in states]
    if mode == "julia":
        return [i + 1 for i in indices]
    elif mode == "python":
        return indices
    else:
        raise ValueError(f"mode {mode} not supported, use 'julia' or 'python'")


def setup_initial_condition_scan(
    values: Sequence[Number] | npt.NDArray[np.generic],
    name: str = "prob_func",
) -> ProblemFunction:
    jl.params = values
    jl.params = jl.seval("collect(params)")
    jl.seval("@everywhere params = $params")
    function_str = f"""
    @everywhere function {name}(prob,i,repeat)
        remake(prob,u0=params[i])
    end"""
    jl.seval(function_str)
    function_str = remove_leading_spaces_to_align(function_str)
    return ProblemFunction(name=name, function=function_str)


def setup_parameter_scan_zipped(
    ode_parameters: odeParameters,
    parameters: str | Sequence[str | Sequence[str]],
    values: Sequence[npt.NDArray[np.generic]] | npt.NDArray[np.generic],
    name: str = "prob_func",
) -> ProblemFunction:
    """
    Convenience function for initializing a 1D parameter scan over
    multiple parameters, with each parameter scanning over a different
    set of parameters.
    """
    # Normalize parameters
    if isinstance(parameters, str):
        parameters_list = [parameters]
    else:
        parameters_list = list(parameters)

    params = np.array(list(zip(*values)))

    # Replace scanned parameters with params[i,j]
    pars = list(ode_parameters.p)
    for idN, parameter in enumerate(parameters_list):
        if isinstance(parameter, (list, tuple)):
            indices = [ode_parameters.get_index_parameter(par) for par in parameter]
        else:
            indices = [ode_parameters.get_index_parameter(parameter)]

        for idx in indices:
            assert isinstance(idx, int)
            pars[idx] = f"params[i,{idN + 1}]"

    # Build _pars EXACTLY like generate_p_julia, but allow strings to pass through
    elems = [pi if isinstance(pi, str) else julia_literal(pi) for pi in pars]
    _pars = f"({elems[0]},)" if len(elems) == 1 else "(" + ", ".join(elems) + ")"

    # Push params to Julia (no const, no renaming)
    jl.params = params
    jl.seval("params = collect(params)")
    jl.seval("@everywhere params = $params")

    # Generate prob_func
    if ode_parameters._method == "expanded":
        function_str = f"""
        @everywhere function {name}(prob, i, repeat)
            remake(prob, p = {_pars})
        end
        """
    elif ode_parameters._method == "matrix":
        function_str = f"""
        @everywhere function {name}(prob, i, repeat)
            p_values = {_pars}
            HamFun = HamFunctor(p_values...)
            p_new = LindbladParameters(HamFun, DissFun, buf)
            remake(prob, p = p_new)
        end
        """
    else:
        raise ValueError(f"method {ode_parameters._method} not supported")

    jl.seval(function_str)
    function_str = remove_leading_spaces_to_align(function_str)
    return ProblemFunction(name=name, function=function_str)


def setup_parameter_scan_ND(
    odePar: odeParameters,
    parameters: str | Sequence[str],
    values: Sequence[npt.NDArray[np.generic]]
    | npt.NDArray[np.generic]
    | Sequence[Number],
) -> ProblemFunction:
    """
    Convenience function for generating an ND parameter scan.
    For each parameter a list or np.ndarray of values is supplied,
    and each possible combination between all parameters is simulated.

    Args:
        odePar (odeParameters): object containing all the parameters for
                                the OBE system.
        parameters (str | List[str]): strs of parameters to scan over.
        values (Sequence[npt.NDArray[np.generic]] | npt.NDArray[np.generic]): list or np.ndarray of values to scan over
                                    for each parameter
    """
    # create all possible combinations between parameter values with meshgrid
    params = np.array(np.meshgrid(*values, indexing="ij")).T.reshape(-1, len(values))

    return setup_parameter_scan_zipped(odePar, parameters, params.T)


def setup_ratio_calculation_state_idxs(
    states: None | Sequence[int] = None,
    output_func: None | str = None,
) -> OutputFunction:
    if output_func is None:
        output_func = "output_func"
    if states is None:
        cmd = "real(sum(sol.u[end]))/real(sum(sol.u[1]))"
    else:
        cmd = f"real(sum(sol.u[end][{states}]))/real(sum(sol.u[1][{states}]))"

    function_str = f"""
    @everywhere function {output_func}(sol,i)
        if size(sol.u)[1] == 1
            return NaN, false
        else
            val = {cmd}
            return val, false
        end
    end"""

    function_str = remove_leading_spaces_to_align(function_str)
    jl.seval(function_str)
    return OutputFunction(name=output_func, function=function_str)


def setup_ratio_calculation(
    states: Sequence[int] | Sequence[Sequence[int]],
    output_func: None | str = None,
) -> OutputFunction:
    if output_func is None:
        output_func = "output_func"
    cmd = ""
    if isinstance(states[0], (list, np.ndarray, tuple)):
        for state in states:
            cmd += (
                f"sum(real(diag(sol.u[end])[{state}]))/"
                f"sum(real(diag(sol.u[1])[{state}])), "
            )
        cmd = cmd.strip(", ")
        cmd = "[" + cmd + "]"
    else:
        cmd = (
            f"sum(real(diag(sol.u[end])[{states}]))/sum(real(diag(sol.u[1])[{states}]))"
        )

    function_str = f"""
    @everywhere function {output_func}(sol,i)
        if size(sol.u)[1] == 1
            return NaN, false
        else
            val = {cmd}
            return val, false
        end
    end"""
    function_str = remove_leading_spaces_to_align(function_str)
    jl.seval(function_str)
    return OutputFunction(name=output_func, function=function_str)


def setup_state_integral_calculation_state_idxs(
    output_func: None | str = None, nphotons: bool = False, Γ: None | float = None
) -> OutputFunction:
    """Setup an integration output_function for an EnsembleProblem.
    Uses trapezoidal integration to integrate the states.

    Args:
        output_func (None | str, optional): name of the output function
        nphotons (bool): flag to calculate the number of photons,
                                    e.g. normalize with Γ
        Γ (float, optional): decay rate in 2π Hz (rad/s), not necessary if already
                                loaded into Julia globals
    """
    if output_func is None:
        output_func = "output_func"
    else:
        output_func = output_func

    if nphotons & jl.seval("@isdefined Γ"):
        function_str = f"""
        @everywhere function {output_func}(sol,i)
            return Γ.*trapz(sol.t, [real(sum(sol.u[j])) for j in 1:size(sol)[2]]), false
        end"""
        jl.seval(function_str)
    else:
        if nphotons:
            assert Γ is not None, (
                "Γ not defined as a global in Julia and not supplied to function"
            )
            jl.seval(f"@everywhere Γ = {Γ}")
            function_str = f"""
            @everywhere function {output_func}(sol,i)
                return {Γ}.*trapz(sol.t, [real(sum(sol.u[j])) for j in 1:size(sol)[2]]), false
            end"""
            jl.seval(function_str)
        else:
            function_str = f"""
            @everywhere function {output_func}(sol,i)
                return trapz(sol.t, [real(sum(sol.u[j])) for j in 1:size(sol)[2]]), false
            end"""
            jl.seval(function_str)

    function_str = remove_leading_spaces_to_align(function_str)
    return OutputFunction(name=output_func, function=function_str)


def setup_state_integral_calculation(
    states: Sequence[int],
    output_func: None | str = None,
    nphotons: bool = False,
    Γ: None | float = None,
) -> OutputFunction:
    """Setup an integration output_function for an EnsembleProblem.
    Uses trapezoidal integration to integrate the states.

    Args:
        states (list): list of state indices to integrate
        nphotons (bool, optional): flag to calculate the number of photons,
                                    e.g. normalize with Γ
        Γ (float, optional): decay rate in 2π Hz (rad/s), not necessary if already
                                loaded into Julia globals
    """
    if output_func is None:
        output_func = "output_func"
    else:
        output_func = output_func
    if nphotons & jl.seval("@isdefined Γ"):
        function_str = f"""
        @everywhere function {output_func}(sol,i)
            return Γ.*trapz(sol.t, [real(sum(diag(sol.u[j])[{states}])) for j in 1:size(sol)[3]]), false
        end"""
        jl.seval(function_str)
    else:
        if nphotons:
            assert Γ is not None, (
                "Γ not defined as a global in Julia and not supplied to function"
            )
            jl.seval(f"@everywhere Γ = {Γ}")
            function_str = f"""
            @everywhere function {output_func}(sol,i)
                return {Γ}.*trapz(sol.t, [real(sum(diag(sol.u[j])[{states}])) for j in 1:size(sol)[3]]), false
            end"""
            jl.seval(function_str)
        else:
            function_str = f"""
            @everywhere function {output_func}(sol,i)
                return trapz(sol.t, [real(sum(diag(sol.u[j])[{states}])) for j in 1:size(sol)[3]]), false
            end"""
            jl.seval(function_str)

    function_str = remove_leading_spaces_to_align(function_str)
    return OutputFunction(name=output_func, function=function_str)


def setup_state_integral_callback(
    states: Sequence[int],
    problem_wrap_name: str = "wrap_prob_func",
    output_func: str = "output_func",
) -> tuple[CallbackFunction, ProblemFunction, OutputFunction]:
    # Embed the tuple literal directly into the Julia code (no globals, no consts).
    states_tuple = tuple(states)

    callback_str = f"""
    @everywhere function nphotons_integrator(u, t, integrator)
        s = 0.0
        @inbounds for j in {states_tuple}
            s += real(u[j, j])
        end
        return s
    end
    """

    prob_func_str = f"""
    @everywhere function {problem_wrap_name}(prob_func_old)
        function prob_func_new(prob, i, repeat)
            prob2 = prob_func_old(prob, i, repeat)

            integ = IntegrandValuesSum(0.0)  # per-trajectory accumulator

            cb = IntegratingSumCallback(nphotons_integrator, integ, 0.0)

            oldcb = prob2.callback
            newcb = oldcb === nothing ? cb : CallbackSet(oldcb, cb)

            # Append integ to p so it can be retrieved in output_func without globals.
            pnew = (prob2.p..., integ)

            return remake(prob2; p = pnew, callback = newcb)
        end
        return prob_func_new
    end
    """

    function_str = f"""
    @everywhere function {output_func}(sol, i)
        return sol.prob.p[end].integrand, false
    end
    """

    jl.seval(callback_str)
    jl.seval(prob_func_str)
    jl.seval(function_str)

    return (
        CallbackFunction(name="cb", function=callback_str),
        ProblemFunction(name=problem_wrap_name, function=prob_func_str),
        OutputFunction(name=output_func, function=function_str),
    )


def setup_discrete_callback_terminate(
    odepars: odeParameters, stop_expression: str, callback_name: None | str = None
) -> CallbackFunction:
    # parse expression string to sympy equation
    expression = sympy_parser.parse_expr(stop_expression)
    # extract symbols in expression and convert to a list of strings
    symbols_in_expression = list(expression.free_symbols)
    symbols_in_expression = [str(sym) for sym in symbols_in_expression]
    # check if all symbols are parameters of the ODE
    odepars.check_symbols_in_parameters(symbols_in_expression)

    # remove t
    symbols_in_expression.remove("t")
    # get indices of symbols
    indices = [
        odepars.get_index_parameter(sym, mode="julia") for sym in symbols_in_expression
    ]
    for idx, sym in zip(indices, symbols_in_expression):
        stop_expression = stop_expression.replace(str(sym), f"integrator.p[{idx}]")
    if callback_name is None:
        callback_name = "cb"
    else:
        callback_name = callback_name
    function_str = f"""
        @everywhere condition(u,t,integrator) = {stop_expression}
        @everywhere affect!(integrator) = terminate!(integrator)
        {callback_name} = DiscreteCallback(condition, affect!)
    """
    function_str = remove_leading_spaces_to_align(function_str)
    jl.seval(function_str)
    return CallbackFunction(name=callback_name, function=function_str)


def setup_problem(
    odepars: odeParameters,
    tspan: list[float] | tuple[float, ...],
    ρ: npt.NDArray[np.complex128],
    problem_name: str = "prob",
) -> None:
    odepars.generate_p_julia()

    jl.ρ = ρ
    jl.seval("ρ = collect(ρ)")

    jl.tspan = tspan
    assert jl.seval("@isdefined Lindblad_rhs!"), (
        "Lindblad function is not defined in Julia"
    )
    jl.seval(
        f"""
        {problem_name} = ODEProblem(Lindblad_rhs!,ρ,tspan,p)
    """
    )


def setup_problem_parameter_scan(scan: OBEEnsembleProblem) -> ProblemFunction:
    odepars = scan.problem.odepars
    tspan = scan.problem.tspan
    ρ = scan.problem.ρ
    problem_name = scan.problem.name
    zipped = scan.zipped
    parameters = scan.parameters
    values = scan.scan_values

    setup_problem(odepars, tspan, ρ, problem_name)
    if zipped:
        prob_func = setup_parameter_scan_zipped(odepars, parameters, values)
    else:
        prob_func = setup_parameter_scan_ND(odepars, parameters, values)

    if scan.output_func is not None:
        jl.seval(
            f"""
            ens_{problem_name} = EnsembleProblem({problem_name},
                                                    prob_func = prob_func,
                                                    output_func = {scan.output_func.name}
                                                )
        """
        )
    else:
        jl.seval(
            f"""
            ens_{problem_name} = EnsembleProblem({problem_name},
                                                    prob_func = prob_func)
        """
        )
    return prob_func


def _generate_problem_solve_string(
    problem: OBEProblem, config: OBEProblemConfig
) -> str:
    save_idxs = "nothing" if config.save_idxs is None else str(config.save_idxs)
    force_dtmin = "false" if config.dtmin == 0 else "true"
    callback = "nothing" if config.callback is None else config.callback.name
    solve_string = f"""
    sol = solve(
        {problem.name},
        {config.method},
        progress={str(config.progress).lower()},
        dt={config.dt},
        abstol={config.abstol},
        reltol={config.reltol},
        callback={callback},
        saveat={config.saveat},
        save_idxs={save_idxs},
        dtmin={config.dtmin},
        force_dtmin={force_dtmin},
        save_everystep={str(config.save_everystep).lower()},
        maxiters={config.maxiters},
        dense={str(config.dense).lower()},
        save_start={str(config.save_start).lower()}
    )
    """

    return remove_leading_spaces_to_align(solve_string)


def solve_problem(
    problem: OBEProblem,
    config: OBEProblemConfig = OBEProblemConfig(),
) -> None:
    solve_string = _generate_problem_solve_string(problem, config)
    jl.seval(f"{solve_string}; tmp=0;")


def _generate_problem_parameter_scan_solve_string(
    problem: OBEEnsembleProblem, config: OBEEnsembleProblemConfig
) -> str:
    trajectories = (
        "size(params)[1]" if config.trajectories is None else str(config.trajectories)
    )
    save_idxs = "nothing" if config.save_idxs is None else str(config.save_idxs)

    callback = "nothing" if config.callback is None else config.callback.name

    solve_string = f"""
    sol = solve(
        {problem.name},
        {config.method},
        {config.distributed_method},
        abstol = {config.abstol},
        reltol = {config.reltol},
        dt = {config.dt},
        trajectories = {trajectories},
        callback = {callback},
        save_everystep = {str(config.save_everystep).lower()},
        saveat = {config.saveat},
        save_idxs = {save_idxs},
        dense = {str(config.dense).lower()},
        save_start = {str(config.save_start).lower()}
    )
    """
    return remove_leading_spaces_to_align(solve_string)


def solve_problem_parameter_scan(
    problem: OBEEnsembleProblem,
    config: OBEEnsembleProblemConfig = OBEEnsembleProblemConfig(),
) -> None:
    solve_string = _generate_problem_parameter_scan_solve_string(problem, config)
    jl.seval(f"{solve_string}; tmp=0;")


def get_results_single() -> OBEResult:
    """Retrieve the results of a single trajectory OBE simulation solution.

    Returns:
        tuple: OBEResult dataclass with the solution of the OBE for a single trajectory
    """
    y = np.array([np.array(mat) for mat in jl.seval("sol[:]")])
    results = np.real(np.einsum("jji->ji", y.T))
    t = np.array(jl.seval("sol.t"))
    return OBEResult(t, results)


def transpose_first_n(a: np.ndarray, n: int) -> np.ndarray:
    if not (0 <= n <= a.ndim):
        raise ValueError(f"n must be in [0, {a.ndim}], got {n}")
    axes = tuple(range(n - 1, -1, -1)) + tuple(range(n, a.ndim))
    return np.transpose(a, axes)


def get_results_parameter_scan(
    scan: OBEEnsembleProblem, config: OBEEnsembleProblemConfig
) -> OBEResultParameterScan:
    """
    Retrieve the results of a parameter scan

    Args:
        scan (ParameterScan): ParameterScan object containing the information of the
        parameter scan.

    Returns:
        OBEResultParameterScan: Dataclass containing the results of the parameter scan.
    """
    trajectories = config.trajectories
    if trajectories is None:
        if scan.zipped or len(scan.scan_values) == 1:
            trajectories = len(scan.scan_values[0])
        else:
            trajectories = int(np.prod([len(v) for v in scan.scan_values]))
    else:
        trajectories = trajectories

    # check if returning 2D density matrices
    return_2D = config.save_idxs is None

    scan_shape = [len(v) for v in scan.scan_values]

    # check if saving multiple timesteps
    saveat_defined = isinstance(config.saveat, (float, int)) or len(config.saveat) > 0
    if scan.zipped:
        if scan.output_func is None:
            if config.save_everystep or saveat_defined or config.save_everystep:
                raise NotImplementedError(
                    "Extracting time-dependent results from parameter scans is not yet implemented."
                )
            results = np.array(
                [jl.seval(f"sol.u[{idx + 1}][end]") for idx in range(trajectories)]
            )
        else:
            results = np.array(jl.seval("sol.u"))
        return OBEResultParameterScan(
            parameters=scan.parameters,
            scan_values=scan.scan_values,
            results=results,
            zipped=True,
        )
    else:
        size = jl.seval("size(sol)")

        if scan.output_func is None:
            if config.save_everystep or saveat_defined or config.save_everystep:
                raise NotImplementedError(
                    "Extracting time-dependent results from parameter scans is not yet implemented."
                )

            if return_2D:
                results = np.array(
                    [jl.seval(f"sol.u[{idx + 1}][end]") for idx in range(trajectories)]
                )
            else:
                if config.save_start:
                    results = np.array(
                        [
                            jl.seval(f"sol.u[{idx + 1}][end]")
                            for idx in range(trajectories)
                        ]
                    )
                else:
                    results = np.array(jl.seval("sol.u"))
        else:
            results = np.array(jl.seval("sol.u"))

        if results.ndim == 1:
            if len(scan.scan_values) > 1:
                results = results.reshape([len(v) for v in scan.scan_values][::-1]).T
        elif len(scan.scan_values) > 1:
            results = results.reshape(scan_shape[::-1] + list(results.shape[1:]))
            results = transpose_first_n(results, n=len(scan.scan_values))

        return OBEResultParameterScan(
            parameters=scan.parameters,
            scan_values=list(np.meshgrid(*scan.scan_values, indexing="ij")),
            results=results,
            zipped=False,
        )


def do_simulation_single(
    problem: OBEProblem,
    config: OBEProblemConfig = OBEProblemConfig(),
) -> OBEResult:
    """
    Perform a single trajectory solve of the OBE equations for a specified
    TlF system.

    Args:
        problem (OBEProblem): dataclass containing the OBE problem information
        config (OBEProblemConfig, optional): dataclass containing the solver
        configuration. Defaults to OBEProblemConfig().

    Returns:
        OBEResult: solver result dataclass
    """
    setup_problem(problem.odepars, problem.tspan, problem.ρ, problem.name)
    solve_problem(problem, config)
    return get_results_single()
