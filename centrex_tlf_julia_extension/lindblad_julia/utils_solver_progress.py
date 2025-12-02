from .utils_julia import jl
from .utils_solver import OBEEnsembleProblem, OBEEnsembleProblemConfig

__all__ = ["solve_problem_parameter_scan_progress"]


def solve_problem_parameter_scan_progress(
    problem: OBEEnsembleProblem,
    config: OBEEnsembleProblemConfig,
) -> None:
    ensemble_problem_name = problem.name
    problem_name = problem.problem.name
    method = config.method
    abstol = config.abstol
    reltol = config.reltol
    # dt = config.dt
    callback = config.callback
    # dtmin = config.dtmin
    # maxiters = config.maxiters
    saveat = config.saveat
    trajectories = config.trajectories
    save_idxs = config.save_idxs
    distributed_method = config.distributed_method
    save_everystep = config.save_everystep
    output_func = problem.output_func

    _trajectories = "size(params)[1]" if trajectories is None else trajectories
    _callback = "nothing" if callback is None else callback
    _saveat = "[]" if saveat is None else str(saveat)
    _save_idxs = "nothing" if save_idxs is None else str(save_idxs)

    if output_func is None:
        jl.seval(
            """
            @everywhere function output_func_progress(sol, i)
                put!(channel, 1)
                sol, false
            end
        """
        )
    else:
        jl.seval(
            f"""
            @everywhere function output_func_progress(sol, i)
                put!(channel, 1)
                a,b = {output_func}(sol, i)
                return a,b
            end
        """
        )
    jl.seval(
        f"""
        {ensemble_problem_name} = EnsembleProblem({problem_name},
                                                prob_func = prob_func,
                                                output_func = output_func_progress
                                            )
    """
    )

    jl.seval(
        """
        if !@isdefined channel
            const channel = RemoteChannel(()->Channel{Int}(1))
            @everywhere const channel = $channel
        end
    """
    )

    jl.seval(
        f"""
        progress = Progress({_trajectories}, showspeed = true)
        @sync sol = begin
            @async begin
                tasksdone = 0
                while tasksdone < {_trajectories}
                    tasksdone += take!(channel)
                    update!(progress, tasksdone)
                end
            end
            @async begin
                @time global sol = solve({ensemble_problem_name}, {method},
                            {distributed_method}, trajectories={_trajectories},
                            abstol = {abstol}, reltol = {reltol},
                            callback = {_callback},
                            save_everystep = {str(save_everystep).lower()},
                            saveat = {_saveat},
                            save_idxs = {_save_idxs})
            end
    end
    """
    )
