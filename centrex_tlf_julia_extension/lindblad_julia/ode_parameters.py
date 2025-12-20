import numbers
from collections.abc import Sequence
from typing import Any, List, Optional, Sequence, Set, Union

import numpy as np
import numpy.typing as npt
import sympy as smp
from centrex_tlf import hamiltonian
from centrex_tlf.couplings import TransitionSelector
from sympy.parsing import sympy_parser
from sympy.printing.julia import julia_code

from .parse_julia_functions import julia_functions
from .utils_julia import jl

__all__ = ["odeParameters", "generate_ode_parameters"]

type_conv = {
    "int": "Int64",
    "float": "Float64",
    "complex": "ComplexF64",
    "float64": "Float64",
    "int32": "Int32",
    "int64": "Int64",
    "complex64": "ComplexF32",
    "complex128": "ComplexF64",
    "float32": "Float32",
    "float16": "Float16",
}


def is_sequence(x: Any) -> bool:
    # Accept 1D numpy arrays as sequences
    if isinstance(x, np.ndarray):
        return x.ndim == 1
    return isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray))


def is_homogeneous_sequence(x: Any) -> bool:
    if not is_sequence(x):
        return False

    # normalize to something indexable for both list/tuple and ndarray
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return False
        # for numpy arrays, homogeneity is about dtype
        return True

    if len(x) == 0:
        return False

    t0 = type(x[0])
    return all(type(e) is t0 for e in x)


def _type_key(value: Any) -> str:
    # NumPy scalar (np.float64, np.int32, np.complex128, etc.)
    if isinstance(value, np.generic):
        return value.dtype.name
    # 0-d numpy array, if that can occur
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.dtype.name
    # Python scalar: 'int', 'float', 'complex', 'bool', ...
    return type(value).__name__


def type_conversion(value: Any, type_conv: dict[str, str]) -> str:
    if is_sequence(value):
        if not is_homogeneous_sequence(value):
            raise TypeError("Inhomogeneous sequences are not supported")

        if isinstance(value, np.ndarray):
            N = int(value.size)
            # map by dtype name for arrays
            key = value.dtype.name
        else:
            N = len(value)
            key = _type_key(value[0])

        julia_type = type_conv.get(key)
        if julia_type is None:
            raise TypeError(f"Type {key} not supported for odeParameters")

        return f"NTuple{{{N},{julia_type}}}"

    # numeric scalar (includes numpy scalar)
    if isinstance(value, numbers.Number) or isinstance(value, np.generic):
        key = _type_key(value)
        julia_type = type_conv.get(key)
        if julia_type is None:
            raise TypeError(f"Type {key} not supported for odeParameters")
        return julia_type

    raise TypeError(f"Type {_type_key(value)} not supported for odeParameters")


def julia_literal(value: Any) -> str:
    """Convert python/numpy scalar or 1D sequence into a Julia literal."""
    # numpy scalar -> python scalar
    if isinstance(value, np.generic):
        value = value.item()

    if is_sequence(value):
        if isinstance(value, np.ndarray):
            elems = [julia_literal(x.item()) for x in value.ravel()]
        else:
            elems = [julia_literal(x) for x in value]

        # Julia tuple literal
        if len(elems) == 1:
            return f"({elems[0]},)"  # 1-tuple needs trailing comma
        return "(" + ", ".join(elems) + ")"

    # scalar literals
    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, complex):
        # Julia: a + b*im
        a = value.real
        b = value.imag
        return f"({a}) + ({b})*im"

    if isinstance(value, (int, float)):
        # str() is fine here; if you care about Float64 formatting, do it upstream
        return str(value)

    raise TypeError(
        f"Cannot convert value of type {type(value).__name__} to Julia literal"
    )


class odeParameters:
    def __init__(self, *args, **kwargs):
        # if elif statement is for legacy support, where a list of parameters was
        # supplied
        if len(args) > 1:
            raise AssertionError(
                "For legacy support supply a list of strings, one for each parameter"
            )
        elif len(args) == 1:
            assert isinstance(args[0][0], str), (
                "For legacy support supply a list of strings, one for each parameter"
            )
            if "ρ" not in args[0]:
                args[0].append("ρ")
            kwargs = {par: 0.0 for par in args[0]}
            odeParameters(**kwargs)

        self._parameters = [
            key for key, val in kwargs.items() if not isinstance(val, str)
        ]
        self._compound_vars = [
            key for key, val in kwargs.items() if isinstance(val, str)
        ]

        for key, val in kwargs.items():
            # ϕ = 'ϕ') results in different unicode representation
            # replace the rhs with the rep. used on the lhs
            if isinstance(val, str):
                val = val.replace("\u03d5", "\u03c6")
            setattr(self, key, val)

        self._check_symbols_defined()
        self._order_compound_vars()

        # checking types, necessary if the ODE parameters contain arrays or list
        # Julia can't do type inference then and this tanks performance
        # storing the input types here for use in generate_preamble, but this is
        # only used if one of the input parameters is an array
        self._parameter_types = [
            type_conversion(getattr(self, par), type_conv) for par in self._parameters
        ]

        self._array_types = {
            par: type_conversion(getattr(self, par), type_conv)
            for par in self._parameters
            if is_sequence(getattr(self, par))
        }
        self._method = "expanded"

    def __setattr__(self, name: str, value: Any) -> None:
        if name in [
            "_parameters",
            "_compound_vars",
            "_parameter_types",
            "_array_types",
        ]:
            super(odeParameters, self).__setattr__(name, value)
        elif name in self._parameters:
            assert not isinstance(value, str), (
                "Cannot change parameter from numeric to str"
            )
            super(odeParameters, self).__setattr__(name, value)
        elif name in self._compound_vars:
            assert isinstance(value, str), "Cannot change parameter from str to numeric"
            super(odeParameters, self).__setattr__(name, value)
        elif name == "_method":
            super(odeParameters, self).__setattr__(name, value)
        else:
            raise AssertionError(
                "Cannot instantiate new parameter on initialized OdeParameters object"
            )

    def _get_defined_symbols(self) -> Set[smp.Symbol]:
        symbols_defined = self._parameters + self._compound_vars
        symbols_defined += ["t"]
        symbols_defined_set = set([smp.Symbol(s) for s in symbols_defined])
        return symbols_defined_set

    def _get_numerical_symbols(self) -> Set[smp.Symbol]:
        symbols_numerical = self._parameters[:]
        symbols_numerical += ["t"]
        symbols_numerical_set = set([smp.Symbol(s) for s in symbols_numerical])
        return symbols_numerical_set

    def _get_expression_symbols(self) -> Set[smp.Symbol]:
        symbols_expressions_list = [
            sympy_parser.parse_expr(getattr(self, s)) for s in self._compound_vars
        ]
        symbols_expressions = set().union(
            *[s.free_symbols for s in symbols_expressions_list]
        )
        return symbols_expressions

    def _check_symbols_defined(self) -> None:
        symbols_defined = self._get_defined_symbols()
        symbols_expressions = self._get_expression_symbols()

        warn_flag = False
        warn_string = "Symbol(s) not defined: "
        for se in symbols_expressions:
            if se not in symbols_defined:
                warn_flag = True
                warn_string += f"{se}, "
        if warn_flag:
            raise AssertionError(warn_string.strip(" ,"))

    def check_symbols_in_parameters(
        self, symbols_other: Union[str, Sequence[str], Set[str], smp.Symbol]
    ) -> None:
        if not isinstance(symbols_other, (list, np.ndarray, tuple, set)):
            symbols_other = [symbols_other]
        elif isinstance(symbols_other, set):
            symbols_other = list(symbols_other)

        if len(symbols_other) == 0:
            return

        if isinstance(symbols_other[0], smp.Symbol):
            symbols_other = [str(sym) for sym in symbols_other]

        parameters = self._parameters[:]
        parameters += ["t"]

        warn_flag = False
        warn_string = "Symbol(s) not defined: "
        for se in symbols_other:
            if se not in parameters:
                warn_flag = True
                warn_string += f"{se}, "
        if warn_flag:
            raise AssertionError(warn_string.strip(" ,"))

    def _order_compound_vars(self) -> None:
        symbols_num = self._get_numerical_symbols()
        unordered = list(self._compound_vars)
        ordered = []

        while len(unordered) != 0:
            for compound in unordered:
                if compound not in ordered:
                    symbols = sympy_parser.parse_expr(
                        getattr(self, compound)
                    ).free_symbols
                    m = [
                        True if (s in symbols_num) or (str(s) in ordered) else False
                        for s in symbols
                    ]
                    if all(m):
                        ordered.append(compound)
            unordered = [val for val in unordered if val not in ordered]
        self._compound_vars = ordered

    def _get_index_parameter(self, parameter: str, mode: str = "python") -> int:
        # OdeParameter(ϕ = 'ϕ') results in different unicode representation
        # replace the rhs with the rep. used on the lhs
        parameter = parameter.replace("\u03d5", "\u03c6")
        if mode == "python":
            return self._parameters.index(parameter)
        elif mode == "julia":
            return self._parameters.index(parameter) + 1
        else:
            raise ValueError(f"mode {mode} not supported, use 'julia' or 'python'")

    @property
    def p(self) -> List[Any]:
        p = [getattr(self, p) for p in self._parameters]
        for idp, pi in enumerate(p):
            if type(pi) == np.ndarray:
                continue
            elif type(pi).__module__ == "numpy":
                p[idp] = pi.item()
            else:
                continue
        return p

    def get_index_parameter(
        self, parameter: Union[str, Sequence[str]], mode: str = "python"
    ) -> Union[int, List[int]]:
        if isinstance(parameter, str):
            return self._get_index_parameter(parameter, mode)
        elif isinstance(parameter, (list, np.ndarray)):
            return [self._get_index_parameter(par, mode) for par in parameter]
        else:
            raise TypeError(f"parameter type {type(parameter)} not supported")

    def check_transition_symbols(
        self, transitions: Sequence[TransitionSelector]
    ) -> bool:
        # check Rabi rate and detuning symbols
        to_check = ["Ω", "δ"]
        symbols_defined = [str(s) for s in self._get_defined_symbols()]
        not_defined = []
        for transition in transitions:
            for var in to_check:
                var = str(getattr(transition, var))
                if var is not None:
                    if var not in symbols_defined:
                        not_defined.append(var)
        if len(not_defined) > 0:
            not_defined_set = set([str(s) for s in not_defined])
            raise AssertionError(
                f"Symbol(s) from transitions not defined: {', '.join(not_defined_set)}"
            )

        # check polarization switching symbols
        to_check_pol = []
        for transition in transitions:
            to_check_pol.extend(transition.polarization_symbols)

        symbols_defined_set = self._get_defined_symbols()

        warn_flag = False
        warn_string = "Symbol(s) in transition polarization switching not defined: "
        for ch in to_check_pol:
            # need to convert to real sympy symbol for comparison
            if smp.Symbol(ch.name) not in symbols_defined_set:
                warn_flag = True
                warn_string += f"{ch}, "
        if warn_flag:
            raise AssertionError(warn_string.strip(" ,"))
        return True

    def generate_p_julia(self) -> str:
        elems = [julia_literal(pi) for pi in self.p]
        if len(elems) == 1:
            jl_string = f"({elems[0]},)"
        else:
            jl_string = "(" + ", ".join(elems) + ")"

        if self._method == "expanded":
            # Define p directly
            jl.seval(f"p = {jl_string}")

        elif self._method == "matrix":
            # One seval, avoid intermediate globals where possible
            jl.seval(
                f"""
                p_values = {jl_string}
                HamFun = HamFunctor(p_values...)
                p = LindbladParameters(HamFun, DissFun, buf)
                """
            )

        else:
            raise ValueError(f"Unknown method: {self._method!r}")

        return jl_string

    def __repr__(self) -> str:
        rep = "OdeParameters("
        for par in self._parameters:
            rep += f"{par}={getattr(self, par)}, "
        return rep.strip(", ") + ")"

    def get_parameter_evolution(
        self, t: npt.NDArray[np.float64], parameter: str
    ) -> npt.NDArray[np.float64]:
        """Get the time evolution of parameters in odeParameters.
        Evaluates expressions in python if possible, otherwise calls julia to
        evaluate the expressions

        Args:
            t (np.ndarray[float]): array of timestamps
            parameter (str): parameter to evaluate over t

        Returns:
            np.ndarray : array with values of parameter corresponding to t

        """
        # get a list of all parameters in odeParameters
        parameters = self._parameters + self._compound_vars
        # check if `parameter` is defined in odeParameters
        assert parameter in parameters, f"{parameter} not defined in odeParameters"
        # if `parameter` is not a compound variable, an array of size t of
        # parameter
        if parameter in self._parameters:
            return np.ones(len(t)) * getattr(self, str(parameter))
        elif parameter in self._compound_vars:
            expression = getattr(self, str(parameter))
            # parse expression string to sympy equation
            expression = sympy_parser.parse_expr(expression)
            while True:
                symbols_in_expression = [
                    sym for sym in expression.free_symbols if sym is not smp.Symbol("t")
                ]
                symbols_in_expression = [str(sym) for sym in symbols_in_expression]
                compound_bool = [
                    sym in self._compound_vars for sym in symbols_in_expression
                ]
                # if any of the symbols in the expression are compound variables
                # fetch the related compound expression and insert it in the
                # expression
                if np.any(compound_bool):
                    for idx in np.where(compound_bool)[0]:
                        compound_var = sympy_parser.parse_expr(
                            getattr(self, symbols_in_expression[idx])
                        )
                        expression = expression.subs(
                            symbols_in_expression[idx], compound_var
                        )
                else:
                    # break if none of the symbols in the expression are compound
                    # variables
                    break

            # substitute the numerical variables for the expressions
            symbols_in_expression = [
                sym for sym in expression.free_symbols if sym is not smp.Symbol("t")
            ]
            array_symbols = []
            for sym in symbols_in_expression:
                # can't subs a symbol with a list, tuple or array
                if isinstance(getattr(self, str(sym)), (list, tuple, np.ndarray)):
                    array_symbols.append(sym)
                expression = expression.subs(sym, getattr(self, str(sym)))
            functions_in_expression = [
                str(fn).split("(")[0] for fn in expression.atoms(smp.Function)
            ]
            # check if any of the functions are special julia defined functions
            if np.any([fn in julia_functions for fn in functions_in_expression]):
                expression = str(expression)
                expression_smp = sympy_parser.parse_expr(expression)
                expression_jl = julia_code(expression_smp, strict=False)
                expression = "\n".join(
                    line
                    for line in expression_jl.splitlines()
                    if not line.strip().startswith("#")
                )
                # broadcast the function, allows for input of an array of t
                for fn in functions_in_expression:
                    expression = expression.replace(fn, f"{fn}.")
                for sym in array_symbols:
                    expression = expression.replace(
                        str(sym), f"Ref({getattr(self, str(sym))})"
                    )
                # quick workaround to convert int to float for function inputs
                # LOOK AT THIS LATER
                expression = expression.replace(", 0,", ", 0.0,")
                # evaluate the specified parameter expression in julia
                jl.seval(f"_tmp_func(t) = {expression}")
                # can't get broadcasting to work if some variables are of array or list
                # type, use map
                jl.tmp_t = t
                return np.array(jl.seval("map(_tmp_func, tmp_t)"))
            else:
                # evaluate the specified parameter expression in python
                func = smp.lambdify(
                    smp.Symbol("t"), expression, modules=["numpy", "scipy"]
                )
                res = func(t)
                if np.shape(res) == ():
                    return np.ones(len(t)) * res
                else:
                    return res

        raise ValueError(f"Parameter {parameter} not found")

    def reorder(self, order: Sequence[str]) -> None:
        """Reorder the parameters in odeParameters according to order"""
        new_order = []
        if not isinstance(order, (list, tuple, np.ndarray, set)):
            raise TypeError("order must be a sequence of parameter names")
        order_set = set(order)
        missing = set(self._parameters) - order_set
        if missing:
            raise ValueError(
                f"Parameter(s) not present in order: {', '.join(sorted(missing))}"
            )

        for par in order:
            if par not in self._parameters:
                raise ValueError(f"Parameter {par} not found in odeParameters")
            new_order.append(par)
        self._parameters = new_order


def generate_ode_parameters(
    transition_selectors: List[TransitionSelector], **kwargs
) -> odeParameters:
    parameters = []
    for idt, ts in enumerate(transition_selectors):
        if ts.phase_modulation:
            pars = [
                (str(ts.Ω), f"Ωt{idt}*phase_modulation(t, β{idt}, ωphase{idt})"),
                (f"Ωt{idt}", hamiltonian.Γ),
                (f"β{idt}", 3.8),
                (f"ωphase{idt}", hamiltonian.Γ),
            ]
            parameters.extend(pars)
        else:
            parameters.append((str(ts.Ω), hamiltonian.Γ))
        parameters.append((str(ts.δ), 0.0))
        ps = ts.polarization_symbols
        if len(ps) == 1:
            parameters.append((str(ps[0]), 1))
        elif len(ps) == 2:
            pars = [
                (f"ω{idt}", hamiltonian.Γ),
                (f"φ{idt}", 0.0),
                (f"P{idt}", f"sin(ω{idt}*t+φ{idt})"),
                (str(ps[0]), f"P{idt} > 0"),
                (str(ps[1]), f"P{idt} <= 0"),
            ]
            parameters.extend(pars)
        else:
            raise ValueError(
                "TransitionSelector with more than two polarization "
                "switching symbols not supported"
            )
    parameters_dict = dict(parameters)
    if kwargs is not None:
        for key, val in kwargs.items():
            parameters_dict[key] = val
    return odeParameters(**parameters_dict)
    return odeParameters(**parameters_dict)
