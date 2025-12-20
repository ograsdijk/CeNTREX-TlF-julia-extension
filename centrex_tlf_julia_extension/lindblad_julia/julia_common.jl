using Distributed
@everywhere using Waveforms

@everywhere begin
    """
        gaussian_2d(x::Float64, y::Float64, a::Float64, μx::Float64, μy::Float64, σx::Float64, σy::Float64)::Float64

    Compute the 2D gaussian at point x,y for an amplitude a, mean value μx and μy,
    and a standard deviation σx and σy
    """
    function gaussian_2d(x::Float64, y::Float64, a::Float64, μx::Float64, μy::Float64, σx::Float64, σy::Float64)::Float64
        a.*exp(.- ((x.-μx).^2 ./ (2 .* σx.*σx) + (y.-μy).^2 ./ (2 .* σy.*σy)))
    end

    """
        gaussian_2d_rotated(x::Float64, y::Float64, amplitude::Float64, μx::Float64, μy::Float64, σx::Float64, σy::Float64, θ::Float64)::Float64

    Compute the rotated 2D gaussian at point x,y for an amplitude a, mean value μx and μy, standard deviation σx and σy
    and rotation angle θ
    """
    function gaussian_2d_rotated(x::Float64, y::Float64, amplitude::Float64, μx::Float64, μy::Float64, σx::Float64, σy::Float64, θ::Float64)::Float64
        a = cos(θ)^2 / (2*σx^2) + sin(θ)^2 / (2*σy^2)
        b = sin(2*θ) / (2*σx^2) - sin(2*θ) / (2*σy^2)
        c = sin(θ)^2 / (2*σx^2) + cos(θ)^2 / (2*σy^2)

        amplitude.*exp(- a*(x-μx)^2 - b*(x-μx)*(y-μy) - c*(y-μy)^2)
    end

    """
        phase_modulation(t::Float64, β::Float64, ω::Float64)::ComplexF64

    Compute phase modulation at frequency ω with a modudulation strength β at time t and returns the relative electric field.
    Need to square to get the relative powers
    """
    function phase_modulation(t::Float64, β::Float64, ω::Float64)::ComplexF64
        φ = β * sin(ω * t)
        sφ, cφ = sincos(φ)
        return cφ + im*sφ
    end

    """
        square_wave(t::Float64, ω::Float64, phase::Float64)

    generate a square wave from 0 to 1 at frequency ω [2π Hz; rad/s] and phase offset ϕ [rad]
    """
    function square_wave(t::Float64, ω::Float64, phase::Float64)::Float64
        0.5.*(1 .+ squarewave(ω.*t .+ phase))
    end

    """
        resonant_polarization_modulation(t::Float64, ω::Float64, phase::Float64)

    generate a single polarization component coming from a resonant polarization
    modulating EOM
    """
    function resonant_polarization_modulation(t::Float64, γ::Float64, ω::Float64)::ComplexF64
        θ = 0.5 * γ * sin(ω * t)        # θ = (γ/2) * sin(Ω t)
        sθ, cθ = sincos(θ)             # compute sin and cos together
        a = cθ + sθ                    # cos(θ) + sin(θ)
        a2 = 0.5 * a                   # (cos(θ) + sin(θ)) / 2
        return complex(a2, a2)         # (1 + i)/2 * (cos(θ) + sin(θ))
    end
    """
        sawtooth_wave(t::Float64, ω::Float64, phase::Float64)::Float64

    generate a sawtooth wave from 0 to 1 at frequency ω [2π Hz; rad/s] and phase offset phase [rad]
    """
    function sawtooth_wave(t::Float64, ω::Float64, phase::Float64)::Float64
        0.5.*(1 .+ sawtoothwave(ω.*t .+ phase - π))
    end

    """

    """
    function variable_on_off(t::Float64, ton::Float64, toff::Float64, phase::Float64)::Float64
        T = ton + toff
        duty = ton / T

        # fractional phase in [0, 1)
        frac = mod(2π * t / T + phase, 2π) * (1 / (2π))

        return ifelse(frac < duty, 1.0, 0.0)
    end


    const INV_2PI = 1.0 / (2π)
    """
        variable_on_off_duty_invT(t::Float64, duty::Float64, invT::Float64, phase::Float64) -> Float64

    Periodic on/off gate (fast form).

    - `duty`  : duty cycle in [0, 1] (fraction of the period that is ON)
    - `invT`  : inverse period, `1/T`
    - `phase` : phase offset [rad]

    This version is optimized for tight inner loops: pass `invT = 1/T` (and usually a constant `phase`)
    precomputed outside the ODE RHS. Returns `1.0` during the ON portion of the cycle and `0.0` otherwise.
    """
    @inline function variable_on_off_duty_invT(
        t::Float64,
        duty::Float64,
        invT::Float64,
        phase::Float64,
    )::Float64
        frac = mod1(t * invT + phase * INV_2PI, 1.0)  # in (0,1]
        return ifelse(frac < duty, 1.0, 0.0)
    end

    """
        multipass_2d_intensity(x, y, amplitudes, xlocs, ylocs, σx, σy) -> Float64

    Compute the total intensity from a multipass consisting of 2D Gaussian profiles.

    Each pass `i` contributes `gaussian_2d(x, y, amplitudes[i], xlocs[i], ylocs[i], σx, σy)`,
    and the result is summed over `i`. The containers `amplitudes`, `xlocs`, and `ylocs`
    may be any indexable collections with matching indices (e.g. `NTuple`, `Vector`,
    `SVector`).

    # Arguments
    - `x`, `y`: transverse coordinates.
    - `amplitudes`: per-pass amplitudes.
    - `xlocs`, `ylocs`: per-pass beam centers.
    - `σx`, `σy`: Gaussian widths.

    # Returns
    - Total intensity at `(x, y)` as `Float64`.
    """
    function multipass_2d_intensity(
        x::Float64,
        y::Float64,
        amplitudes,
        xlocs,
        ylocs,
        σx::Float64,
        σy::Float64,
    )::Float64
        intensity::Float64 = 0.0
        @inbounds for i in eachindex(amplitudes, xlocs, ylocs)
            intensity += gaussian_2d(x, y, amplitudes[i], xlocs[i], ylocs[i], σx, σy)
        end
        return intensity
    end

    """
        rabi_from_intensity(intensity::Float64, coupling::Float64, D::Float64=2.6675506e-30)::Float64

    generate the rabi rate from intensity with the default D for the X to B TlF transition.
    """

    function rabi_from_intensity(intensity::Float64, coupling::Float64, D::Float64=2.6675506e-30)::Float64
        hbar = 1.0545718176461565e-34
        c = 299792458.0
        ϵ0 = 8.8541878128e-12
        E = sqrt(intensity * 2 / (c * ϵ0))
        Ω = E * coupling  * D / hbar
        return Ω
    end

    """
        multipass_2d_rabi(x, y, intensities, xlocs, ylocs, σx, σy, main_coupling; D=2.6675506e-30) -> Float64

    Generate a multipass with 2D intensity profiles for each pass and convert it to a Rabi rate.
    Default `D` is set for the X→B TlF transition.
    """
    @inline function multipass_2d_rabi(
        x::Float64,
        y::Float64,
        intensities,
        xlocs,
        ylocs,
        σx::Float64,
        σy::Float64,
        main_coupling::Float64,
        D::Float64 = 2.6675506e-30,
    )::Float64
        intensity = multipass_2d_intensity(x, y, intensities, xlocs, ylocs, σx, σy)
        return rabi_from_intensity(intensity, main_coupling, D)
    end

    """
    gaussian_beam_rabi(x::Float64, y::Float64, intensity::Float64, xloc::Float64, yloc::Float64, σx::Float64, σy::Float64, main_coupling::Float64, D::Float64=2.6675506e-30)::Float64
        2D gaussian beam that takes the beam size and peak intensity and converts it to a rabi rate
    """
    function gaussian_beam_rabi(x::Float64, y::Float64, intensity::Float64, xloc::Float64, yloc::Float64, σx::Float64, σy::Float64, main_coupling::Float64, D::Float64=2.6675506e-30)::Float64
        intensity = gaussian_2d(x, y, intensity, xloc, yloc, σx, σy)
        rabi = rabi_from_intensity(intensity, main_coupling, D)
        return rabi
    end

    function mirror_hermitian!(C::StridedMatrix{T}, uplo::Char = 'U') where {T<:Complex}
        n = size(C, 1)
        @inbounds if uplo == 'U'
            # copy upper → lower (conjugate)
            for k in 1:n-1
                i = 1
                j = k + 1
                @simd for _ = 1:n-k
                    C[j, i] = conj(C[i, j])
                    i += 1
                    j += 1
                end
            end
        else
            # copy lower → upper (conjugate)
            for k in 1:n-1
                i = 1
                j = k + 1
                @simd for _ = 1:n-k
                    C[i, j] = conj(C[j, i])
                    i += 1
                    j += 1
                end
            end
        end
        return C
    end

    @inline _alpha_beta(::Type{T}) where {T<:Complex} = (T(zero(real(T)), one(real(T))), zero(real(T)))

    function liouvillian_commutator_her2k!(C::StridedMatrix{T},
                                        A::StridedMatrix{T},
                                        B::StridedMatrix{T};
                                        uplo::Char='U') where {T<:Complex}
        α, β = _alpha_beta(T)
        # HER2K with trans='N': C := α*A*B' + conj(α)*B*A'
        # If A,B are Hermitian, this equals i*(BA - AB).
        BLAS.her2k!(uplo, 'N', α, B, A, β, C)   # note order (B,A) to match i*(BA - AB)
        # mirror to full Hermitian C
        mirror_hermitian!(C, uplo)
        return nothing
    end

    """
        liouvillian_commutator!(C, A, B)

    Compute the quantum commutator -i[A,B] = -i(AB - BA) and store the result in C.

    Efficiently computes the matrix commutator in-place using optimized matrix operations.
    This is commonly used in quantum mechanics to represent Heisenberg equations of motion.

    # Arguments
    - `C`: Output matrix where the commutator result will be stored
    - `A`: First input matrix
    - `B`: Second input matrix

    # Returns
    - `nothing`: This function modifies `C` in-place

    # Notes
    - Requires all matrices to have compatible dimensions

    # Example
    ```julia
    A = rand(ComplexF64, 3, 3)
    B = rand(ComplexF64, 3, 3)
    C = zeros(ComplexF64, 3, 3)
    liouvillian_commutator!(C, A, B)  # C now contains -i[A,B]
    ```
    """
    function liouvillian_commutator!(C, A, B)
        @inbounds begin
            mul!(C, B, A)                # C = BA
            mul!(C, A, B, -1im, 1im)     # C = 1im * BA + (-1im) * AB = i(BA - AB) = -i[A,B]
        end
        return nothing
    end
end