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

    Compute phase modulation at frequency ω with a modudulation strength β at time t
    """
    function phase_modulation(t::Float64, β::Float64, ω::Float64)::ComplexF64
        return exp(1im.*β.*sin(ω.*t))
    end

    """
        square_wave(t::Float64, ω::Float64, phase::Float64)

    generate a square wave from 0 to 1 at frequency ω [2π Hz; rad/s] and phase offset ϕ [rad]
    """
    function square_wave(t::Float64, ω::Float64, phase::Float64)::Float64
        0.5.*(1 .+ squarewave(ω.*t .+ phase))
    end

    """
        resonant_switching(t::Float64, ω::Float64, phase::Float64)

    generate the polarization coming from a resonant EOM
    """
    function resonant_switching(t::Float64, ω::Float64, phase::Float64)::Float64
        -cos(pi*(1 .+ cos(ω .* t .+ phase))/2)/2 + 1/2
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
        ω = 2π*1.0/(ton+toff)
        sawtooth_wave(t, ω, phase) <= ton/(ton+toff) ? 1. : 0.
    end

    """
        multipass_2d_intensity(x::Float64, y::Float64, amplitudes::Vector{Float64}, xlocs::Vector{Float64}, ylocs::Vector{Float64}, σx::Float64, σy::Float64)::Float64

    generate a multipass with 2D gaussian intensity profiles for each pass
    """
    function multipass_2d_intensity(x::Float64, y::Float64, amplitudes::Vector{Float64}, xlocs::Vector{Float64}, ylocs::Vector{Float64}, σx::Float64, σy::Float64)::Float64
        intensity::Float64 = 0.0
        for i = 1:length(amplitudes::Vector{Float64})
            @inbounds intensity += gaussian_2d(x,y,amplitudes[i],xlocs[i],ylocs[i], σx,σy)
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
    multipass_2d_rabi(x::Float64, y::Float64, intensities::Vector{Float64}, xlocs::Vector{Float64}, ylocs::Vector{Float64}, σx::Float64, σy::Float64, main_coupling::Float64, D::Float64=2.6675506e-30)::Float64
        generate a multipass with 2D intensity profiles for each pass and convert to a rabi rate for with
        the default D set for the X to B TlF transition.
    """
    function multipass_2d_rabi(x::Float64, y::Float64, intensities::Vector{Float64}, xlocs::Vector{Float64}, ylocs::Vector{Float64}, σx::Float64, σy::Float64, main_coupling::Float64, D::Float64=2.6675506e-30)::Float64
        intensity = multipass_2d_intensity(x, y, intensities, xlocs, ylocs, σx, σy)
        Ω = rabi_from_intensity(intensity, main_coupling, D)
        return Ω
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
end