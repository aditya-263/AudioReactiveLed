
from numpy import abs, append, arange, insert, linspace, log10, round, zeros


def hertz_to_mel(freq):

    return 2595.0 * log10(1 + (freq / 700.0))


def mel_to_hertz(mel):
    return 700.0 * (10**(mel / 2595.0)) - 700.0


def melfrequencies_mel_filterbank(num_bands, freq_min, freq_max, num_fft_bands):

    mel_max = hertz_to_mel(freq_max)
    mel_min = hertz_to_mel(freq_min)
    delta_mel = abs(mel_max - mel_min) / (num_bands + 1.0)
    frequencies_mel = mel_min + delta_mel * arange(0, num_bands + 2)
    lower_edges_mel = frequencies_mel[:-2]
    upper_edges_mel = frequencies_mel[2:]
    center_frequencies_mel = frequencies_mel[1:-1]
    return center_frequencies_mel, lower_edges_mel, upper_edges_mel


def compute_melmat(num_mel_bands=12, freq_min=64, freq_max=8000,
                   num_fft_bands=513, sample_rate=16000):
    center_frequencies_mel, lower_edges_mel, upper_edges_mel =  \
        melfrequencies_mel_filterbank(
            num_mel_bands,
            freq_min,
            freq_max,
            num_fft_bands
        )

    center_frequencies_hz = mel_to_hertz(center_frequencies_mel)
    lower_edges_hz = mel_to_hertz(lower_edges_mel)
    upper_edges_hz = mel_to_hertz(upper_edges_mel)
    freqs = linspace(0.0, sample_rate / 2.0, num_fft_bands)
    melmat = zeros((num_mel_bands, num_fft_bands))

    for imelband, (center, lower, upper) in enumerate(zip(
            center_frequencies_hz, lower_edges_hz, upper_edges_hz)):

        left_slope = (freqs >= lower) == (freqs <= center)
        melmat[imelband, left_slope] = (
            (freqs[left_slope] - lower) / (center - lower)
        )

        right_slope = (freqs >= center) == (freqs <= upper)
        melmat[imelband, right_slope] = (
            (upper - freqs[right_slope]) / (upper - center)
        )

    return melmat, (center_frequencies_mel, freqs)
