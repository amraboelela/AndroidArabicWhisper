#include "feature_extractor.h"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <complex>
#include <algorithm>
#include <optional>

// Function to simulate numpy's rfftfreq
std::vector<float> rfftfreq(int n, float d) {
  std::vector<float> freqs;
  freqs.reserve(n / 2 + 1);
  for (int i = 0; i <= n / 2; ++i) {
    freqs.push_back(i / (n * d));
  }
  return freqs;
}

// Function to simulate np.linspace
std::vector<float> linspace(float start, float end, int num) {
  std::vector<float> result;
  if (num <= 0) return result;
  result.reserve(num);
  float step = (end - start) / (num - 1);
  for (int i = 0; i < num; ++i) {
    result.push_back(start + i * step);
  }
  return result;
}

// A simple vector dot product
std::vector<float> dot(const Matrix& a, const std::vector<float>& b) {
  if (a.empty()) return {};
  if (a[0].size() != b.size()) {
    throw std::invalid_argument("Matrix dimensions do not match for dot product.");
  }

  std::vector<float> result(a.size(), 0.0f);
  for (size_t i = 0; i < a.size(); ++i) {
    for (size_t j = 0; j < a[0].size(); ++j) {
      result[i] += a[i][j] * b[j];
    }
  }
  return result;
}

// --- FeatureExtractor Class Implementation ---

FeatureExtractor::FeatureExtractor(
    int feature_size,
    int sampling_rate,
    int hop_length,
    int chunk_length,
    int n_fft
) : n_fft(n_fft),
    hop_length(hop_length),
    chunk_length(chunk_length),
    sampling_rate(sampling_rate)
{
  n_samples = chunk_length * sampling_rate;
  nb_max_frames = n_samples / hop_length;
  time_per_frame = (float)hop_length / sampling_rate;
  mel_filters = get_mel_filters(sampling_rate, n_fft, feature_size);
}

Matrix FeatureExtractor::get_mel_filters(int sr, int n_fft, int n_mels) {
  std::vector<float> fftfreqs = rfftfreq(n_fft, 1.0f / sr);

  const float min_mel = 0.0f;
  const float max_mel = 45.245640471924965f;
  std::vector<float> mels = linspace(min_mel, max_mel, n_mels + 2);

  const float f_min = 0.0f;
  const float f_sp = 200.0f / 3.0f;
  std::vector<float> freqs(mels.size());
  for (size_t i = 0; i < mels.size(); ++i) {
    freqs[i] = f_min + f_sp * mels[i];
  }

  const float min_log_hz = 1000.0f;
  const float min_log_mel = (min_log_hz - f_min) / f_sp;
  const float logstep = log(6.4f) / 27.0f;
  for (size_t i = 0; i < mels.size(); ++i) {
    if (mels[i] >= min_log_mel) {
      freqs[i] = min_log_hz * exp(logstep * (mels[i] - min_log_mel));
    }
  }

  std::vector<float> fdiff(freqs.size() - 1);
  for (size_t i = 0; i < fdiff.size(); ++i) {
    fdiff[i] = freqs[i+1] - freqs[i];
  }

  Matrix weights(n_mels, std::vector<float>(fftfreqs.size()));

  for (int i = 0; i < n_mels; ++i) {
    float f_i = freqs[i];
    float f_i1 = freqs[i+1];
    float f_i2 = freqs[i+2];

    for (size_t j = 0; j < fftfreqs.size(); ++j) {
      float f_j = fftfreqs[j];
      float ramp;
      if (f_j >= f_i && f_j <= f_i1) {
        ramp = (f_j - f_i) / (f_i1 - f_i);
      } else if (f_j >= f_i1 && f_j <= f_i2) {
        ramp = (f_i2 - f_j) / (f_i2 - f_i1);
      } else {
        ramp = 0.0f;
      }
      weights[i][j] = ramp;
    }
  }

  std::vector<float> enorm(n_mels);
  for (int i = 0; i < n_mels; ++i) {
    enorm[i] = 2.0f / (freqs[i+2] - freqs[i]);
  }

  for (int i = 0; i < n_mels; ++i) {
    for (size_t j = 0; j < fftfreqs.size(); ++j) {
      weights[i][j] *= enorm[i];
    }
  }

  return weights;
}

std::vector<std::vector<std::complex<float>>> FeatureExtractor::stft(
    const std::vector<float>& input_array,
    int n_fft,
    int hop_length,
    int win_length,
    const std::vector<float>& window,
    bool center
) {
  // This is a simplified C++ implementation of STFT.
  // A full implementation requires a robust FFT library.
  // This is a placeholder for the actual computation.
  std::cout << "Warning: STFT is a placeholder implementation. Use a real FFT library for a production application." << std::endl;
  std::vector<std::vector<std::complex<float>>> result;
  return result;
}

Matrix FeatureExtractor::compute_mel_spectrogram(
    const std::vector<float>& waveform,
    int padding,
    std::optional<int> chunk_length
) {
  if (chunk_length) {
    n_samples = chunk_length.value() * sampling_rate;
    nb_max_frames = n_samples / hop_length;
  }

  std::vector<float> processed_waveform = waveform;
  if (padding > 0) {
    processed_waveform.insert(processed_waveform.end(), padding, 0.0f);
  }

  std::vector<float> window(n_fft);
  for (int i = 0; i < n_fft; ++i) {
    window[i] = 0.5f * (1.0f - cos(2.0f * M_PI * i / (n_fft - 1)));
  }

  auto stft_output = stft(
      processed_waveform,
      n_fft,
      hop_length,
      n_fft,
      window,
      true
  );

  Matrix magnitudes(stft_output.size(), std::vector<float>(stft_output[0].size()));
  for (size_t i = 0; i < stft_output.size(); ++i) {
    for (size_t j = 0; j < stft_output[i].size(); ++j) {
      magnitudes[i][j] = std::norm(stft_output[i][j]);
    }
  }

  // Perform matrix multiplication: mel_filters @ magnitudes
  Matrix mel_spec(mel_filters.size(), std::vector<float>(magnitudes.size()));
  for (size_t i = 0; i < mel_filters.size(); ++i) {
    for (size_t j = 0; j < magnitudes.size(); ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < mel_filters[0].size(); ++k) {
        sum += mel_filters[i][k] * magnitudes[j][k];
      }
      mel_spec[i][j] = sum;
    }
  }

  Matrix log_spec(mel_spec.size(), std::vector<float>(mel_spec[0].size()));
  for (size_t i = 0; i < mel_spec.size(); ++i) {
    for (size_t j = 0; j < mel_spec[i].size(); ++j) {
      float value = std::max(mel_spec[i][j], 1e-10f);
      log_spec[i][j] = log10(value);
    }
  }

  float max_log = -8.0f; // Simplified max for a reasonable baseline
  if (!log_spec.empty() && !log_spec[0].empty()) {
    max_log = log_spec[0][0];
    for (size_t i = 0; i < log_spec.size(); ++i) {
      for (size_t j = 0; j < log_spec[i].size(); ++j) {
        if (log_spec[i][j] > max_log) {
          max_log = log_spec[i][j];
        }
      }
    }
  }

  for (size_t i = 0; i < log_spec.size(); ++i) {
    for (size_t j = 0; j < log_spec[i].size(); ++j) {
      log_spec[i][j] = (std::max(log_spec[i][j], max_log - 8.0f) + 4.0f) / 4.0f;
    }
  }

  return log_spec;
}
