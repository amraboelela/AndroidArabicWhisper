#include "feature_extractor.h"
#include "whisper_audio.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <complex>
#include <algorithm>
#include <optional>
#include <chrono>
#include <ctime>
#include <sstream>

// Helper function to log with timestamp
std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto value = now_ms.time_since_epoch();
    auto duration = value.count();

    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* local_time = std::localtime(&now_time);

    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(2) << local_time->tm_hour << ":"
        << std::setfill('0') << std::setw(2) << local_time->tm_min << ":"
        << std::setfill('0') << std::setw(2) << local_time->tm_sec << "."
        << std::setfill('0') << std::setw(3) << (duration % 1000);
    return oss.str();
}

void logFeatureTimestamp(const std::string& message) {
    std::cout << "[" << getTimestamp() << "] " << message << std::endl;
}

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
    sampling_rate_(sampling_rate)
{
  n_samples = chunk_length * sampling_rate_;
  nb_max_frames_ = n_samples / hop_length;
  time_per_frame_ = (float)hop_length / sampling_rate_;
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
  // Simplified STFT implementation
  // For production, use a proper FFT library like FFTW or Kiss FFT

  if (input_array.empty()) {
    return std::vector<std::vector<std::complex<float>>>();
  }

  // Calculate number of frames
  int n_frames = 1 + (input_array.size() - n_fft) / hop_length;
  if (n_frames <= 0) n_frames = 1;

  // Frequency bins (n_fft/2 + 1 for real FFT)
  int n_freq_bins = n_fft / 2 + 1;

  // Initialize result matrix [freq_bins][time_frames]
  std::vector<std::vector<std::complex<float>>> result(
      n_freq_bins,
      std::vector<std::complex<float>>(n_frames, std::complex<float>(0.0f, 0.0f))
  );

  // Simple DFT implementation (inefficient but functional)
  for (int frame = 0; frame < n_frames; ++frame) {
    int start_idx = frame * hop_length;

    // Extract frame with windowing
    std::vector<float> frame_data(n_fft, 0.0f);
    for (int i = 0; i < n_fft && (start_idx + i) < static_cast<int>(input_array.size()); ++i) {
      float win_val = (i < static_cast<int>(window.size())) ? window[i] : 1.0f;
      frame_data[i] = input_array[start_idx + i] * win_val;
    }

    // Compute DFT for positive frequencies only (real FFT)
    for (int k = 0; k < n_freq_bins; ++k) {
      std::complex<float> sum(0.0f, 0.0f);

      for (int n = 0; n < n_fft; ++n) {
        float angle = -2.0f * M_PI * k * n / n_fft;
        std::complex<float> twiddle(cos(angle), sin(angle));
        sum += frame_data[n] * twiddle;
      }

      result[k][frame] = sum;
    }
  }

  return result;
}

Matrix FeatureExtractor::compute_mel_spectrogram(
    const std::vector<float>& waveform,
    int padding,
    std::optional<int> chunk_length
) {
  logFeatureTimestamp("Starting feature extraction");
  // logFeatureTimestamp("DEBUG: feature_extractor.__call__ called");
  // std::cout << "  Input waveform shape: (" << waveform.size() << ",)" << std::endl;
  // std::cout << "  Padding: " << padding << std::endl;
  // std::cout << "  Chunk length: " << (chunk_length.has_value() ? std::to_string(chunk_length.value()) : "None") << std::endl;

  // Handle chunking if specified
  std::vector<float> audio_to_process = waveform;

  if (chunk_length.has_value()) {
    // Chunk the audio to the specified length in seconds
    int max_samples = chunk_length.value() * sampling_rate_;
    if (static_cast<int>(waveform.size()) > max_samples) {
      audio_to_process = std::vector<float>(waveform.begin(), waveform.begin() + max_samples);
    }
  }

  // Apply padding (matches Python's np.pad(waveform, (0, padding)))
  if (padding > 0) {
    audio_to_process.insert(audio_to_process.end(), padding, 0.0f);
    // std::cout << "  After padding shape: (" << audio_to_process.size() << ",)" << std::endl;
  }

  // Log audio statistics before STFT
  // float audio_min = *std::min_element(audio_to_process.begin(), audio_to_process.end());
  // float audio_max = *std::max_element(audio_to_process.begin(), audio_to_process.end());
  // double audio_sum = std::accumulate(audio_to_process.begin(), audio_to_process.end(), 0.0);
  // float audio_mean = audio_sum / audio_to_process.size();
  // std::cout << "  Audio stats before STFT: min=" << std::fixed << std::setprecision(6)
  //           << audio_min << ", max=" << audio_max << ", mean=" << audio_mean << std::endl;

  // Print first 10 audio samples
  // std::cout << "  First 10 audio samples: [";
  // for (size_t i = 0; i < std::min(size_t(10), audio_to_process.size()); ++i) {
  //   if (i > 0) std::cout << " ";
  //   // Use Python-like compact float format
  //   float val = audio_to_process[i];
  //   if (val == 0.0f) {
  //     std::cout << "0.";
  //   } else {
  //     std::cout << val;
  //   }
  // }
  // std::cout << "]" << std::endl;

  // Log window statistics before STFT (to match Python's order)
  auto window = whisper::AudioProcessor::apply_hann_window(WHISPER_N_FFT);
  // float win_min = *std::min_element(window.begin(), window.end());
  // float win_max = *std::max_element(window.begin(), window.end());
  // double win_sum = std::accumulate(window.begin(), window.end(), 0.0);
  // float win_mean = win_sum / window.size();
  // std::cout << "  Window stats: min=" << std::fixed << std::setprecision(6) << win_min << ", max=" << win_max << ", mean=" << win_mean << std::endl;

  // Print first 10 window values
  // std::cout << "  First 10 window values: [";
  // for (size_t i = 0; i < std::min(size_t(10), window.size()); ++i) {
  //   if (i > 0) std::cout << " ";
  //   // Add line break after 5th value to match Python's display
  //   if (i == 5) std::cout << "\n ";
  //   std::cout << std::scientific << std::setprecision(7) << window[i];
  // }
  // std::cout << "]" << std::endl;
  // std::cout << std::fixed; // Reset to fixed notation

  // Use whisper-compatible mel spectrogram extraction
  auto whisper_mel_spec = whisper::AudioProcessor::extract_mel_spectrogram(audio_to_process);

  if (whisper_mel_spec.empty()) {
    std::cerr << "Failed to extract mel spectrogram using whisper audio processing" << std::endl;
    // Fall back to original implementation
    return compute_mel_spectrogram_original(waveform, padding, chunk_length);
  }

  // Apply log transform for whisper compatibility
  auto log_mel_spec = whisper::AudioProcessor::apply_log_transform(whisper_mel_spec);

  // Apply normalization matching Python's faster-whisper implementation:
  // log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
  // log_spec = (log_spec + 4.0) / 4.0

  // Find max value before normalization
  float max_val = -std::numeric_limits<float>::infinity();
  float min_val = std::numeric_limits<float>::infinity();
  for (const auto& row : log_mel_spec) {
    for (float val : row) {
      max_val = std::max(max_val, val);
      min_val = std::min(min_val, val);
    }
  }

  // std::cout << "Before normalization - max: " << max_val << ", min: " << min_val << std::endl;

  // Apply clamping first (before scaling)
  for (auto& row : log_mel_spec) {
    for (float& val : row) {
      val = std::max(val, max_val - 8.0f);  // Clamp to max - 8
    }
  }

  // Log statistics after clamping (before scaling)
  float clamp_min = std::numeric_limits<float>::infinity();
  float clamp_max = -std::numeric_limits<float>::infinity();
  double clamp_sum = 0.0;
  size_t clamp_count = 0;
  for (const auto& row : log_mel_spec) {
    for (float val : row) {
      clamp_min = std::min(clamp_min, val);
      clamp_max = std::max(clamp_max, val);
      clamp_sum += val;
      clamp_count++;
    }
  }
  float clamp_mean = clamp_sum / clamp_count;
  // std::cout << "  After clamping stats: min=" << std::fixed << std::setprecision(6) << clamp_min << ", max=" << clamp_max << ", mean=" << clamp_mean << std::endl;

  // Apply final scaling
  for (auto& row : log_mel_spec) {
    for (float& val : row) {
      val = (val + 4.0f) / 4.0f;            // Scale to [-1, ~1.5] range
    }
  }

  // Log final shape after normalization
  // std::cout << "  Final log_spec shape: (" << log_mel_spec.size() << ", "
  //           << (log_mel_spec.empty() ? 0 : log_mel_spec[0].size()) << ")" << std::endl;

  // Log final statistics
  // float final_min = std::numeric_limits<float>::infinity();
  // float final_max = -std::numeric_limits<float>::infinity();
  // double final_sum = 0.0;
  // size_t final_count = 0;
  // for (const auto& row : log_mel_spec) {
  //   for (float val : row) {
  //     final_min = std::min(final_min, val);
  //     final_max = std::max(final_max, val);
  //     final_sum += val;
  //     final_count++;
  //   }
  // }
  // float final_mean = final_sum / final_count;
  // std::cout << "  Final log_spec stats: min=" << final_min << ", max=" << final_max << ", mean=" << final_mean << std::endl;

  logFeatureTimestamp("Feature extraction completed");
  return log_mel_spec;
}

Matrix FeatureExtractor::compute_mel_spectrogram_original(
    const std::vector<float>& waveform,
    int padding,
    std::optional<int> chunk_length
) {
  // Original implementation as fallback
  if (chunk_length) {
    n_samples = chunk_length.value() * sampling_rate_;
    nb_max_frames_ = n_samples / hop_length;
  }

  std::vector<float> processed_waveform = waveform;
  if (padding > 0) {
    processed_waveform.insert(processed_waveform.end(), padding, 0.0f);
  }

  std::vector<float> window(n_fft);
  for (int i = 0; i < n_fft; ++i) {
    window[i] = 0.5f * (1.0f - cos(2.0f * M_PI * i / (n_fft - 1)));
  }

  logFeatureTimestamp("Starting STFT computation");
  auto stft_output = stft(
      processed_waveform,
      n_fft,
      hop_length,
      n_fft,
      window,
      true
  );

  if (stft_output.empty()) {
    std::cerr << "STFT computation failed, returning empty matrix" << std::endl;
    return Matrix();
  }

  Matrix magnitudes(stft_output.size(), std::vector<float>(stft_output[0].size()));
  for (size_t i = 0; i < stft_output.size(); ++i) {
    for (size_t j = 0; j < stft_output[i].size(); ++j) {
      magnitudes[i][j] = std::norm(stft_output[i][j]);
    }
  }

  logFeatureTimestamp("STFT completed, starting mel filtering");
  // Perform matrix multiplication: mel_filters @ magnitudes
  Matrix mel_spec(mel_filters.size(), std::vector<float>(magnitudes.size()));
  for (size_t i = 0; i < mel_filters.size(); ++i) {
    for (size_t j = 0; j < magnitudes.size(); ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < mel_filters[0].size() && k < magnitudes[0].size(); ++k) {
        sum += mel_filters[i][k] * magnitudes[j][k];
      }
      mel_spec[i][j] = sum;
    }
  }

  // Apply log transform with proper clamping
  Matrix log_spec(mel_spec.size(), std::vector<float>(mel_spec[0].size()));
  for (size_t i = 0; i < mel_spec.size(); ++i) {
    for (size_t j = 0; j < mel_spec[i].size(); ++j) {
      // Clamp to avoid log(0) and apply log10
      float value = std::max(mel_spec[i][j], 1e-10f);
      log_spec[i][j] = log10(value);
    }
  }

  // Find max value for normalization
  float max_log = -10.0f;
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

  // Normalize to reasonable range for whisper compatibility
  // Typical range: [max_log - 8, max_log] -> [-8, 0] after normalization
  for (size_t i = 0; i < log_spec.size(); ++i) {
    for (size_t j = 0; j < log_spec[i].size(); ++j) {
      // Clamp to dynamic range of 8 dB
      log_spec[i][j] = std::max(log_spec[i][j], max_log - 8.0f);
      // Normalize to [0, 8] then shift to [-8, 0]
      log_spec[i][j] = log_spec[i][j] - max_log;
    }
  }

  return log_spec;
}
