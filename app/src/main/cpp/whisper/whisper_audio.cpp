#include "whisper_audio.h"
#include "fft.h"
#include <fstream>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <chrono>
#include <ctime>
#include <sstream>
#include <android/log.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define LOG_TAG "#whisper-onnx"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

// Helper function to log with timestamp
std::string getAudioTimestamp() {
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

void logAudioTimestamp(const std::string& message) {
    std::cout << "[" << getAudioTimestamp() << "] " << message << std::endl;
}

namespace whisper {

std::vector<float> AudioProcessor::decode_audio(const std::string& input_file, int sampling_rate, bool split_stereo) {
  WavReader::WavHeader header;
  std::vector<float> audio;

  if (!WavReader::read_wav_file(input_file, audio, header)) {
      std::cerr << "Failed to load audio file: " << input_file << std::endl;
      return {};
  }

  // Convert to mono if stereo (unless split_stereo is requested)
  if (header.num_channels == 2 && !split_stereo) {
      audio = stereo_to_mono(audio);
  }

  // Resample if needed
  if (header.sample_rate != sampling_rate) {
      audio = resample_audio(audio, header.sample_rate);
  }

  return audio;
}

std::vector<float> AudioProcessor::load_audio(const std::string& filename) {
  WavReader::WavHeader header;
  std::vector<float> audio;

  if (!WavReader::read_wav_file(filename, audio, header)) {
      std::cerr << "Failed to load audio file: " << filename << std::endl;
      return {};
  }

  // Convert to mono if stereo
  if (header.num_channels == 2) {
      audio = stereo_to_mono(audio);
  }

  // Resample to 16kHz if needed
  if (header.sample_rate != WHISPER_SAMPLE_RATE) {
      audio = resample_audio(audio, header.sample_rate);
  }

  // DO NOT normalize audio - faster-whisper expects raw audio values
  // The audio is already normalized to [-1, 1] range when converting from int16 PCM

  return audio;
}

std::vector<float> AudioProcessor::resample_audio(const std::vector<float>& audio, int input_sample_rate) {
  if (input_sample_rate == WHISPER_SAMPLE_RATE) {
      return audio;
  }

  // Simple linear interpolation resampling
  double ratio = static_cast<double>(input_sample_rate) / WHISPER_SAMPLE_RATE;
  size_t output_size = static_cast<size_t>(audio.size() / ratio);
  std::vector<float> resampled(output_size);

  for (size_t i = 0; i < output_size; ++i) {
      double src_index = i * ratio;
      size_t index = static_cast<size_t>(src_index);
      double frac = src_index - index;

      if (index + 1 < audio.size()) {
      resampled[i] = audio[index] * (1.0f - frac) + audio[index + 1] * frac;
      } else {
      resampled[i] = audio[index];
      }
  }

  return resampled;
}

std::vector<float> AudioProcessor::stereo_to_mono(const std::vector<float>& stereo_audio) {
  std::vector<float> mono_audio;
  mono_audio.reserve(stereo_audio.size() / 2);

  for (size_t i = 0; i < stereo_audio.size(); i += 2) {
      mono_audio.push_back((stereo_audio[i] + stereo_audio[i + 1]) * 0.5f);
  }

  return mono_audio;
}

std::vector<float> AudioProcessor::normalize_audio(const std::vector<float>& audio) {
  if (audio.empty()) return audio;

  // Find max absolute value
  float max_val = 0.0f;
  for (float sample : audio) {
      max_val = std::max(max_val, std::abs(sample));
  }

  if (max_val == 0.0f) {
      return audio; // All zeros, nothing to normalize
  }

  // Normalize to [-1, 1] range
  std::vector<float> normalized;
  normalized.reserve(audio.size());
  for (float sample : audio) {
      normalized.push_back(sample / max_val);
  }

  return normalized;
}

std::vector<float> AudioProcessor::apply_preemphasis(const std::vector<float>& audio, float alpha) {
  if (audio.empty()) return audio;

  std::vector<float> filtered;
  filtered.reserve(audio.size());
  filtered.push_back(audio[0]); // First sample unchanged

  for (size_t i = 1; i < audio.size(); ++i) {
      filtered.push_back(audio[i] - alpha * audio[i - 1]);
  }

  return filtered;
}

std::vector<std::vector<float>> AudioProcessor::extract_mel_spectrogram(const std::vector<float>& audio) {
  auto startTime = std::chrono::high_resolution_clock::now();
  LOGD("========================================");
  LOGD("🎯 C++ preprocessing for %zu samples", audio.size());
  LOGD("========================================");

  // Compute STFT directly (no pre-emphasis to match Python's faster-whisper)
  LOGD("🔧 Computing STFT...");
  auto stftStart = std::chrono::high_resolution_clock::now();
  auto stft = compute_stft(audio);
  auto stftEnd = std::chrono::high_resolution_clock::now();
  auto stftTime = std::chrono::duration_cast<std::chrono::milliseconds>(stftEnd - stftStart).count();

  // Drop the last frame to match Python's behavior (stft[..., :-1])
  // Python intentionally drops the last time frame
  // After transposition, stft is [freq_bins][time_frames]
  if (!stft.empty() && !stft[0].empty()) {
    for (auto& freq_band : stft) {
      freq_band.pop_back();
    }
  }

  // Log STFT magnitudes shape (after dropping last frame, to match Python)
  // stft is now [freq_bins][time_frames]
  size_t n_freq_bins = stft.size();
  size_t n_frames = stft.empty() ? 0 : stft[0].size();
  LOGD("✅ STFT shape: %zu x %zu (%lldms)", n_freq_bins, n_frames, stftTime);

  // Apply mel filter bank
  LOGD("🔧 Applying mel filterbank...");
  auto melStart = std::chrono::high_resolution_clock::now();
  auto mel_filters = get_mel_filter_bank();

  // Apply mel filters to STFT magnitude
  // STFT is now [freq_bins][time_frames], mel_spec should be [mel_bins][time_frames]
  std::vector<std::vector<float>> mel_spec(WHISPER_N_MEL);
  size_t num_time_frames = stft.empty() ? 0 : stft[0].size();

  for (int mel = 0; mel < WHISPER_N_MEL; ++mel) {
      mel_spec[mel].resize(num_time_frames);
      for (size_t frame = 0; frame < num_time_frames; ++frame) {
      float mel_value = 0.0f;
      // Sum over frequency bins: mel_value = sum(mel_filter[freq] * stft[freq][frame])
      for (size_t freq = 0; freq < stft.size() && freq < mel_filters[mel].size(); ++freq) {
          mel_value += mel_filters[mel][freq] * stft[freq][frame];
      }
      mel_spec[mel][frame] = mel_value;
      }
  }
  auto melEnd = std::chrono::high_resolution_clock::now();
  auto melTime = std::chrono::duration_cast<std::chrono::milliseconds>(melEnd - melStart).count();
  LOGD("✅ Mel spectrogram shape: %zu x %zu (%lldms)", mel_spec.size(), (mel_spec.empty() ? 0 : mel_spec[0].size()), melTime);

  // Apply log transform
  LOGD("🔧 Applying log10 transform...");
  auto logStart = std::chrono::high_resolution_clock::now();
  auto log_mel_spec = apply_log_transform(mel_spec);
  auto logEnd = std::chrono::high_resolution_clock::now();
  auto logTime = std::chrono::duration_cast<std::chrono::milliseconds>(logEnd - logStart).count();
  LOGD("✅ Log-mel shape: %zu x %zu (%lldms)", log_mel_spec.size(), (log_mel_spec.empty() ? 0 : log_mel_spec[0].size()), logTime);

  // Apply Whisper normalization: (log_mel - mean) / std
  // Whisper uses global statistics from training data
  LOGD("🔧 Applying Whisper normalization...");
  auto normStart = std::chrono::high_resolution_clock::now();
  const float WHISPER_MEL_MEAN = -4.2677393f;  // Global mean from Whisper training
  const float WHISPER_MEL_STD = 4.5689974f;    // Global std from Whisper training

  for (auto& mel_band : log_mel_spec) {
    for (float& value : mel_band) {
      value = (value - WHISPER_MEL_MEAN) / WHISPER_MEL_STD;
    }
  }
  auto normEnd = std::chrono::high_resolution_clock::now();
  auto normTime = std::chrono::duration_cast<std::chrono::milliseconds>(normEnd - normStart).count();
  LOGD("✅ Normalization complete (%lldms)", normTime);

  auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(normEnd - startTime).count();
  LOGD("⏱️ TOTAL C++ preprocessing time: %lldms (STFT: %lldms, Mel: %lldms, Log: %lldms, Norm: %lldms)",
       totalTime, stftTime, melTime, logTime, normTime);
  LOGD("========================================");

  return log_mel_spec;
}

std::vector<std::vector<float>> AudioProcessor::apply_log_transform(const std::vector<std::vector<float>>& mel_spectrogram) {
  std::vector<std::vector<float>> log_mel_spec = mel_spectrogram;

  for (auto& mel_band : log_mel_spec) {
      for (float& value : mel_band) {
      value = std::log10(std::max(value, 1e-10f)); // Use log10 to match Python's np.log10
      }
  }

  return log_mel_spec;
}

std::vector<std::vector<float>> AudioProcessor::compute_stft(const std::vector<float>& audio) {
  const int window_size = WHISPER_N_FFT;
  const int hop_size = WHISPER_HOP_LENGTH;

  auto window = apply_hann_window(window_size);

  // Apply center padding (matches Python's center=True in STFT)
  const int pad_amount = window_size / 2;
  std::vector<float> padded_audio(audio.size() + 2 * pad_amount, 0.0f);
  std::copy(audio.begin(), audio.end(), padded_audio.begin() + pad_amount);

  // Calculate number of frames using padded length
  int num_frames = (padded_audio.size() - window_size) / hop_size + 1;
  if (num_frames <= 0) num_frames = 1;

  // Pre-allocate frame_data outside the loop to avoid repeated allocations
  std::vector<float> frame_data(window_size);

  // Pre-calculate frequency bins size
  const int n_freq_bins = window_size / 2 + 1;

  // Allocate result in final transposed format: [freq_bins, time_frames]
  std::vector<std::vector<float>> stft_magnitude(n_freq_bins, std::vector<float>(num_frames));

  for (int frame = 0; frame < num_frames; ++frame) {
      int start_idx = frame * hop_size;

      // Extract and window the frame (reuse frame_data buffer)
      for (int n = 0; n < window_size && start_idx + n < padded_audio.size(); ++n) {
          frame_data[n] = padded_audio[start_idx + n] * window[n];
      }
      // Zero out any remaining space (if start_idx + n >= padded_audio.size())
      for (int n = std::min(window_size, static_cast<int>(padded_audio.size() - start_idx)); n < window_size; ++n) {
          frame_data[n] = 0.0f;
      }

      // Debug: log first frame data for frame 100
      static bool logged_frame_data = false;
      if (!logged_frame_data && frame == 100) {
        std::cout << "  DEBUG frame 100 input: First 10 values: [";
        for (size_t i = 0; i < std::min(size_t(10), frame_data.size()); ++i) {
          if (i > 0) std::cout << " ";
          // Add line breaks after 4th and 7th values to match Python's display
          if (i == 4 || i == 8) std::cout << "\n ";
          std::cout << std::scientific << std::setprecision(7) << frame_data[i];
        }
        std::cout << std::fixed << "]" << std::endl;
        logged_frame_data = true;
      }

      // Compute FFT using proper FFT implementation
      auto fft_result = FFT::rfft(frame_data);

      // Debug: log first FFT result for first non-zero frame
      static bool logged_fft = false;
      if (!logged_fft && frame == 100) {  // Check frame 100 to avoid all-zero frames
        logAudioTimestamp("DEBUG FFT frame 100: First 5 complex values");
        std::cout << "  DEBUG FFT frame 100: First 5 complex values: [";
        for (size_t i = 0; i < std::min(size_t(5), fft_result.size()); ++i) {
          // Add proper spacing between complex numbers
          if (i > 0) std::cout << " ";
          // Add line break after 3rd value to match Python's display
          if (i == 3) std::cout << "\n ";

          // Format complex numbers to match Python's output
          float real_part = fft_result[i].real();
          float imag_part = fft_result[i].imag();

          // Format real part with appropriate width and precision
          std::cout << std::setw(12) << std::fixed << std::setprecision(8) << real_part;

          // Format imaginary part
          if (std::abs(imag_part) < 1e-9f) {
            // For zero imaginary part, use "0.j" format like Python
            std::cout << "+0.j";
          } else {
            // For non-zero imaginary part, add sign and value
            if (imag_part >= 0) {
              std::cout << "+";
            }
            std::cout << std::setprecision(8) << imag_part << "j";
          }
        }
        std::cout << "]" << std::endl;
        logged_fft = true;
      }

      // Compute magnitude squared and store directly in transposed format
      for (size_t i = 0; i < fft_result.size(); ++i) {
          float mag = std::abs(fft_result[i]);
          stft_magnitude[i][frame] = mag * mag;  // Square the magnitude, store in [freq][frame]
      }
  }

  return stft_magnitude;
}

std::vector<float> AudioProcessor::apply_hann_window(int window_size) {
  // Match Python's np.hanning(n_fft + 1)[:-1]
  // Create window_size + 1 elements, then drop the last one
  std::vector<float> window_temp(window_size + 1);
  for (int i = 0; i < window_size + 1; ++i) {
      window_temp[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / window_size));
  }

  // Drop the last element
  std::vector<float> window(window_temp.begin(), window_temp.begin() + window_size);
  return window;
}

std::vector<std::vector<float>> AudioProcessor::get_mel_filter_bank() {
  std::vector<std::vector<float>> mel_filters(WHISPER_N_MEL);

  // Match Python's Slaney-style mel scale (not HTK mel scale)
  // Python uses a piecewise linear/logarithmic scale
  const int n_fft = WHISPER_N_FFT;
  const int sr = WHISPER_SAMPLE_RATE;

  // Center freqs of each FFT bin
  std::vector<float> fftfreqs(n_fft / 2 + 1);
  for (int i = 0; i <= n_fft / 2; ++i) {
    fftfreqs[i] = i * sr / static_cast<float>(n_fft);
  }

  // Mel scale parameters (from Python's faster-whisper)
  const float min_mel = 0.0f;
  const float max_mel = 45.245640471924965f;

  // Create equally spaced mel points
  std::vector<float> mels(WHISPER_N_MEL + 2);
  for (int i = 0; i < WHISPER_N_MEL + 2; ++i) {
    mels[i] = min_mel + (max_mel - min_mel) * i / (WHISPER_N_MEL + 1);
  }

  // Fill in the linear scale
  const float f_min = 0.0f;
  const float f_sp = 200.0f / 3.0f;
  std::vector<float> freqs(mels.size());
  for (size_t i = 0; i < mels.size(); ++i) {
    freqs[i] = f_min + f_sp * mels[i];
  }

  // And now the nonlinear scale
  const float min_log_hz = 1000.0f;  // beginning of log region (Hz)
  const float min_log_mel = (min_log_hz - f_min) / f_sp;  // same (Mels)
  const float logstep = std::log(6.4f) / 27.0f;  // step size for log region

  for (size_t i = 0; i < mels.size(); ++i) {
    if (mels[i] >= min_log_mel) {
      freqs[i] = min_log_hz * std::exp(logstep * (mels[i] - min_log_mel));
    }
  }

  // Compute fdiff
  std::vector<float> fdiff(freqs.size() - 1);
  for (size_t i = 0; i < fdiff.size(); ++i) {
    fdiff[i] = freqs[i + 1] - freqs[i];
  }

  // Create ramps matrix: freqs.reshape(-1, 1) - fftfreqs.reshape(1, -1)
  // ramps[i][j] = freqs[i] - fftfreqs[j]
  std::vector<std::vector<float>> ramps(freqs.size(), std::vector<float>(fftfreqs.size()));
  for (size_t i = 0; i < freqs.size(); ++i) {
    for (size_t j = 0; j < fftfreqs.size(); ++j) {
      ramps[i][j] = freqs[i] - fftfreqs[j];
    }
  }

  // Create mel filters
  for (int mel = 0; mel < WHISPER_N_MEL; ++mel) {
    mel_filters[mel].resize(fftfreqs.size(), 0.0f);

    // lower = -ramps[mel] / fdiff[mel]
    // upper = ramps[mel+2] / fdiff[mel+1]
    // weights = max(0, min(lower, upper))

    for (size_t j = 0; j < fftfreqs.size(); ++j) {
      float lower = -ramps[mel][j] / fdiff[mel];
      float upper = ramps[mel + 2][j] / fdiff[mel + 1];
      mel_filters[mel][j] = std::max(0.0f, std::min(lower, upper));
    }

    // Apply Slaney-style normalization
    float enorm = 2.0f / (freqs[mel + 2] - freqs[mel]);
    for (size_t j = 0; j < mel_filters[mel].size(); ++j) {
      mel_filters[mel][j] *= enorm;
    }
  }

  return mel_filters;
}

float AudioProcessor::hz_to_mel(float hz) {
  return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float AudioProcessor::mel_to_hz(float mel) {
  return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

// WavReader implementation
bool WavReader::read_wav_file(const std::string& filename, std::vector<float>& audio, WavHeader& header) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
      return false;
  }

  // Read RIFF header (12 bytes)
  uint8_t riff_header[12];
  file.read(reinterpret_cast<char*>(riff_header), 12);

  if (file.gcount() != 12) {
      return false;
  }

  // Check RIFF header
  if (std::memcmp(riff_header, "RIFF", 4) != 0 || std::memcmp(riff_header + 8, "WAVE", 4) != 0) {
      return false;
  }

  // Initialize header fields
  header.num_channels = 0;
  header.sample_rate = 0;
  header.bits_per_sample = 0;
  header.data_size = 0;

  // Read chunks until we find fmt and data chunks
  bool found_fmt = false;
  bool found_data = false;

  while (!found_fmt || !found_data) {
      // Read chunk header (8 bytes: 4-byte ID + 4-byte size)
      uint8_t chunk_header[8];
      file.read(reinterpret_cast<char*>(chunk_header), 8);

      if (file.gcount() != 8) {
          break; // End of file or error
      }

      uint32_t chunk_size = bytes_to_uint32(chunk_header + 4);

      if (std::memcmp(chunk_header, "fmt ", 4) == 0) {
          // Read fmt chunk data
          if (chunk_size < 16) {
              return false; // Invalid fmt chunk
          }

          uint8_t fmt_data[16];
          file.read(reinterpret_cast<char*>(fmt_data), 16);

          if (file.gcount() != 16) {
              return false;
          }

          // Parse fmt chunk
          uint16_t audio_format = bytes_to_uint16(fmt_data);
          header.num_channels = bytes_to_uint16(fmt_data + 2);
          header.sample_rate = bytes_to_uint32(fmt_data + 4);
          header.bits_per_sample = bytes_to_uint16(fmt_data + 14);

          // Check if it's PCM format
          if (audio_format != 1) {
              return false; // Only support PCM
          }

          found_fmt = true;

          // Skip any remaining bytes in this chunk
          if (chunk_size > 16) {
              file.seekg(chunk_size - 16, std::ios::cur);
          }
      } else if (std::memcmp(chunk_header, "data", 4) == 0) {
          header.data_size = chunk_size;
          found_data = true;

          // Don't skip this chunk - we'll read the data next
          break;
      } else {
          // Skip unknown chunk
          file.seekg(chunk_size, std::ios::cur);
      }

      // Ensure we're aligned to even byte boundary
      if (chunk_size % 2 == 1) {
          file.seekg(1, std::ios::cur);
      }
  }

  if (!found_fmt || !found_data) {
      return false;
  }

  // Read audio data
  size_t num_samples = header.data_size / (header.bits_per_sample / 8);

  // For stereo files, num_samples includes both channels
  // We want the total number of sample values, not sample frames
  audio.resize(num_samples);

  if (header.bits_per_sample == 16) {
      std::vector<int16_t> int16_data(num_samples);
      file.read(reinterpret_cast<char*>(int16_data.data()), header.data_size);

      if (file.gcount() != static_cast<std::streamsize>(header.data_size)) {
          return false;
      }

      // Convert to float [-1, 1]
      for (size_t i = 0; i < num_samples; ++i) {
          audio[i] = static_cast<float>(int16_data[i]) / 32768.0f;
      }
  } else {
      // For simplicity, only support 16-bit WAV files
      return false;
  }

  return true;
}

int16_t WavReader::bytes_to_int16(const uint8_t* bytes) {
  return static_cast<int16_t>(bytes[0] | (bytes[1] << 8));
}

uint32_t WavReader::bytes_to_uint32(const uint8_t* bytes) {
  return bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
}

uint16_t WavReader::bytes_to_uint16(const uint8_t* bytes) {
  return bytes[0] | (bytes[1] << 8);
}

} // namespace whisper
