#include "whisper_audio.h"
#include <fstream>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <cmath>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace whisper {

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

  // Normalize audio
  audio = normalize_audio(audio);

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

std::vector<float> AudioProcessor::pad_or_trim(const std::vector<float>& audio, size_t length) {
  if (audio.size() == length) {
      return audio;
  } else if (audio.size() > length) {
      // Trim
      return std::vector<float>(audio.begin(), audio.begin() + length);
  } else {
      // Pad with zeros
      std::vector<float> padded = audio;
      padded.resize(length, 0.0f);
      return padded;
  }
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
  // Apply pre-emphasis
  auto filtered_audio = apply_preemphasis(audio);

  // Compute STFT
  auto stft = compute_stft(filtered_audio);

  // Apply mel filter bank
  auto mel_filters = get_mel_filter_bank();

  // Apply mel filters to STFT magnitude
  std::vector<std::vector<float>> mel_spec(WHISPER_N_MEL);
  for (int mel = 0; mel < WHISPER_N_MEL; ++mel) {
      mel_spec[mel].resize(stft.size());
      for (size_t frame = 0; frame < stft.size(); ++frame) {
      float mel_value = 0.0f;
      for (size_t freq = 0; freq < stft[frame].size() && freq < mel_filters[mel].size(); ++freq) {
          mel_value += stft[frame][freq] * mel_filters[mel][freq];
      }
      mel_spec[mel][frame] = mel_value;
      }
  }

  return mel_spec;
}

std::vector<std::vector<float>> AudioProcessor::apply_log_transform(const std::vector<std::vector<float>>& mel_spectrogram) {
  std::vector<std::vector<float>> log_mel_spec = mel_spectrogram;

  for (auto& mel_band : log_mel_spec) {
      for (float& value : mel_band) {
      value = std::log(std::max(value, 1e-10f)); // Avoid log(0)
      }
  }

  return log_mel_spec;
}

std::vector<std::vector<float>> AudioProcessor::compute_stft(const std::vector<float>& audio) {
  const int window_size = WHISPER_N_FFT;
  const int hop_size = WHISPER_HOP_LENGTH;

  auto window = apply_hann_window(window_size);

  // Calculate number of frames
  int num_frames = (audio.size() - window_size) / hop_size + 1;
  if (num_frames <= 0) num_frames = 1;

  std::vector<std::vector<float>> stft_magnitude(num_frames);

  for (int frame = 0; frame < num_frames; ++frame) {
      int start_idx = frame * hop_size;
      stft_magnitude[frame].resize(window_size / 2 + 1);

      // Simple magnitude computation (not full FFT)
      // This is a simplified version - for production use a proper FFT library
      for (int freq = 0; freq < window_size / 2 + 1; ++freq) {
      float real = 0.0f, imag = 0.0f;

      for (int n = 0; n < window_size && start_idx + n < audio.size(); ++n) {
          float windowed_sample = audio[start_idx + n] * window[n];
          float angle = -2.0f * M_PI * freq * n / window_size;
          real += windowed_sample * std::cos(angle);
          imag += windowed_sample * std::sin(angle);
      }

      stft_magnitude[frame][freq] = std::sqrt(real * real + imag * imag);
      }
  }

  return stft_magnitude;
}

std::vector<float> AudioProcessor::apply_hann_window(int window_size) {
  std::vector<float> window(window_size);
  for (int i = 0; i < window_size; ++i) {
      window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (window_size - 1)));
  }
  return window;
}

std::vector<std::vector<float>> AudioProcessor::get_mel_filter_bank() {
  std::vector<std::vector<float>> mel_filters(WHISPER_N_MEL);

  // Create mel filter bank
  float mel_low = hz_to_mel(0.0f);
  float mel_high = hz_to_mel(WHISPER_SAMPLE_RATE / 2.0f);

  // Create equally spaced mel points
  std::vector<float> mel_points(WHISPER_N_MEL + 2);
  for (int i = 0; i < WHISPER_N_MEL + 2; ++i) {
      mel_points[i] = mel_low + (mel_high - mel_low) * i / (WHISPER_N_MEL + 1);
  }

  // Convert mel points back to Hz
  std::vector<float> hz_points(WHISPER_N_MEL + 2);
  for (int i = 0; i < WHISPER_N_MEL + 2; ++i) {
      hz_points[i] = mel_to_hz(mel_points[i]);
  }

  // Convert Hz to FFT bin numbers
  std::vector<int> bin_points(WHISPER_N_MEL + 2);
  for (int i = 0; i < WHISPER_N_MEL + 2; ++i) {
      bin_points[i] = static_cast<int>(std::floor((WHISPER_N_FFT + 1) * hz_points[i] / WHISPER_SAMPLE_RATE));
  }

  // Create triangular filters
  for (int mel = 0; mel < WHISPER_N_MEL; ++mel) {
      mel_filters[mel].resize(WHISPER_N_FFT / 2 + 1, 0.0f);

      int left = bin_points[mel];
      int center = bin_points[mel + 1];
      int right = bin_points[mel + 2];

      // Left slope
      for (int bin = left; bin < center; ++bin) {
      if (bin < mel_filters[mel].size()) {
          mel_filters[mel][bin] = static_cast<float>(bin - left) / (center - left);
      }
      }

      // Right slope
      for (int bin = center; bin < right; ++bin) {
      if (bin < mel_filters[mel].size()) {
          mel_filters[mel][bin] = static_cast<float>(right - bin) / (right - center);
      }
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