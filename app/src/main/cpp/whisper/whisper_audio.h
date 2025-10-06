#ifndef WHISPER_AUDIO_H
#define WHISPER_AUDIO_H

#include <vector>
#include <string>
#include <cstdint>
#include <cmath>

// Constants matching whisper.cpp expectations
constexpr int WHISPER_SAMPLE_RATE = 16000;
constexpr int WHISPER_N_FFT = 400;
constexpr int WHISPER_HOP_LENGTH = 160;
constexpr int WHISPER_CHUNK_SIZE = 30 * WHISPER_SAMPLE_RATE; // 30 seconds
constexpr int WHISPER_N_MEL = 80;

namespace whisper {

/**
 * Audio preprocessing utilities compatible with whisper.cpp expectations
 */
class AudioProcessor {
public:
  /**
   * Decode audio from file
   * @param input_file Path to audio file
   * @param sampling_rate Target sampling rate (default 16kHz)
   * @param split_stereo Whether to split stereo channels
   * @return Vector of float samples at specified sample rate
   */
  static std::vector<float> decode_audio(const std::string& input_file, int sampling_rate = WHISPER_SAMPLE_RATE, bool split_stereo = false);

  /**
   * Load audio from file and convert to whisper-compatible format
   * @param filename Path to audio file
   * @return Vector of float samples at 16kHz mono
   */
  static std::vector<float> load_audio(const std::string& filename);

  /**
   * Resample audio to 16kHz if needed
   * @param audio Input audio samples
   * @param input_sample_rate Original sample rate
   * @return Resampled audio at 16kHz
   */
  static std::vector<float> resample_audio(const std::vector<float>& audio, int input_sample_rate);

  /**
   * Convert stereo to mono by averaging channels
   * @param stereo_audio Interleaved stereo samples
   * @return Mono audio samples
   */
  static std::vector<float> stereo_to_mono(const std::vector<float>& stereo_audio);

  /**
   * Normalize audio to [-1, 1] range
   * @param audio Input audio
   * @return Normalized audio
   */
  static std::vector<float> normalize_audio(const std::vector<float>& audio);

  /**
   * Apply pre-emphasis filter (high-pass filter)
   * @param audio Input audio
   * @param alpha Pre-emphasis coefficient (typically 0.97)
   * @return Filtered audio
   */
  static std::vector<float> apply_preemphasis(const std::vector<float>& audio, float alpha = 0.97f);

  /**
   * Extract mel spectrogram features compatible with whisper models
   * @param audio Input audio samples at 16kHz
   * @return Mel spectrogram matrix [n_mels, n_frames]
   */
  static std::vector<std::vector<float>> extract_mel_spectrogram(const std::vector<float>& audio);

  /**
   * Apply log mel spectrogram transformation
   * @param mel_spectrogram Input mel spectrogram
   * @return Log mel spectrogram
   */
  static std::vector<std::vector<float>> apply_log_transform(const std::vector<std::vector<float>>& mel_spectrogram);

  /**
   * Apply Hann window function
   * @param window_size Size of the window
   * @return Hann window coefficients
   */
  static std::vector<float> apply_hann_window(int window_size);

private:
  // FFT and STFT utilities
  static std::vector<std::vector<float>> compute_stft(const std::vector<float>& audio);
  static std::vector<std::vector<float>> get_mel_filter_bank();

  // Helper functions
  static float hz_to_mel(float hz);
  static float mel_to_hz(float mel);
};

/**
 * Simple WAV file reader for basic audio loading
 */
class WavReader {
public:
  struct WavHeader {
      uint32_t sample_rate;
      uint16_t num_channels;
      uint16_t bits_per_sample;
      uint32_t data_size;
  };

  static bool read_wav_file(const std::string& filename, std::vector<float>& audio, WavHeader& header);

private:
  static int16_t bytes_to_int16(const uint8_t* bytes);
  static uint32_t bytes_to_uint32(const uint8_t* bytes);
  static uint16_t bytes_to_uint16(const uint8_t* bytes);
};

} // namespace whisper

#endif // WHISPER_AUDIO_H