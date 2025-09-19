#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <vector>
#include <complex>

// A simple 2D vector to represent a matrix, analogous to a NumPy array.
using Matrix = std::vector<std::vector<float>>;

class FeatureExtractor {
public:
  // C++ constructor to match the Python `__init__`
  FeatureExtractor(
      int feature_size = 80,
      int sampling_rate = 16000,
      int hop_length = 160,
      int chunk_length = 30,
      int n_fft = 400
  );

  // C++ equivalent of the `__call__` method.
  Matrix compute_mel_spectrogram(
      const std::vector<float>& waveform,
      int padding = 160,
      std::optional<int> chunk_length = std::nullopt
  );

  int n_fft;
  int hop_length;
  int chunk_length;
  int n_samples;
  int nb_max_frames;
  float time_per_frame;
  int sampling_rate;
  Matrix mel_filters;

  // Static helper methods, equivalent to Python's @staticmethod
  static Matrix get_mel_filters(int sr, int n_fft, int n_mels);

  static std::vector<std::vector<std::complex<float>>> stft(
      const std::vector<float>& input_array,
      int n_fft,
      int hop_length,
      int win_length,
      const std::vector<float>& window,
      bool center = true
  );
};

#endif // FEATURE_EXTRACTOR_H
