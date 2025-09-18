#ifndef AUDIO_DECODER_H
#define AUDIO_DECODER_H

#include <string>
#include <vector>
#include <istream>
#include <utility>
#include <optional>

class AudioDecoder {
public:
  // Decodes audio from a file or input stream.
  // This is a conceptual implementation as the actual FFmpeg C API calls are complex.
  static std::vector<float> decode_audio(
      const std::string& input_file,
      int sampling_rate = 16000
  );

  static std::pair<std::vector<float>, std::vector<float>> decode_audio_split_stereo(
      const std::string& input_file,
      int sampling_rate = 16000
  );

  // Pads or trims a 1D vector to a specific length.
  static std::vector<float> pad_or_trim(
      const std::vector<float>& array,
      size_t length = 3000
  );

private:
  // Helper methods that would wrap FFmpeg's low-level functions.
  // These are placeholders and not functionally implemented.
  static void _ignore_invalid_frames();
  static void _group_frames();
  static void _resample_frames();
};

#endif // AUDIO_DECODER_H
