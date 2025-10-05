#ifndef FFT_H
#define FFT_H

#include <vector>
#include <complex>
#include <cmath>

namespace whisper {

class FFT {
public:
    // Compute FFT using Cooley-Tukey algorithm
    static std::vector<std::complex<float>> compute(const std::vector<float>& input) {
        size_t n = input.size();

        // Convert input to complex
        std::vector<std::complex<float>> x(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = std::complex<float>(input[i], 0.0f);
        }

        // Perform FFT
        fft_recursive(x);

        return x;
    }

    // Compute real FFT (returns only positive frequencies)
    static std::vector<std::complex<float>> rfft(const std::vector<float>& input) {
        auto full_fft = compute(input);
        size_t n = input.size();
        size_t rfft_size = n / 2 + 1;

        std::vector<std::complex<float>> result(rfft_size);
        for (size_t i = 0; i < rfft_size; ++i) {
            result[i] = full_fft[i];
        }

        return result;
    }

private:
    static void fft_recursive(std::vector<std::complex<float>>& x) {
        size_t n = x.size();

        if (n <= 1) return;

        // Divide
        std::vector<std::complex<float>> even(n / 2);
        std::vector<std::complex<float>> odd(n / 2);

        for (size_t i = 0; i < n / 2; ++i) {
            even[i] = x[i * 2];
            odd[i] = x[i * 2 + 1];
        }

        // Conquer
        fft_recursive(even);
        fft_recursive(odd);

        // Combine
        for (size_t k = 0; k < n / 2; ++k) {
            float angle = -2.0f * M_PI * k / n;
            std::complex<float> t = std::polar(1.0f, angle) * odd[k];
            x[k] = even[k] + t;
            x[k + n / 2] = even[k] - t;
        }
    }
};

} // namespace whisper

#endif // FFT_H
