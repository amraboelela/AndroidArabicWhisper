#ifndef FFT_H
#define FFT_H

#include <vector>
#include <complex>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace whisper {

class FFT {
public:
    // Check if n is a power of 2
    static bool is_power_of_2(size_t n) {
        return n > 0 && (n & (n - 1)) == 0;
    }

    // Compute FFT using Cooley-Tukey algorithm (power of 2) or DFT (arbitrary size)
    static std::vector<std::complex<float>> compute(const std::vector<float>& input) {
        size_t n = input.size();

        // Convert input to complex double for better precision
        std::vector<std::complex<double>> x(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = std::complex<double>(input[i], 0.0);
        }

        // Use FFT if power of 2, otherwise use DFT
        if (is_power_of_2(n)) {
            fft_recursive_double(x);
        } else {
            x = dft_double(x);
        }

        // Convert back to float
        std::vector<std::complex<float>> result(n);
        for (size_t i = 0; i < n; ++i) {
            result[i] = std::complex<float>(static_cast<float>(x[i].real()), static_cast<float>(x[i].imag()));
        }

        return result;
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
    // Direct DFT computation for arbitrary sizes (double precision)
    static std::vector<std::complex<double>> dft_double(const std::vector<std::complex<double>>& x) {
        size_t n = x.size();
        std::vector<std::complex<double>> result(n);

        for (size_t k = 0; k < n; ++k) {
            std::complex<double> sum(0.0, 0.0);
            for (size_t t = 0; t < n; ++t) {
                double angle = -2.0 * M_PI * k * t / n;
                std::complex<double> twiddle(std::cos(angle), std::sin(angle));
                sum += x[t] * twiddle;
            }
            result[k] = sum;
        }

        return result;
    }

    // Direct DFT computation for arbitrary sizes
    static std::vector<std::complex<float>> dft(const std::vector<std::complex<float>>& x) {
        size_t n = x.size();
        std::vector<std::complex<float>> result(n);

        for (size_t k = 0; k < n; ++k) {
            std::complex<float> sum(0.0f, 0.0f);
            for (size_t t = 0; t < n; ++t) {
                float angle = -2.0f * M_PI * k * t / n;
                std::complex<float> twiddle(std::cos(angle), std::sin(angle));
                sum += x[t] * twiddle;
            }
            result[k] = sum;
        }

        return result;
    }

    static void fft_recursive_double(std::vector<std::complex<double>>& x) {
        size_t n = x.size();

        if (n <= 1) return;

        // Divide
        std::vector<std::complex<double>> even(n / 2);
        std::vector<std::complex<double>> odd(n / 2);

        for (size_t i = 0; i < n / 2; ++i) {
            even[i] = x[i * 2];
            odd[i] = x[i * 2 + 1];
        }

        // Conquer
        fft_recursive_double(even);
        fft_recursive_double(odd);

        // Combine
        for (size_t k = 0; k < n / 2; ++k) {
            double angle = -2.0 * M_PI * k / n;
            std::complex<double> t = std::polar(1.0, angle) * odd[k];
            x[k] = even[k] + t;
            x[k + n / 2] = even[k] - t;
        }
    }

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
