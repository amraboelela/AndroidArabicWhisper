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

    // Compute FFT using Cooley-Tukey algorithm (power of 2) or Bluestein's algorithm (arbitrary size)
    static std::vector<std::complex<float>> compute(const std::vector<float>& input) {
        size_t n = input.size();

        // Convert input to complex double for better precision
        std::vector<std::complex<double>> x(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = std::complex<double>(input[i], 0.0);
        }

        // Use FFT if power of 2, otherwise use Bluestein's algorithm
        if (is_power_of_2(n)) {
            fft_recursive_double(x);
        } else {
            x = fft_bluestein(x);
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
    // Bluestein's algorithm for arbitrary-size FFT (O(N log N))
    // This converts an arbitrary DFT into circular convolution, which can be computed using power-of-2 FFT
    static std::vector<std::complex<double>> fft_bluestein(const std::vector<std::complex<double>>& x) {
        size_t n = x.size();

        // Find next power of 2 that's >= 2*n - 1
        size_t m = 1;
        while (m < 2 * n - 1) {
            m *= 2;
        }

        // Compute the chirp sequence: exp(-i * pi * k^2 / n)
        std::vector<std::complex<double>> chirp(n);
        for (size_t k = 0; k < n; ++k) {
            double angle = -M_PI * k * k / n;
            chirp[k] = std::complex<double>(std::cos(angle), std::sin(angle));
        }

        // Compute a = x * chirp (element-wise multiplication)
        std::vector<std::complex<double>> a(m, 0.0);
        for (size_t k = 0; k < n; ++k) {
            a[k] = x[k] * chirp[k];
        }

        // Compute b = conj(chirp) padded and wrapped
        std::vector<std::complex<double>> b(m, 0.0);
        b[0] = std::conj(chirp[0]);
        for (size_t k = 1; k < n; ++k) {
            b[k] = std::conj(chirp[k]);
            b[m - k] = std::conj(chirp[k]);
        }

        // Compute FFT of a and b
        fft_recursive_double(a);
        fft_recursive_double(b);

        // Pointwise multiply: c = a * b
        std::vector<std::complex<double>> c(m);
        for (size_t i = 0; i < m; ++i) {
            c[i] = a[i] * b[i];
        }

        // Compute inverse FFT of c
        // For inverse FFT: conjugate, FFT, conjugate, divide by m
        for (size_t i = 0; i < m; ++i) {
            c[i] = std::conj(c[i]);
        }
        fft_recursive_double(c);
        for (size_t i = 0; i < m; ++i) {
            c[i] = std::conj(c[i]) / static_cast<double>(m);
        }

        // Extract result and multiply by chirp
        std::vector<std::complex<double>> result(n);
        for (size_t k = 0; k < n; ++k) {
            result[k] = c[k] * chirp[k];
        }

        return result;
    }

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
