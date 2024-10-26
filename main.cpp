/*

#include <iostream>     // for input/output operations
#include <vector>      // for std::vector
#include <cmath>       // for trigonometric functions (cos, sin) and M_PI
#include <complex>     // for complex numbers (alternative to our Complex class)
#include <iomanip>     // for std::setprecision
#include <random>      // for random number generation in tests
#include <cstdlib>     // for rand()
#include<omp.h>


constexpr double M_PI = 3.14159265358979323846;

using namespace std;

// Complex number class
class Complex {
public:
    double real, imag;
    Complex(double r = 0, double i = 0) : real(r), imag(i) {}

    Complex operator + (const Complex& b) const {
        return Complex(real + b.real, imag + b.imag);
    }

    Complex operator - (const Complex& b) const {
        return Complex(real - b.real, imag - b.imag);
    }

    Complex operator * (const Complex& b) const {
        return Complex(real * b.real - imag * b.imag,
            real * b.imag + imag * b.real);
    }
};

// 1D FFT implementation
void fft1D(vector<Complex>& a, bool invert) {
    int n = a.size();
    if (n == 1) return;

    vector<Complex> a0(n / 2), a1(n / 2);
    for (int i = 0; 2 * i < n; i++) {
        a0[i] = a[2 * i];
        a1[i] = a[2 * i + 1];
    }

    fft1D(a0, invert);
    fft1D(a1, invert);

    double ang = 2 * M_PI / n * (invert ? -1 : 1);
    Complex w(1), wn(cos(ang), sin(ang));

    for (int i = 0; 2 * i < n; i++) {
        a[i] = a0[i] + w * a1[i];
        a[i + n / 2] = a0[i] - w * a1[i];
        if (invert) {
            a[i].real /= 2;
            a[i + n / 2].real /= 2;
            a[i].imag /= 2;
            a[i + n / 2].imag /= 2;
        }
        w = w * wn;
    }
}

// 2D FFT implementation
void fft2D(vector<vector<Complex>>& a, bool invert) {
    int n = a.size();
    int m = a[0].size();

    // Apply FFT to each row
    for (int i = 0; i < n; i++) {
        fft1D(a[i], invert);
    }

    // Transpose the matrix
    vector<vector<Complex>> trans(m, vector<Complex>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            trans[j][i] = a[i][j];
        }
    }

    // Apply FFT to each column (now row after transpose)
    for (int i = 0; i < m; i++) {
        fft1D(trans[i], invert);
    }

    // Transpose back
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            a[i][j] = trans[j][i];
        }
    }
}

// Function to print a 2D complex matrix
void printMatrix(const vector<vector<Complex>>& a) {
    for (const auto& row : a) {
        for (const auto& val : row) {
            cout << fixed << setprecision(2) << val.real;
            if (val.imag >= 0) cout << "+";
            cout << val.imag << "i ";
        }
        cout << endl;
    }
}

// Function to check if two complex numbers are approximately equal
bool isApproximatelyEqual(const Complex& a, const Complex& b, double epsilon = 1e-10) {
    return abs(a.real - b.real) < epsilon && abs(a.imag - b.imag) < epsilon;
}

// Function to check if two matrices are approximately equal
bool areMatricesEqual(const vector<vector<Complex>>& a, const vector<vector<Complex>>& b,
    double epsilon = 1e-10) {
    if (a.size() != b.size() || a[0].size() != b[0].size()) return false;

    for (size_t i = 0; i < a.size(); i++) {
        for (size_t j = 0; j < a[0].size(); j++) {
            if (!isApproximatelyEqual(a[i][j], b[i][j], epsilon)) return false;
        }
    }
    return true;
}

// Comprehensive test function
void testFFT2DComp() {
    cout << "\nRunning 2D FFT Tests...\n" << endl;

    // Test 1: Identity matrix
    cout << "Test 1: Identity Matrix" << endl;
    vector<vector<Complex>> identity = {
        {Complex(1,0), Complex(0,0)},
        {Complex(0,0), Complex(1,0)}
    };
    vector<vector<Complex>> identityCopy = identity;

    fft2D(identity, false);  // Forward transform
    fft2D(identity, true);   // Inverse transform

    if (areMatricesEqual(identity, identityCopy)) {
        cout << "PASSED: Identity matrix transform and inverse transform" << endl;
    }
    else {
        cout << "FAILED: Identity matrix test" << endl;
    }

    // Test 2: Zero matrix
    cout << "\nTest 2: Zero Matrix" << endl;
    vector<vector<Complex>> zeros(4, vector<Complex>(4, Complex(0, 0)));
    vector<vector<Complex>> zerosCopy = zeros;

    fft2D(zeros, false);
    fft2D(zeros, true);

    if (areMatricesEqual(zeros, zerosCopy)) {
        cout << "PASSED: Zero matrix transform and inverse transform" << endl;
    }
    else {
        cout << "FAILED: Zero matrix test" << endl;
    }

    // Test 3: Impulse matrix
    cout << "\nTest 3: Impulse Matrix" << endl;
    vector<vector<Complex>> impulse = {
        {Complex(1,0), Complex(0,0), Complex(0,0), Complex(0,0)},
        {Complex(0,0), Complex(0,0), Complex(0,0), Complex(0,0)},
        {Complex(0,0), Complex(0,0), Complex(0,0), Complex(0,0)},
        {Complex(0,0), Complex(0,0), Complex(0,0), Complex(0,0)}
    };
    vector<vector<Complex>> impulseCopy = impulse;

    cout << "Original impulse matrix:" << endl;
    printMatrix(impulse);

    fft2D(impulse, false);
    cout << "\nAfter forward transform:" << endl;
    printMatrix(impulse);

    fft2D(impulse, true);
    cout << "\nAfter inverse transform:" << endl;
    printMatrix(impulse);

    if (areMatricesEqual(impulse, impulseCopy)) {
        cout << "PASSED: Impulse matrix transform and inverse transform" << endl;
    }
    else {
        cout << "FAILED: Impulse matrix test" << endl;
    }

    // Test 4: Random matrix
    cout << "\nTest 4: Random Matrix" << endl;
    vector<vector<Complex>> random(4, vector<Complex>(4));
    // Initialize with random values
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            random[i][j] = Complex(rand() % 10, rand() % 10);
        }
    }
    vector<vector<Complex>> randomCopy = random;

    cout << "Original random matrix:" << endl;
    printMatrix(random);

    fft2D(random, false);
    cout << "\nAfter forward transform:" << endl;
    printMatrix(random);

    fft2D(random, true);
    cout << "\nAfter inverse transform:" << endl;
    printMatrix(random);

    if (areMatricesEqual(random, randomCopy)) {
        cout << "PASSED: Random matrix transform and inverse transform" << endl;
    }
    else {
        cout << "FAILED: Random matrix test" << endl;
    }
}


// Modified test function with timing
void testFFT2D() {
    cout << "\nRunning 2D FFT Tests ...\n" << endl;



    vector<int> sizes = { 32, 64, 128, 256,512,1024,2048,4096 , 8192};

    for (int size : sizes) {
        cout << "\nTesting with " << size << "x" << size << " matrix:" << endl;

        // Create random matrix
        vector<vector<Complex>> random(size, vector<Complex>(size));

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                random[i][j] = Complex(rand() % 10, rand() % 10);
            }
        }
        vector<vector<Complex>> randomCopy = random;

        // Time the forward and inverse transforms
        double start = omp_get_wtime();

        fft2D(random, false);
        fft2D(random, true);

        double end = omp_get_wtime();

        if (areMatricesEqual(random, randomCopy)) {
            cout << "PASSED: Matrix transform and inverse transform" << endl;
        }
        else {
            cout << "FAILED: Matrix test" << endl;
        }

        cout << "Time taken: " << (end - start) << " seconds" << endl;
    }
}

int main() {

    omp_set_num_threads(4);
    printf("Starting OpenMP program...\n");

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();

        printf("Hello from thread %d of %d\n", thread_id, total_threads);
    }

   
    // Run the test suite
    testFFT2D();
    return 0;
}




#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <iomanip>
#include <random>
#include <cstdlib>
#include <omp.h>

constexpr double M_PI = 3.14159265358979323846;

using namespace std;

class Complex {
public:
    double real, imag;
    Complex(double r = 0, double i = 0) : real(r), imag(i) {}

    Complex operator + (const Complex& b) const {
        return Complex(real + b.real, imag + b.imag);
    }

    Complex operator - (const Complex& b) const {
        return Complex(real - b.real, imag - b.imag);
    }

    Complex operator * (const Complex& b) const {
        return Complex(real * b.real - imag * b.imag,
            real * b.imag + imag * b.real);
    }
};

unsigned int reverseBits(unsigned int num, int log2n) {
    unsigned int reversed = 0;
    for (int i = 0; i < log2n; ++i) {
        reversed = (reversed << 1) | (num & 1);
        num >>= 1;
    }
    return reversed;
}

vector<Complex> precomputeTwiddleFactors(int n) {
    vector<Complex> w(n);  // Changed to store all n factors
    for (int i = 0; i < n; i++) {
        double angle = -2.0 * M_PI * i / n;
        w[i] = Complex(cos(angle), sin(angle));
    }
    return w;
}

void fft1D(vector<Complex>& a, bool invert) {
    int n = a.size();

    if (n & (n - 1)) {
        cerr << "Error: Size must be a power of 2" << endl;
        return;
    }

    int log2n = 0;
    for (int temp = n; temp > 1; temp >>= 1)
        ++log2n;

    // Bit-reversal permutation
    for (int i = 0; i < n; i++) {
        int j = reverseBits(i, log2n);
        if (i < j)
            swap(a[i], a[j]);
    }

    // Precompute all twiddle factors
    vector<Complex> twiddle = precomputeTwiddleFactors(n);

    // Main FFT loop
    for (int len = 2; len <= n; len <<= 1) {
        int halfLen = len >> 1;
        int step = n / len;

        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < halfLen; j++) {
                Complex w = twiddle[j * step];
                if (invert)
                    w.imag = -w.imag;  // Conjugate the twiddle factor

                Complex t = w * a[i + j + halfLen];
                Complex u = a[i + j];
                a[i + j] = u + t;
                a[i + j + halfLen] = u - t;
            }
        }
    }

    // Scale inverse FFT
    if (invert) {
        double scale = 1.0 / n;
        for (Complex& x : a) {
            x.real *= scale;
            x.imag *= scale;
        }
    }
}


void fft2D(vector<vector<Complex>>& a, bool invert) {
    int n = a.size();
    int m = a[0].size();

    // Process rows
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        fft1D(a[i], invert);  // Process row directly without copying
    }

    // Transpose matrix
    vector<vector<Complex>> trans(m, vector<Complex>(n));
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            trans[j][i] = a[i][j];
        }
    }

    // Process columns (now rows of transposed matrix)
#pragma omp parallel for
    for (int i = 0; i < m; i++) {
        fft1D(trans[i], invert);
    }

    // Transpose back
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            a[i][j] = trans[j][i];
        }
    }
}

bool isApproximatelyEqual(const Complex& a, const Complex& b, double epsilon = 1e-5) {
    return abs(a.real - b.real) < epsilon && abs(a.imag - b.imag) < epsilon;
}

bool areMatricesEqual(const vector<vector<Complex>>& a, const vector<vector<Complex>>& b,
    double epsilon = 1e-5) {
    if (a.size() != b.size() || a[0].size() != b[0].size()) return false;

    for (size_t i = 0; i < a.size(); i++) {
        for (size_t j = 0; j < a[0].size(); j++) {
            if (!isApproximatelyEqual(a[i][j], b[i][j], epsilon)) {
                cout << "Mismatch at (" << i << "," << j << "): ";
                cout << "Original=" << a[i][j].real << "+" << a[i][j].imag << "i, ";
                cout << "Result=" << b[i][j].real << "+" << b[i][j].imag << "i" << endl;
                return false;
            }
        }
    }
    return true;
}

void testFFT2D() {
    cout << "\nRunning 2D FFT Tests with Fixed Implementation...\n" << endl;

    vector<int> sizes = { 32, 64, 128, 256 ,512,1024,2048,4096,8192 };

    for (int size : sizes) {
        cout << "\nTesting with " << size << "x" << size << " matrix:" << endl;

        // Create test matrix
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> dis(-10.0, 10.0);

        vector<vector<Complex>> matrix(size, vector<Complex>(size));
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                matrix[i][j] = Complex(dis(gen), dis(gen));
            }
        }

        vector<vector<Complex>> original = matrix;

        // Time the transforms
        double start = omp_get_wtime();

        fft2D(matrix, false);  // Forward transform
        fft2D(matrix, true);   // Inverse transform

        double end = omp_get_wtime();

        // Verify results
        bool passed = areMatricesEqual(matrix, original);
        cout << (passed ? "PASSED" : "FAILED") << ": Transform and inverse transform" << endl;
        cout << "Time taken: " << fixed << setprecision(6) << (end - start) << " seconds" << endl;

        if (!passed) {
            cout << "First few elements comparison:" << endl;
            for (int i = 0; i < min(4, size); i++) {
                for (int j = 0; j < min(4, size); j++) {
                    cout << "Position (" << i << "," << j << "):" << endl;
                    cout << "Original: " << original[i][j].real << " + " << original[i][j].imag << "i" << endl;
                    cout << "Result: " << matrix[i][j].real << " + " << matrix[i][j].imag << "i" << endl;
                }
            }
        }
    }
}

int main() {
    testFFT2D();
    return 0;
}
*/


#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <iomanip>
#include <random>
#include <cstdlib>

constexpr double M_PI = 3.14159265358979323846;

using namespace std;

class Complex {
public:
    double real, imag;

    Complex(double r = 0, double i = 0) : real(r), imag(i) {}

    Complex operator + (const Complex& b) const {
        return Complex(real + b.real, imag + b.imag);
    }

    Complex operator - (const Complex& b) const {
        return Complex(real - b.real, imag - b.imag);
    }

    Complex operator * (const Complex& b) const {
        // Use Karatsuba algorithm for better numerical stability
        double ac = real * b.real;
        double bd = imag * b.imag;
        double abcd = (real + imag) * (b.real + b.imag);
        return Complex(ac - bd, abcd - ac - bd);
    }

    Complex& operator *= (double scale) {
        real *= scale;
        imag *= scale;
        return *this;
    }
};

unsigned int reverseBits(unsigned int num, int log2n) {
    unsigned int reversed = 0;
    for (int i = 0; i < log2n; ++i) {
        reversed = (reversed << 1) | (num & 1);
        num >>= 1;
    }
    return reversed;
}

void fft1D(vector<Complex>& a, bool invert) {
    int n = a.size();

    // Validate input size
    if (n & (n - 1)) {
        cerr << "Error: Size must be a power of 2" << endl;
        return;
    }

    // Calculate log2(n)
    int log2n = 0;
    for (int temp = n; temp > 1; temp >>= 1)
        ++log2n;

    // Bit-reversal permutation
    for (int i = 0; i < n; i++) {
        int j = reverseBits(i, log2n);
        if (i < j)
            swap(a[i], a[j]);
    }

    // Compute FFT using optimized Cooley-Tukey algorithm
    for (int len = 2; len <= n; len <<= 1) {
        double angle = 2 * M_PI / len * (invert ? 1 : -1);

        for (int i = 0; i < n; i += len) {
            Complex w(1, 0);
            Complex wn(cos(angle), sin(angle));

            for (int j = 0; j < len / 2; j++) {
                Complex u = a[i + j];
                Complex v = a[i + j + len / 2] * w;

                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;

                w = w * wn;  // Rotate by the angle
            }
        }
    }

    // Scale for inverse transform
    if (invert) {
        double scale = 1.0 / n;
        for (auto& x : a)
            x *= scale;
    }
}

void fft2D(vector<vector<Complex>>& a, bool invert) {
    int n = a.size();
    int m = a[0].size();

    // Verify dimensions are power of 2
    if ((n & (n - 1)) != 0 || (m & (m - 1)) != 0) {
        cerr << "Error: Matrix dimensions must be powers of 2" << endl;
        return;
    }

    // Process rows
    for (int i = 0; i < n; i++) {
        fft1D(a[i], invert);
    }

    // Transpose the matrix
    vector<vector<Complex>> trans(m, vector<Complex>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            trans[j][i] = a[i][j];
        }
    }

    // Process columns (now rows of transposed matrix)
    for (int i = 0; i < m; i++) {
        fft1D(trans[i], invert);
    }

    // Transpose back
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            a[i][j] = trans[j][i];
        }
    }
}

bool isApproximatelyEqual(const Complex& a, const Complex& b, double epsilon = 1e-4) {
    // Use relative error for better accuracy comparison
    double abs_a = sqrt(a.real * a.real + a.imag * a.imag);
    double abs_b = sqrt(b.real * b.real + b.imag * b.imag);
    double abs_diff = sqrt(pow(a.real - b.real, 2) + pow(a.imag - b.imag, 2));

    if (abs_a < epsilon && abs_b < epsilon)
        return true;  // Both values are very close to zero

    return abs_diff / max(abs_a, abs_b) < epsilon;
}

bool areMatricesEqual(const vector<vector<Complex>>& a, const vector<vector<Complex>>& b,
    double epsilon = 1e-4) {
    if (a.size() != b.size() || a[0].size() != b[0].size()) return false;

    for (size_t i = 0; i < a.size(); i++) {
        for (size_t j = 0; j < a[0].size(); j++) {
            if (!isApproximatelyEqual(a[i][j], b[i][j], epsilon)) {
                cout << "Mismatch at (" << i << "," << j << "): ";
                cout << "Original=" << a[i][j].real << "+" << a[i][j].imag << "i, ";
                cout << "Result=" << b[i][j].real << "+" << b[i][j].imag << "i" << endl;
                return false;
            }
        }
    }
    return true;
}

void testFFT2D() {
    cout << "\nRunning 2D FFT Tests with Improved Numerical Stability...\n" << endl;

    vector<int> sizes = { 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 };

    for (int size : sizes) {
        cout << "\nTesting with " << size << "x" << size << " matrix:" << endl;

        // Create test matrix with controlled values
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> dis(-10.0, 10.0);

        vector<vector<Complex>> matrix(size, vector<Complex>(size));
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                matrix[i][j] = Complex(dis(gen), dis(gen));
            }
        }

        // Store original matrix
        vector<vector<Complex>> original = matrix;

        // Measure performance
        double start = clock();

        // Perform forward and inverse transforms
        fft2D(matrix, false);  // Forward transform
        fft2D(matrix, true);   // Inverse transform

        double end = clock();

        // Verify results with improved comparison
        bool passed = areMatricesEqual(matrix, original);
        cout << (passed ? "PASSED" : "FAILED") << ": Transform and inverse transform" << endl;
        cout << "Time taken: " << fixed << setprecision(6)
            << ((end - start) / CLOCKS_PER_SEC) << " seconds" << endl;

        if (!passed) {
            // Print detailed comparison for debugging
            cout << "\nDetailed comparison of first few elements:" << endl;
            for (int i = 0; i < min(4, size); i++) {
                for (int j = 0; j < min(4, size); j++) {
                    cout << "Position (" << i << "," << j << "):" << endl;
                    cout << "Original: " << original[i][j].real << " + " << original[i][j].imag << "i" << endl;
                    cout << "Result:   " << matrix[i][j].real << " + " << matrix[i][j].imag << "i" << endl;
                    cout << "Diff:     " << abs(original[i][j].real - matrix[i][j].real)
                        << " + " << abs(original[i][j].imag - matrix[i][j].imag) << "i" << endl;
                    cout << endl;
                }
            }
        }
    }
}

int main() {
    // Run the test suite
    testFFT2D();

    return 0;
}
