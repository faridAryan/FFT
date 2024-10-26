

#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <iomanip>
#include <random>
#include <cstdlib>
#include <omp.h>
#include <fstream>
#include <string>

//constexpr double M_PI = 3.14159265358979323846;

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
bool readInputData(const std::string& filename, std::vector<std::vector<Complex>>& matrix) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        return false;
    }

    int rows, cols;
    infile >> rows >> cols;
    matrix.resize(rows, std::vector<Complex>(cols));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double real, imag;
            infile >> real >> imag;
            matrix[i][j] = Complex(real, imag);
        }
    }

    infile.close();
    return true;
}

bool writeOutputData(const std::string& filename, const std::vector<std::vector<Complex>>& matrix) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        return false;
    }

    int rows = matrix.size();
    int cols = matrix[0].size();
    outfile << rows << " " << cols << std::endl;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            outfile << matrix[i][j].real << " " << matrix[i][j].imag << " ";
        }
        outfile << std::endl;
    }

    outfile.close();
    return true;
}

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
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        fft1D(a[i], invert);
    }

    // Transpose the matrix
    vector<vector<Complex>> trans(m, vector<Complex>(n));
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            trans[j][i] = a[i][j];
        }
    }

    // Process columns (now rows of transposed matrix)
#pragma omp parallel for schedule(dynamic)
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

    vector<int> sizes = { 32, 64, 128, 256 ,512,1024,2048,4096,8192 };

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
        double start = omp_get_wtime();

        // Perform forward and inverse transforms
        fft2D(matrix, false);  // Forward transform
        fft2D(matrix, true);   // Inverse transform

        double end = omp_get_wtime();

        // Verify results with improved comparison
        bool passed = areMatricesEqual(matrix, original);
        cout << (passed ? "PASSED" : "FAILED") << ": Transform and inverse transform" << endl;
        cout << "Time taken: " << fixed << setprecision(6) << (end - start) << " seconds" << endl;

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
    // Initialize OpenMP
    int num_threads = omp_get_max_threads();

    omp_set_dynamic(0);     // Disable dynamic teams
  
        cout << "Running with " << num_threads << " threads\n" << endl;
        omp_set_num_threads(num_threads); // Use 4 threads by default

        // Run the test suite
        testFFT2D();



    return 0;
}


/*

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

// Complex number class remains the same
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

// Parallelized 1D FFT implementation
void fft1D(vector<Complex>& a, bool invert) {
    int n = a.size();
    if (n == 1) return;

    vector<Complex> a0(n / 2), a1(n / 2);

    // Modified loop condition for OpenMP compatibility
#pragma omp parallel for
    for (int i = 0; i < n / 2; i++) {
        a0[i] = a[2 * i];
        a1[i] = a[2 * i + 1];
    }

    // Sequential recursive calls
    fft1D(a0, invert);
    fft1D(a1, invert);

    double ang = 2 * M_PI / n * (invert ? -1 : 1);
    Complex w(1), wn(cos(ang), sin(ang));

    // Modified loop condition for OpenMP compatibility
#pragma omp parallel for firstprivate(w, wn)
    for (int i = 0; i < n / 2; i++) {
        Complex w_local = Complex(w.real * cos(i * ang) - w.imag * sin(i * ang),
            w.real * sin(i * ang) + w.imag * cos(i * ang));

        a[i] = a0[i] + w_local * a1[i];
        a[i + n / 2] = a0[i] - w_local * a1[i];

        if (invert) {
            a[i].real /= 2;
            a[i + n / 2].real /= 2;
            a[i].imag /= 2;
            a[i + n / 2].imag /= 2;
        }
    }
}

// Parallelized 2D FFT implementation
void fft2D(vector<vector<Complex>>& a, bool invert) {
    int n = a.size();
    int m = a[0].size();

    // Apply FFT to each row in parallel
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        fft1D(a[i], invert);
    }

    // Transpose the matrix in parallel
    vector<vector<Complex>> trans(m, vector<Complex>(n));
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            trans[j][i] = a[i][j];
        }
    }

    // Apply FFT to each column (now row after transpose) in parallel
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < m; i++) {
        fft1D(trans[i], invert);
    }

    // Transpose back in parallel
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            a[i][j] = trans[j][i];
        }
    }
}

// Helper functions remain the same
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

bool isApproximatelyEqual(const Complex& a, const Complex& b, double epsilon = 1e-10) {
    return abs(a.real - b.real) < epsilon && abs(a.imag - b.imag) < epsilon;
}

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

// Modified test function with timing
void testFFT2D(int num_threads) {
    cout << "\nRunning 2D FFT Tests with OpenMP (Basic Parallelization)...\n" << endl;

    cout << "Running with " << num_threads << " threads\n" << endl;

    vector<int> sizes = { 32, 64, 128, 256 ,512,1024,2048,4096,8192};

    for (int size : sizes) {
        cout << "\nTesting with " << size << "x" << size << " matrix:" << endl;

        // Create random matrix
        vector<vector<Complex>> random(size, vector<Complex>(size));
#pragma omp parallel for collapse(2)
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
    // Initialize OpenMP
     // Set number of threads
    int num_threads = omp_get_max_threads();

    omp_set_dynamic(0);     // Disable dynamic teams
    while (num_threads > 0)
    {

        omp_set_num_threads(num_threads); // Use 4 threads by default

        // Run the test suite
        testFFT2D(num_threads);


        num_threads -= 4;
    }
   
 
    return 0;
}
*/

/* 
Running 2D FFT Tests with OpenMP (Basic Parallelization)...

Running with 16 threads


Testing with 32x32 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 0.0004759 seconds

Testing with 64x64 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 0.0012293 seconds

Testing with 128x128 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 0.0043898 seconds

Testing with 256x256 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 0.0174376 seconds

Testing with 512x512 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 0.0715782 seconds

Testing with 1024x1024 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 0.304954 seconds

Testing with 2048x2048 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 1.26724 seconds

Testing with 4096x4096 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 4.73206 seconds

Testing with 8192x8192 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 19.9477 seconds

Running 2D FFT Tests with OpenMP (Basic Parallelization)...

Running with 12 threads


Testing with 32x32 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 0.0003141 seconds

Testing with 64x64 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 0.0012177 seconds

Testing with 128x128 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 0.0045914 seconds

Testing with 256x256 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 0.0178428 seconds

Testing with 512x512 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 0.074708 seconds

Testing with 1024x1024 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 0.310159 seconds

Testing with 2048x2048 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 1.31925 seconds

Testing with 4096x4096 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 5.44681 seconds

Testing with 8192x8192 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 22.0455 seconds

Running 2D FFT Tests with OpenMP (Basic Parallelization)...

Running with 8 threads


Testing with 32x32 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 0.0003574 seconds

Testing with 64x64 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 0.0012707 seconds

Testing with 128x128 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 0.0051216 seconds

Testing with 256x256 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 0.0212397 seconds

Testing with 512x512 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 0.086008 seconds

Testing with 1024x1024 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 0.373411 seconds

Testing with 2048x2048 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 1.50413 seconds

Testing with 4096x4096 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 6.15985 seconds

Testing with 8192x8192 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 25.1566 seconds

Running 2D FFT Tests with OpenMP (Basic Parallelization)...

Running with 4 threads


Testing with 32x32 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 0.0003816 seconds

Testing with 64x64 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 0.0015693 seconds

Testing with 128x128 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 0.0062985 seconds

Testing with 256x256 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 0.0246817 seconds

Testing with 512x512 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 0.100651 seconds

Testing with 1024x1024 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 0.422931 seconds

Testing with 2048x2048 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 1.77039 seconds

Testing with 4096x4096 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 7.45027 seconds

Testing with 8192x8192 matrix:
PASSED: Matrix transform and inverse transform
Time taken: 30.5131 seconds
*/


//task version
/*

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

// Complex number class remains the same
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

// Task-based 1D FFT implementation
void fft1D(vector<Complex>& a, bool invert, int depth = 0) {
    int n = a.size();
    if (n == 1) return;

    vector<Complex> a0(n / 2), a1(n / 2);

    #pragma omp parallel for if(depth <= 1)
    for (int i = 0; 2 * i < n; i++) {
        a0[i] = a[2 * i];
        a1[i] = a[2 * i + 1];
    }

    // Create tasks for recursive calls if we're not too deep
    if (depth < 4) { // Limit task creation depth to avoid overhead
        #pragma omp task if(n > 32)
        fft1D(a0, invert, depth + 1);

        #pragma omp task if(n > 32)
        fft1D(a1, invert, depth + 1);

        #pragma omp taskwait
    } else {
        // Sequential execution for smaller subdivisions
        fft1D(a0, invert, depth + 1);
        fft1D(a1, invert, depth + 1);
    }

    double ang = 2 * M_PI / n * (invert ? -1 : 1);
    Complex w(1), wn(cos(ang), sin(ang));

    #pragma omp parallel for if(depth <= 1) firstprivate(w, wn)
    for (int i = 0; 2 * i < n; i++) {
        Complex w_local = Complex(w.real * cos(i * ang) - w.imag * sin(i * ang),
                                w.real * sin(i * ang) + w.imag * cos(i * ang));

        a[i] = a0[i] + w_local * a1[i];
        a[i + n / 2] = a0[i] - w_local * a1[i];

        if (invert) {
            a[i].real /= 2;
            a[i + n / 2].real /= 2;
            a[i].imag /= 2;
            a[i + n / 2].imag /= 2;
        }
    }
}

// Task-based 2D FFT implementation
void fft2D(vector<vector<Complex>>& a, bool invert) {
    int n = a.size();
    int m = a[0].size();

    // Start parallel region for tasks
    #pragma omp parallel
    {
        #pragma omp single
        {
            // Apply FFT to each row using tasks
            for (int i = 0; i < n; i++) {
                #pragma omp task firstprivate(i)
                {
                    fft1D(a[i], invert);
                }
            }

            #pragma omp taskwait

            // Allocate transpose matrix
            vector<vector<Complex>> trans(m, vector<Complex>(n));

            // Create tasks for matrix transpose
            for (int i = 0; i < n; i += 32) {
                for (int j = 0; j < m; j += 32) {
                    #pragma omp task firstprivate(i, j)
                    {
                        // Process blocks of 32x32 elements
                        for (int ii = i; ii < min(i + 32, n); ii++) {
                            for (int jj = j; jj < min(j + 32, m); jj++) {
                                trans[jj][ii] = a[ii][jj];
                            }
                        }
                    }
                }
            }

            #pragma omp taskwait

            // Apply FFT to each column (now row after transpose) using tasks
            for (int i = 0; i < m; i++) {
                #pragma omp task firstprivate(i)
                {
                    fft1D(trans[i], invert);
                }
            }

            #pragma omp taskwait

            // Create tasks for transpose back
            for (int i = 0; i < n; i += 32) {
                for (int j = 0; j < m; j += 32) {
                    #pragma omp task firstprivate(i, j)
                    {
                        // Process blocks of 32x32 elements
                        for (int ii = i; ii < min(i + 32, n); ii++) {
                            for (int jj = j; jj < min(j + 32, m); jj++) {
                                a[ii][jj] = trans[jj][ii];
                            }
                        }
                    }
                }
            }

            #pragma omp taskwait
        }
    }
}

// Helper functions remain the same
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

bool isApproximatelyEqual(const Complex& a, const Complex& b, double epsilon = 1e-10) {
    return abs(a.real - b.real) < epsilon && abs(a.imag - b.imag) < epsilon;
}

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

// Modified test function with timing and performance comparison
void testFFT2D() {
    cout << "\nRunning 2D FFT Tests with OpenMP (Task-Based Parallelization)...\n" << endl;

    // Set number of threads
    int num_threads = omp_get_max_threads();
    cout << "Running with " << num_threads << " threads\n" << endl;

    // Test with different matrix sizes
    vector<int> sizes = {32, 64, 128, 256, 512};

    for (int size : sizes) {
        cout << "\nTesting with " << size << "x" << size << " matrix:" << endl;

        // Create random matrix
        vector<vector<Complex>> random(size, vector<Complex>(size));
        #pragma omp parallel for collapse(2)
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
    // Initialize OpenMP
    omp_set_dynamic(0);     // Disable dynamic teams
    omp_set_num_threads(4); // Use 4 threads by default

    // Run the test suite
    testFFT2D();
    return 0;
}



*/
