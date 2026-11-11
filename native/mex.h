// Private compatibility surface required by the unchanged upstream kernels.
// This is not a MATLAB runtime or a general MEX object implementation.
#pragma once
#include <vector>
#include <string>
#include <stdexcept>
#include <cstddef>
struct mxArray {
    std::size_t rows = 0, cols = 0;
    std::vector<double> values;
    std::string text;
    mxArray(std::size_t m, std::size_t n, const double* data = nullptr)
        : rows(m), cols(n), values(m*n, 0) {
        if (data) values.assign(data, data+m*n);
    }
};
constexpr int mxREAL = 0;
inline std::size_t mxGetM(const mxArray* a) { return a->rows; }
inline std::size_t mxGetN(const mxArray* a) { return a->cols; }
inline double* mxGetPr(const mxArray* a) { return const_cast<double*>(a->values.data()); }
inline double* mxGetPi(const mxArray*) { return nullptr; }
inline mxArray* mxCreateDoubleMatrix(std::size_t m, std::size_t n, int) { return new mxArray(m,n); }
inline char* mxArrayToString(const mxArray* a) { return const_cast<char*>(a->text.c_str()); }
inline void mexErrMsgTxt(const char* message) { throw std::runtime_error(message); }
