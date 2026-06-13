#include "mat.h"
#include "vec.h"
#include <stdexcept>
#include <vector>
#include <iostream>

using namespace std;

Mat::Mat(int rows, int cols): entries(rows, vector<double>(cols, 0)), dim({rows, cols}), rows(rows), cols(cols) {}

Mat::Mat(vector<vector<double>> entries): entries(entries), dim({(int)entries.size(), (int)entries[0].size()}), rows((int)entries.size()), cols((int)entries[0].size()) {}

Vec Mat::mat_mul(const Vec& v) const {
    vector<double> vals(rows, 0);
    Vec res(vals);
    
    if (dim.second != v.dim) throw invalid_argument("matrix vector dimension mismatch");
    
    for (int i = 0; i < entries.size(); i++) {
        double rowSum = 0;
        for (int j = 0; j < entries[0].size(); j++) rowSum += (v[j] * entries[i][j]);
        res[i] = rowSum;
    }
    return res;
}

Vec Mat::operator*(const Vec& v) const { return mat_mul(v); }

Mat Mat::operator+(Mat m) const {
    if (dim != m.dim) throw invalid_argument("matrix dimension mismatch");
    Mat res(rows, cols);
    for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) res.entries[i][j] = entries[i][j] + m.entries[i][j];
    return res;
}

Mat Mat::add_vec_to_row(const Vec& v) {
    if (cols != v.dim) throw invalid_argument("columns and vec length mismatch");
    Mat res(rows, cols);
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            res.entries[i][j] = entries[i][j] + v[j];
        }
    }
    return res;
}

Mat Mat::operator+(double c) const {
    Mat res(rows, cols);
    for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) res.entries[i][j] = entries[i][j] + c;
    return res;
}
Mat Mat::operator-(Mat m) const {
    if (dim != m.dim) throw invalid_argument("matrix dimension mismatch");
    Mat res(rows, cols);
    for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) res.entries[i][j] = entries[i][j] - m.entries[i][j];
    return res;
}

Mat Mat::operator-(double c) const {
    Mat res(rows, cols);
    for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) res.entries[i][j] = entries[i][j] - c;
    return res;
}

Mat Mat::operator*(double c) const {
    Mat res(rows, cols);
    for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) res.entries[i][j] = c * entries[i][j];
    return res;
}

Vec Mat::row_vec(int row) const {
    if (row < 0 || row >= rows) throw out_of_range("row index out of bounds");
    vector<double> vals(cols);
    for (int j = 0; j < cols; j++) vals[j] = entries[row][j];
    return Vec(vals);
}

Vec Mat::col_vec(int col) const {
    if (col < 0 || col >= cols) throw out_of_range("column index out of bounds");
    vector<double> vals(rows);
    for (int i = 0; i < rows; i++) vals[i] = entries[i][col];
    return Vec(vals);
}

Mat Mat::operator*(const Mat& m) const {
    if (cols != m.rows) throw invalid_argument("mat * mat, matrix dimension incompatibility");
    Mat res(rows, m.cols);
    for (int j = 0; j < m.cols; j++) for (int i = 0; i < rows; i++) res.entries[i][j] = this->row_vec(i) * m.col_vec(j);
    return res;
}

Mat Mat::multiply_element_wise(const Mat& m) const{
    if (rows != m.rows != cols != m.cols) throw invalid_argument("mat * mat, matrix dimension incompatibility");
    Mat res(rows, cols);
    for (int i = 0; i < rows; i++)  {
        for (int j = 0; j < cols; j++) {
            res.entries[i][j] = entries[i][j] * m.entries[i][j];
        }
    }
    return res;
}

Mat Mat::sigmoid_element_wise() const {
    Mat res(rows, cols);
    for (int i = 0; i < rows; i++)  {
        for (int j = 0; j < cols; j++) {
            res.entries[i][j] = 1.0 / (1.0 + exp(-(*this).entries[i][j]));
        }
    }
    return res;
}

// dont know if epsilon clipping should be done here or in loss func
Mat Mat::log_element_wise() const {
    Mat res(rows, cols);
    const double epsilon = 1e-7;

    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            if (entries[i][j] < epsilon) res.entries[i][j] = log(epsilon);
            else if (entries[i][j] > 1 - epsilon) res.entries[i][j] = log(1 - epsilon);
            else res.entries[i][j] = log((*this).entries[i][j]);
        }
    }
    return res;
}

double Mat::sum_entries() const {
    double sum = 0;
    for (int j = 0; j < cols; j ++) {
        for (int i = 0; i < rows; i++) {
            sum += entries[i][j];
        }
    }
    return sum;
}

Mat Mat::transpose() const {
    Mat res(cols, rows);
    for (int j = 0; j < cols; j++) for (int i = 0; i < rows; i++) res.entries[j][i] = entries[i][j];
    return res;
}

Vec Mat::to_vector() const {
    if (cols != 1) throw invalid_argument("cannot convert");
    vector<double> vals(rows);
    for (int i = 0; i < rows; i++) vals[i] = entries[i][0];
    return Vec(vals);
}