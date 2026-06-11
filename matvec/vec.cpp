#include "vec.h"
#include "mat.h"
#include <cmath>
#include <stdexcept>

using namespace std;

struct Mat;

Vec::Vec(int dim): dim(dim), comps(vector<double>(dim, 0)) {}

Vec::Vec(int dim, double val): dim(dim), comps(vector<double>(dim, val)) {}

Vec::Vec(vector<double> comps): comps(comps), dim(comps.size()) {}

Vec::Vec(initializer_list<double> init): comps(init), dim(comps.size()) {}

double& Vec::operator[](size_t index) {
    if (index >= dim) throw out_of_range("Index out of bounds");
    return comps[index];
}

const double& Vec::operator[](size_t index) const {
    if (index >= dim) throw out_of_range("Index out of bounds");
    return comps[index];
}

Vec Vec::operator+(const Vec& v) {
    Vec res(dim);
    if (v.dim != dim) throw invalid_argument("dimension mismatch");
    for (int i = 0; i < dim; i++) res[i] = v[i] + (*this)[i];
    return res;
}

Vec Vec::operator-(const Vec& v) {
    Vec res(dim);
    if (v.dim != dim) throw invalid_argument("dimension mismatch");
    for (int i = 0; i < dim; i++) res[i] = (*this)[i] - v[i];
    return res;
}

Vec Vec::operator*(double c) const {
    Vec res(dim);
    for (int i = 0; i < dim; i++) res[i] = c * (*this)[i];
    return res;
}

double Vec::dot(const Vec& v) const {
    if (dim != (int)v.comps.size()) throw invalid_argument("vector dimensions do not match");
    double tot = 0;
    for (int i = 0; i < dim; i++) tot += comps[i] * v.comps[i];
    return tot;
}

double Vec::operator*(const Vec& v) const {
    return this->dot(v);
}

Vec Vec::log_element_wise() const {
    Vec res(dim);
    for (int i = 0; i < dim; i++) res[i] = log((*this)[i]);
    return res;
}

Vec Vec::add_element_wise(double x) const {
    Vec res(dim);
    for (int i = 0; i < dim; i++) res[i] = x + (*this)[i];
    return res;
}

Vec Vec::mul_element_wise(const Vec& v) const {
    Vec res(dim);
    if (v.dim != dim) throw invalid_argument("dimension mismatch");
    for (int i = 0; i < dim; i++) res[i] = (*this)[i] * v[i];
    return res;
}

Vec Vec::sigmoid_element_wise() const {
    Vec res(dim);
    for (int i = 0; i < dim; i++) res[i] = 1.0 / (1.0 + exp(-(*this)[i]));
    return res;
}

Mat Vec::transpose() const {
    Mat res(1, dim);
    for (int j = 0; j < dim; j++) res.entries[0][j] = (*this)[j];
    return res;
}

Mat Vec::to_matrix() const {
    Mat m(dim, 1);
    for (int i = 0; i < dim; i++) m.entries[i][0] = comps[i];
    return m;
}


