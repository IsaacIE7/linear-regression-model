#include <utility>
#include <vector>
#include <cmath>
#include <stdexcept>
#include "../matvec/mat.h"
using namespace std;

struct Vec {
    vector<double> comps;
    int dim = 0;

    Vec(int dim): //create n dimensional vector with components initialized to 0
    dim(dim), 
    comps(vector<double> (dim, 0)) {}

    Vec(int dim, double val): //create n dimensional vector with components initialized to desired val
    dim(dim), 
    comps(vector<double> (dim, val)) {}

    Vec(vector<double> comps): comps(comps), dim(comps.size()) {} //initialize with components
    
    //initializer list is fixed size, read only, and used mainly for constuctors and function arguments
    Vec(initializer_list<double> init): comps(init), dim(comps.size()) {} // allows vector declaration, Vec v = {1.0, 2.0, 3.0};

    //read and modify
    double& operator[](size_t index) { //size_t is an unsigned 64 bit integer long long
        if (index >= dim) throw out_of_range("Index out of bounds");
        return comps[index]; // Returns a reference
    }
    
    //read only
    // const at end means this function will not change the vector
    // const at start means you can see the number but cant edit it
    const double& operator[](size_t index) const { 
        if (index >= dim) throw out_of_range("Index out of bounds");
        return comps[index]; // 
    }

    Vec operator+(const Vec& v) {
        Vec res(dim);
        if (v.dim != dim) throw invalid_argument("dimension mismatch");
        for (int i = 0; i < dim; i++) {
            res[i] = v[i] + (*this)[i]; // deference this ptr
        }
        return res;
    }

    Vec operator-(const Vec& v) {
        Vec res(dim);
        if (v.dim != dim) throw invalid_argument("dimension mismatch");
        for (int i = 0; i < dim; i++) {
            res[i] = (*this)[i] - v[i]; // deference this ptr
        }
        return res;
    }

    Vec operator*(double c) const {
        Vec res(dim);
        for (int i = 0; i < dim; i++) {
            res[i] = c * (*this)[i]; // deference this ptr
        }
        return res;
    }

    double dot(const Vec& v) const {
        if (dim != v.comps.size()) throw invalid_argument("vector dimensions do not match") ;
        double tot = 0;
        for (int i = 0; i < dim; i++) {
            tot += comps[i] * v.comps[i];
        }
        return tot;
    }

    double operator*(const Vec& v) const {
        return (*this).dot(v);
    }

    // Declare Mat-returning Vec methods here. Their definitions come after Mat is fully defined.
    Mat transpose () const;
    Mat to_matrix() const;

    Vec log_element_wise() const {
        Vec res(dim);
        for (int i = 0; i < dim; i++) {
            res[i] = log((*this)[i]);
        }
        return res;
    }

    Vec add_element_wise(double x) const {
        Vec res(dim);
        for (int i = 0; i < dim; i++) {
            res[i] = x + (*this)[i];
        }
        return res;
    }

    Vec mul_element_wise(const Vec& v) const {
        Vec res(dim);
        if (v.dim != dim) throw invalid_argument("dimension mismatch");
        for (int i = 0; i < dim; i++) {
            res[i] = (*this)[i] * v[i];
        }
        return res;
    }

    Vec sigmoid_element_wise() const {
        Vec res(dim);
        for (int i = 0; i < dim; i++) {
            res[i] = 1 / (1 + exp(-(*this)[i]));
        }
        return res;
    }

    // Define Mat-returning Vec methods only after Mat is fully declared.
Mat transpose() const {
    Mat res(1, dim);
    for (int j = 0; j < dim; j++) {
        res.entries[0][j] = (*this)[j];
    }
    return res;
}

    Mat to_matrix() const {
        Mat m(dim, 1);
        for (int i = 0; i < dim; i++) {
            m.entries[i][0] = comps[i];
        }
        return m;
    }
};


