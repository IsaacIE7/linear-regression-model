#include <iostream>
#include <cmath>
#include <utility>
#include <vector>

using namespace std;

// double m = 2;
// double b = 2;

struct Mat; // forward declaration Mat so Vec can reference it before Mat is defined

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
};

struct Mat {
    vector<vector<double>> entries;
    int rows;
    int cols;
    pair<int, int> dim;

    Mat(int rows, int cols): //all entries 0 m x n matrix
    entries(rows, vector<double> (cols ,0)), // to initialize vector without naming, vector<type>(size, value)
    dim({rows, cols}),
    rows(rows), cols(cols)
    {}

    Mat(vector<vector<double>> entries): // create matrix with initialized values
    entries(entries), 
    dim({entries.size(), entries[0].size()}),
    rows(entries.size()),
    cols(entries[0].size())
    {}


    Vec mat_mul(const Vec& v) const {
        vector<double> vals(rows, 0);
        Vec res(vals); 
        if (dim.second != v.dim) throw invalid_argument("matrix vector dimension mismatch");
        for (int i = 0; i < entries.size(); i++) {
            double rowSum = 0;
            for (int j = 0; j < entries[0].size(); j++) {
                rowSum += (v[j] * entries[i][j]);
            }
            res[i] = rowSum;
        }
        return res;
    }

    Vec operator*(const Vec& v) const {
        return mat_mul(v);
    }

    Mat operator+(Mat m) const {
        if (dim != m.dim) throw invalid_argument("matrix dimension mismatch");
        vector<vector<double>> vals(rows, vector<double>(cols, 0));
        Mat res(vals);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res.entries[i][j] = entries[i][j] + m.entries[i][j];
            }
        }
        return res;
    }

    Mat operator-(Mat m) const {
        if (dim != m.dim) throw invalid_argument("matrix dimension mismatch");
        vector<vector<double>> vals(rows, vector<double>(cols, 0));
        Mat res(vals);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res.entries[i][j] = entries[i][j] - m.entries[i][j];
            }
        }
        return res;
    }

    Mat operator*(double c) const {
        vector<vector<double>> vals(rows, vector<double>(cols, 0));
        Mat res(vals);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res.entries[i][j] = c * entries[i][j];
            }
        }
        return res;
    }

    Vec row_vec(int row) const {
        if (row < 0 || row >= rows) throw out_of_range("row index out of bounds");
        vector<double> vals(cols);
        for (int j = 0; j < cols; j++) {
            vals[j] = entries[row][j];
        }
        return Vec(vals);
    }

    Vec col_vec(int col) const {
        if (col < 0 || col >= cols) throw out_of_range("column index out of bounds");
        vector<double> vals(rows);
        for (int i = 0; i < rows; i++) {
            vals[i] = entries[i][col];
        }
        return Vec(vals);
    }

    Mat operator*(const Mat& m) {
        if (cols != m.rows) throw invalid_argument("matrix dimension incompatibility");
        Mat res(m.rows, m.cols);
        for (int j = 0; j < m.cols; j++) {
            for (int i = 0; i < rows; i++) {
                res.entries[i][j] = (*this).row_vec(i) * m.col_vec(j);
            }
        }
        return res;
    }


    Mat transpose () const { //lets compiler know that original Mat isnt edited
        Mat res(cols, rows);

        for (int j = 0; j < cols; j++ ) { //swapped loop order , more cache efficient for large datasets
            for (int i = 0; i < rows; i++) {
                res.entries[j][i] = entries[i][j];
            }
        }
        return res;
    }

    Vec to_vector() const {
        if (cols != 1) throw invalid_argument("cannot convert");
        vector<double> vals(rows);
        for (int i = 0; i < rows; i++) {
            vals[i] = entries[i][0];
        }
        return Vec(vals);
    }

    
};

// Define Mat-returning Vec methods only after Mat is fully declared.
Mat Vec::transpose() const {
    Mat res(1, dim);
    for (int j = 0; j < dim; j++) {
        res.entries[0][j] = (*this)[j];
    }
    return res;
}

Mat Vec::to_matrix() const {
    Mat m(dim, 1);
    for (int i = 0; i < dim; i++) {
        m.entries[i][0] = comps[i];
    }
    return m;
}

struct Layer {
    Mat weights;
    Vec bias;

    Layer(int neurons, int weightAmnt): // layer with n neurons and k weights, weights and biases init to 0
        weights(Mat(neurons, weightAmnt)), 
        bias(neurons, 0) {}

    Layer(Mat weights, Vec bias): 
        weights(weights),
        bias(bias) {}

    Vec forward(const Vec& inputs) {
        Vec z = (weights * inputs) + bias;
        Vec p = z.sigmoid_element_wise();
        zValues.push_back(z);
        activations.push_back(p);
        return p;
    }
};


struct NeuralNet {
    vector<int> layout;
    vector<Layer> layers;

    vector<Vec> activations;
    vector<Vec> zValues;

    NeuralNet(vector<int> layout):
    layout(layout)
    {
        for (int i = 1; i < layout.size(); i++) {
            layers.push_back(Layer(layout[i], layout[i - 1]));
        }
    }
};

int main(){
    
}
