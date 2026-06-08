#include <iostream>
#include <cmath>
#include <utility>
#include <vector>

using namespace std;

// double m = 2;
// double b = 2;

struct Mat; // forward-declare Mat so Vec can reference it in signatures before Mat is defined

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
    // allow Vec sample = {1.0, 2.0}; without ambiguity between int and vector constructors
    Vec(initializer_list<double> init): comps(init), dim(comps.size()) {}

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

Vec compute_z(const Vec& weights, double b, const Mat& X) {
    cout << "in compute_z_mat" << endl;
    cout << "X rows = " << X.rows << endl;
    cout << "X cols = " << X.cols << endl;
    cout << "weights dim = " << weights.dim << endl;
    Vec v = X * weights;

        cout << "v dim = " << v.dim << endl;
    Vec bias(v.dim, b);
    if (v.dim != bias.dim) throw invalid_argument("weights vector and bias vector mismatch");
    return v + bias;
}

double compute_z(const Vec& weights, const double& b, const Vec& x) {
    cout << "in compute_z_vector" << endl;
    return (weights.transpose() * x).comps[0] + b;
}

Vec sigmoid(Vec z) {
    Vec res(z.dim, 0);
    int i = 0;
    for (double a: z.comps) {
        res[i] = 1 / (1 + exp(a * -1));
        i++;
    }
    return res;
}

double sigmoid(double z) {
    return 1 / (1 + exp(-z));
}

double sigmoid_binary(double z) {
    return sigmoid(z) >= 0.5 ? 1.0 : 0.0;
}

Vec sigmoid_binary(const Vec& sig) {
    Vec res(sig.dim, 0);
    int i = 0;
    for (double a: sig.comps) {
        (a >= 0.5) ? res[i] = 1 : res[i] =  0;
    }
    return res;
}

double error(const Vec& p, const Vec& y) {
    cout << "in error" << endl;
    if (p.dim != y.dim) throw invalid_argument("vector dim mismatch");
    Vec v1 = y.transpose() * p.log_element_wise();
    Vec v2 = ((y.add_element_wise(-1)) * (-1.0)).transpose() * ((p.add_element_wise(-1) * (-1.0)).log_element_wise());
    Vec res = v1 + v2;

    double err = 0;
    for (double a: res.comps) {
        err += a;
    }
    return -err / res.dim;
}

Vec gradient(Vec& p, const Vec& y, const Mat& X) {
    cout << "in gradient" << endl;
    Mat Xt = X.transpose();
    
    Vec grad = Xt * (p - y);
    return grad * (1.0/p.dim); 
} 

double sum_partials(const Vec& gradient) {
    double tot = 0;
    for (double p: gradient.comps) {
        tot += abs(p);
    }
    return tot;
}

double partialB(Vec& p, const Vec& y) {
    return (Vec(p.dim, 1).transpose() * (p - y)).comps[0] * 1.0/p.dim;
}


pair<Vec, double> train(const Vec weights, double b, const Vec& y, const Mat& X){
    Vec weightsP = weights;
    Vec z = compute_z(weightsP, b, X);
    Vec p = sigmoid(z);
    Vec g(weights.dim);
    double parB = partialB(p, y);
    double bp = b;

    int i = 0;
    // while (error(p, y) > 0.1) {
    while (i < 1e7) {
        z = compute_z(weightsP, bp, X);
        p = sigmoid(z);
        g = gradient(p, y, X);
        parB = partialB(p, y);
        
        for (int j = 0; j < weightsP.dim; j++) {
            weightsP[j] -= 0.01 * g[j];
        }
        if (abs(parB) + sum_partials(g) < 0.01) break;

        bp -= (0.01  * parB);

        i++;
    }

    return {weightsP, bp};
}

pair<Vec, double> train_display(const Vec weights, double b, const Vec& y, const Mat& X){
    Vec weightsP = weights;
    Vec z = compute_z(weightsP, b, X);
    Vec p = sigmoid(z);
    Vec g(weights.dim);
    double parB = partialB(p, y);
    double bp = b;

    int i = 0;
    // while (error(p, y) > 0.1) {
    while (i < 1e7) {
        z = compute_z(weightsP, bp, X);
        p = sigmoid(z);
        g = gradient(p, y, X);
        parB = partialB(p, y);
        
        if (i % 250000 == 0) cout << "Loss: " << error(p, y) << "\n";


        for (int j = 0; j < weightsP.dim; j++) {
            weightsP[j] -= 0.01 * g[j];
        }
        if (abs(parB) + sum_partials(g) < 0.01) break;

        bp -= (0.01  * parB);

        i++;
    }

    return {weightsP, bp};
}


int predict_print(const Vec& weights, double b, const Vec& features) {
    int res = sigmoid_binary(compute_z(weights, b, features));
    cout << res << endl;
    return res;
}

int main() {
    cout << "starting" << endl;
    // Test data with 2 features per sample
    vector<vector<double>> rawX = {
        {1.0, 2.0},
        {2.0, 3.0},
        {1.5, 2.5},
        {3.0, 4.0},
        {0.5, 1.5}
    };
    vector<double> rawY = {0.0, 1.0, 0.0, 1.0, 0.0};

    Mat X(rawX);
    Vec y(rawY);
    Vec weights({0.1, 0.1});
    double bias = 0;

    cout << "before training: " << endl;

    pair<Vec, double> res = train_display(weights, bias, y, X);
    Vec finalWeights = res.first;
    double finalB = res.second;

    predict_print(finalWeights, finalB, Vec({1.0, 2.0})); // Should be close to 0
    


    return 0;
}
