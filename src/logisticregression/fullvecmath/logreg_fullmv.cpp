#include <iostream>
#include <cmath>
#include <utility>
#include <vector>

#include "../../matvec/vec.h"
#include "../../matvec/mat.h"

using namespace std;

Vec compute_z(const Vec& weights, double b, const Mat& X) {
    Vec v = X * weights;
    Vec bias(v.dim, b);
    if (v.dim != bias.dim) throw invalid_argument("weights vector and bias vector mismatch");
    return v + bias;
}

double compute_z(const Vec& weights, const double& b, const Vec& x) {
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


int predict(const Vec& weights, double b, const Vec& features) {
    int res = sigmoid_binary(compute_z(weights, b, features));
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
        {0.5, 1.5},
        {}
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

    for (double w: finalWeights.comps) {
        cout << "w: " << w << endl;
    }
    cout << "bias: " << finalB << endl;

    cout << "prediction: " << predict(finalWeights, finalB, Vec({1.0, 2.0})) << endl 
    << predict(finalWeights, finalB, Vec({3.5, 2.5})) << endl;
    


    return 0;
}
