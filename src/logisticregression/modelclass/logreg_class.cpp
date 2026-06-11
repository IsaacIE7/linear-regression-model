#include <iostream>
#include <cmath>
#include <utility>
#include <vector>

#include "../../matvec/vec.h"
#include "../../matvec/mat.h"

using namespace std;

struct Model {
    Vec weights;
    double bias;

    Model(int numFeatures):
    weights(Vec(numFeatures, 0)),
    bias(0)
    {}

    // void add_data(Mat X, Vec val) {
    //     for (int i = 0; i < X.rows; i++) {
    //         data.entries.push_back(X.col_vec(i).comps);
    //     }
    //     dataVals.comps.insert(dataVals.comps.end(), 
    //     make_move_iterator(val.comps.begin()), 
    //     make_move_iterator(val.comps.end()));
    // }

    Vec compute_z(const Mat& X) {
        Vec v = X * weights;
        Vec b(v.dim, bias);
        return v + b;
    }

    double compute_z(const Vec& x) {
        return (weights.transpose() * x).comps[0] + bias;
    }

    static Vec sigmoid(Vec z) {
        Vec res(z.dim, 0);
        int i = 0;
        for (double a: z.comps) {
            res[i] = 1 / (1 + exp(a * -1));
            i++;
        }
        return res;
    }

    static double sigmoid(double z) {
        return 1 / (1 + exp(-z));
    }

    static double sigmoid_binary(double z) {
        return sigmoid(z) >= 0.5 ? 1.0 : 0.0;
    }

    static Vec sigmoid_binary(const Vec& sig) {
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

    static Vec gradient(Vec& p, const Vec& y, const Mat& X) {
    Mat Xt = X.transpose();
    
    Vec grad = Xt * (p - y);
    return grad * (1.0/p.dim); 
    } 

    static double sum_partials(const Vec& gradient) {
        double tot = 0;
        for (double p: gradient.comps) {
            tot += abs(p);
        }
        return tot;
        }

    static double partialB(Vec& p, const Vec& y) {
        return (Vec(p.dim, 1).transpose() * (p - y)).comps[0] * 1.0/p.dim;
    }

    pair<Vec, double> train(const Vec& y, const Mat& X){
        Vec weightsP = weights;
        Vec z = compute_z(X);
        Vec p = sigmoid(z);
        Vec g(weights.dim);
        double parB = partialB(p, y);
        double bp = bias;

        int i = 0;
        // while (error(p, y) > 0.1) {
        while (i < 1e8) {
            z = compute_z(X);
            p = sigmoid(z);
            g = gradient(p, y, X);
            parB = partialB(p, y);

            if (i % 30000 == 0) cout << "Loss: " << error(p, y) << "\n";
            
            for (int j = 0; j < weightsP.dim; j++) {
                weightsP[j] -= 0.01 * g[j];
            }
            if (abs(parB) + sum_partials(g) < 0.001) {
                cout << "partials small at iteration  " << i << "\n";
                break;
            }
            bp -= (0.01  * parB);

            i++;

            weights = weightsP;
            bias = bp;
        }

        

        return {weightsP, bp};
    }

    int predict(const Vec& features) {
        int res = sigmoid_binary(compute_z(features));
        return res;
    }    
};


int main() {
    cout << "starting" << endl;
    //Test data with 2 features per sample
    vector<vector<double>> rawX = {
        {1.0, 2.0},
        {2.0, 3.0},
        {1.5, 2.5},
        {3.0, 4.0},
        {0.5, 1.5}
    };

    vector<vector<double>> perfectX = {
        {1.0, 2.0},
        {2.0, 3.0},
        {3.0, 4.0},
        {4.0, 5.0},
        {5.0, 6.0}
    };

    vector<double> rawY = {0.0, 1.0, 0.0, 1.0, 0.0};

    Mat X(rawX);
    Vec y(rawY);

    Model m(2);

    double bias = 0;

    cout << "before training: " << endl;

    m.train(y, X);

    cout << "finished training" << endl;

    Vec z = m.compute_z(X);
    Vec p = m.sigmoid(z);

    cout << "loss: " << m.error(p, y) << endl;
    cout << "prediction: " << m.predict(Vec({4.0, 5.0})) << endl;

    return 0;
}
