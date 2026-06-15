#include <iostream>
#include <cmath>
#include <utility>
#include <vector>
#include <cstdlib>
#include "../matvec/vec.h"
#include "../matvec/mat.h"

using namespace std;


struct Layer {
    Mat weights;
    Vec bias;

    int neurons;
    int weightAmnt;

    Layer(int neurons, int weightAmnt): // layer with n neurons and k weights, weights and biases init to 0
        weights(Mat(neurons, weightAmnt)), 
        bias(neurons, 0),
        neurons(neurons),
        weightAmnt(weightAmnt)
        {
            for (int j = 0; j < weightAmnt; j++) {
                for (int i = 0; i < neurons; i++) {
                    weights.entries[i][j] = randomWeight();
                }
            }

            for (double& b: bias.comps) {
                b = randomWeight();
            }
        }

    Layer(Mat weights, Vec bias): 
        weights(weights),
        bias(bias), 
        neurons(weights.rows),
        weightAmnt(weights.cols) {}

    pair<Vec, Vec> forward_lyr_single(const Vec& inputs) {
        Vec z = (weights * inputs) + bias;
        Vec p = z.sigmoid_element_wise();
        return {z, p};
    }

    pair<Mat, Mat> forward_lyr(const Mat& inputs) {
        Mat z = (inputs * weights.transpose()).add_vec_to_row(bias);
        Mat p = z.sigmoid_element_wise();

        return {z, p};
    }

    Mat gradient_w_lyr(const Mat& prev_activation, const Mat& D) { // pass in p - y if output layer
        return D.transpose() * prev_activation * (1.0 / prev_activation.rows);
    }

    Mat compute_D(const Mat& weights_next,  const Mat& D_next, const Mat& crnt_activation) {
        return  (D_next * weights_next).multiply_element_wise(crnt_activation.multiply_element_wise((crnt_activation - 1) * -1));
    }

    Mat compute_D_output(const Mat& p, const Mat& y) {
        return p - y;
    }

    Vec gradient_b_lyr(const Mat& D) {
        return (D.transpose() * Vec(D.rows, 1)) * (1.0 / D.rows);
    }

    double randomWeight() {
        return -1.0 + ((double)rand() / RAND_MAX) * 2.0;
    }
};


struct NeuralNet {
    vector<int> layout;
    vector<Layer> layers; // doesnt include input layer, just hidden and output layers

    vector<Mat> activations; // includes input layer, hidden layers, and output layer
    vector<Mat> zValues;

    vector<Mat> gradWeights;
    vector<Vec> gradBiases;

    NeuralNet(vector<int> layout):
    layout(layout), 
    activations({}), 
    zValues({}),
    gradWeights({}),
    gradBiases({})
    {
        for (int i = 1; i < layout.size(); i++) {
            layers.push_back(Layer(layout[i], layout[i - 1]));
        }
    }


   Mat forward(const Mat& data) {
        activations.clear();
        zValues.clear();

        activations.push_back(data);
        
        auto current = layers[0].forward_lyr(data); //get first hidden layer activation and z values
        zValues.push_back(current.first); // add  first hidden layer z predictions to zvals list
        activations.push_back(current.second); // add first hidden layer activations ot activation list

        for (int i = 1; i < layers.size(); i++) {
            current = layers[i].forward_lyr(activations[i]); //layers doesnt include input layer, activations does
            zValues.push_back(current.first);
            activations.push_back(current.second); 
        }
        return activations.back();
    }

    double loss(Mat y) {
        Mat p = activations.back();

        Mat m1 = y.multiply_element_wise(p.log_element_wise());
        Mat m2 = ((y - 1) * (-1.0)).multiply_element_wise((((p - 1) * (-1.0)).log_element_wise()));
        
        Mat res = (m1 + m2) * -1;

        return (res * (1.0 / (y.rows * y.cols))).sum_entries(); 
    }

    // void backprop(Mat y) {
    //     gradWeights.clear();
    //     gradBiases.clear();

    //     Mat D = layers.back().compute_D_output(activations.back(), y);

    //     gradWeights.push_back(layers.back().gradient_w_lyr(activations[layers.size() - 1], D));
    //     gradBiases.push_back(layers.back().gradient_b_lyr(D));

    //     Mat Dnext = D;

    //     for (int i = layers.size() - 2; i >= 0; i--) {
    //         D = layers[i].compute_D(layers[i + 1].weights, Dnext, activations[i + 1]);
    //         Mat gradcurrent = layers[i].gradient_w_lyr(activations[i], D);
    //         Vec gradbias = layers[i].gradient_b_lyr(D);
    //         Dnext = D;

    //         gradWeights.push_back(gradcurrent);
    //         gradBiases.push_back(gradbias);
    //     }
    // }

    void backprop(const Mat& y) {
        gradWeights.clear();
        gradBiases.clear();

        gradWeights.resize(layers.size(), Mat(0, 0));
        gradBiases.resize(layers.size(), Vec(0));

        Mat D = layers.back().compute_D_output(activations.back(), y);

        gradWeights[layers.size() - 1] = (layers.back().gradient_w_lyr(activations[layers.size() - 1], D));
        gradBiases[layers.size() - 1] = (layers.back().gradient_b_lyr(D));

        Mat Dnext = D;

        for (int i = layers.size() - 2; i >= 0; i--) {
            D = layers[i].compute_D(layers[i + 1].weights, Dnext, activations[i + 1]);
            Mat gradcurrent = layers[i].gradient_w_lyr(activations[i], D);
            Vec gradbias = layers[i].gradient_b_lyr(D);
            Dnext = D;

            gradWeights[i] = gradcurrent;
            gradBiases[i] = gradbias;
        }
    }

    // void update(double learning_rate) {
    //     for (int i = 0; i < gradWeights.size(); i++) { // gradients start with output layer and moves backwards
    //         int in = gradWeights.size() - 1 - i;       //layers start with first hidden
    //         layers[i].weights = layers[i].weights - (gradWeights[in] * learning_rate);
    //         layers[i].bias = layers[i].bias - (gradBiases[in] * learning_rate);
    //     }
    // }

    void update(double learning_rate) {
        for (int i = 0; i < gradWeights.size(); i++) {
            layers[i].weights = layers[i].weights - (gradWeights[i] * learning_rate);
            layers[i].bias = layers[i].bias - (gradBiases[i] * learning_rate);
        }
    }

    void train(const Mat& data, const Mat& y, int epochs, double learning_rate) {
        forward(data);
        double current_loss = loss(y);
        int i = 0;

        while (i < epochs && current_loss > 1e-3) {
            forward(data);
            current_loss = loss(y);
            backprop(y);
            update(learning_rate);
            i++;
        }
    }

    Vec predict(const Mat& input) { // should be m x 1 only one sample
        return forward(input).to_vector(); // if input is m x 1 output mat should be n x 1
    }

    Mat predict_binary(const Mat& data) {
        Mat p = forward(data);
        Mat res(p.rows, p.cols);

        for (int i = 0; i < p.rows; i++) {
            for (int j = 0; j < p.cols; j++) {
                res.entries[i][j] = p.entries[i][j] >= 0.5 ? 1.0 : 0.0;
            }
        }

        return res;
    }

    double accuracy_binary(const Mat& data, const Mat& y) {
        Mat pred = predict_binary(data);

        if (pred.rows != y.rows || pred.cols != y.cols) {
            throw invalid_argument("accuracy dimension mismatch");
        }

        int correct = 0;
        int total = pred.rows * pred.cols;

        for (int i = 0; i < pred.rows; i++) {
            for (int j = 0; j < pred.cols; j++) {
                if (pred.entries[i][j] == y.entries[i][j]) {
                    correct++;
                }
            }
        }

        return (double)correct / total;
    }


};

// int main() {
//     NeuralNet N({2, 3, 1});
//     Mat p = N.forward(Mat({{1.0, 1.0}, {2.0, 1.0}, {3.0, 1.0}}));
//     Mat y({{0.0}, {0.0}, {1.0}});
//     double L = N.loss(y);
//     for (auto a: p.entries) {
//         for (auto b: a) {
//             cout << b << " ";
//         }
//     }
//     cout << L << endl;
// }

int main() {
    try {
        srand(0);

        auto print_results = [](NeuralNet& N, const Mat& X, const Mat& y) {
            Mat p = N.forward(X);
            cout << "Loss: " << N.loss(y) << endl;

            for (int i = 0; i < p.rows; i++) {
                double prob = p.entries[i][0];
                int pred = prob >= 0.5 ? 1 : 0;

                cout << X.entries[i][0] << ", " << X.entries[i][1]
                     << " -> prob: " << prob
                     << " pred: " << pred
                     << " actual: " << y.entries[i][0]
                     << endl;
            }
        };

        Mat X({
            {0.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
            {1.0, 1.0}
        });

        Mat y_and({
            {0.0},
            {0.0},
            {0.0},
            {1.0}
        });

        Mat y_xor({
            {0.0},
            {1.0},
            {1.0},
            {0.0}
        });

        cout << "================ AND TEST ================" << endl;

        NeuralNet and_net({2, 3, 1});

        cout << "\nBefore training:" << endl;
        print_results(and_net, X, y_and);

        cout << "\nTraining AND..." << endl;
        and_net.train(X, y_and, 50000, 0.1);

        cout << "\nAfter training:" << endl;
        print_results(and_net, X, y_and);


        cout << "\n\n================ XOR TEST ================" << endl;

        NeuralNet xor_net({2, 4, 1});

        cout << "\nBefore training:" << endl;
        print_results(xor_net, X, y_xor);

        cout << "\nTraining XOR..." << endl;
        xor_net.train(X, y_xor, 100000, 0.5);

        cout << "\nAfter training:" << endl;
        print_results(xor_net, X, y_xor);

        cout << "XOR accuracy: " << xor_net.accuracy_binary(X, y_xor) << endl;

    }
    catch (const std::exception& e) {
        cout << "Runtime error: " << e.what() << endl;
    }

    return 0;
}
