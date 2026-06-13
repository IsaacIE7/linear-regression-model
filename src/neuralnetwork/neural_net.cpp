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

    

    double randomWeight() {
        return -1.0 + ((double)rand() / RAND_MAX) * 2.0;
    }
};


struct NeuralNet {
    vector<int> layout;
    vector<Layer> layers;

    vector<Mat> activations;
    vector<Mat> zValues;

    NeuralNet(vector<int> layout):
    layout(layout), 
    activations({}), 
    zValues({})
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

    Mat backprop() {
        
    }
};

int main() {
    NeuralNet N({2, 3, 1});
    Mat p = N.forward(Mat({{1.0, 1.0}, {2.0, 1.0}, {3.0, 1.0}}));
    Mat y({{0.0}, {0.0}, {1.0}});
    double L = N.loss(y);
    for (auto a: p.entries) {
        for (auto b: a) {
            cout << b << " ";
        }
    }
    cout << L << endl;

}
