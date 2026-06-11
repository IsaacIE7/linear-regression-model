#include <iostream>
#include <cmath>
#include <utility>
#include <vector>

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
        bias(neurons, 0) {}

    Layer(Mat weights, Vec bias): 
        weights(weights),
        bias(bias) {}

    pair<Vec, Vec> forward(const Vec& inputs) {
        Vec z = (weights * inputs) + bias;
        Vec p = z.sigmoid_element_wise();
        return {z, p};
    }
};


struct NeuralNet {
    vector<int> layout;
    vector<Layer> layers;

    vector<Vec> activations;
    vector<Vec> zValues;
    vector<Vec> inputs;

    NeuralNet(vector<int> layout):
    layout(layout)
    {
        for (int i = 1; i < layout.size(); i++) {
            layers.push_back(Layer(layout[i], layout[i - 1]));
        }
    }

    Vec forward(const Vec& input) {
        inputs.push_back(input);

        for (int i = 1; i < layers.size(); i++) {
            auto pair = layers[i].forward(activations[i]);
            activations.push_back(pair.second);
            zValues.push_back(pair.second);   
            inputs.push_back(pair.second);    
        }
        return activations.back();
    }

    Vec loss_layer(int layerNum) {
        Vec p = activations[layerNum - 1];
        Vec y = inputs[layerNum - 1];

        Vec v1 = y.transpose() * p.log_element_wise();
        Vec v2 = ((y.add_element_wise(-1)) * (-1.0)).transpose() * ((p.add_element_wise(-1) * (-1.0)).log_element_wise());
        
        Vec res = v1 + v2;

        return res * (1.0 / y.dim); 
    }


};

int main(){
    
}
