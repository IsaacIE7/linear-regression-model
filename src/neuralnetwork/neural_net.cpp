#include <iostream>
#include <cmath>
#include <utility>
#include <vector>
#include <cstdlib>
#include <algorithm>

#include "raylib.h"

#include "mnist/mnist_reader_less.hpp"

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

    double loss(const Mat& y) {
        Mat p = activations.back();

        Mat m1 = y.multiply_element_wise(p.log_element_wise());
        Mat m2 = ((y - 1) * (-1.0)).multiply_element_wise((((p - 1) * (-1.0)).log_element_wise()));
        
        Mat res = (m1 + m2) * -1;

        return (res * (1.0 / (y.rows * y.cols))).sum_entries(); 
    }

    void backprop(const Mat& y) {
        gradWeights.clear();
        gradBiases.clear();

        Mat D = layers.back().compute_D_output(activations.back(), y);

        gradWeights.push_back(layers.back().gradient_w_lyr(activations[layers.size() - 1], D));
        gradBiases.push_back(layers.back().gradient_b_lyr(D));

        Mat Dnext = D;

        for (int i = layers.size() - 2; i >= 0; i--) {
            D = layers[i].compute_D(layers[i + 1].weights, Dnext, activations[i + 1]);
            Mat gradcurrent = layers[i].gradient_w_lyr(activations[i], D);
            Vec gradbias = layers[i].gradient_b_lyr(D);
            Dnext = D;

            gradWeights.push_back(gradcurrent);
            gradBiases.push_back(gradbias);
        }
        reverse(gradWeights.begin(), gradWeights.end());
        reverse(gradBiases.begin(), gradBiases.end());
    }

    // void backprop(const Mat& y) {
    //     gradWeights.clear();
    //     gradBiases.clear();

    //     gradWeights.resize(layers.size(), Mat(0, 0));
    //     gradBiases.resize(layers.size(), Vec(0));

    //     Mat D = layers.back().compute_D_output(activations.back(), y);

    //     gradWeights[layers.size() - 1] = (layers.back().gradient_w_lyr(activations[layers.size() - 1], D));
    //     gradBiases[layers.size() - 1] = (layers.back().gradient_b_lyr(D));

    //     Mat Dnext = D;

    //     for (int i = layers.size() - 2; i >= 0; i--) {
    //         D = layers[i].compute_D(layers[i + 1].weights, Dnext, activations[i + 1]);
    //         Mat gradcurrent = layers[i].gradient_w_lyr(activations[i], D);
    //         Vec gradbias = layers[i].gradient_b_lyr(D);
    //         Dnext = D;

    //         gradWeights[i] = gradcurrent;
    //         gradBiases[i] = gradbias;
    //     }
    // }

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

            if (i % 10000 == 0) {
            cout << "iteration " << i
                 << " Loss: " << current_loss
                 << " Accuracy: " << accuracy_binary(data, y)
                 << endl;
            }

            backprop(y);
            update(learning_rate);
            i++;
        }
    }

    Vec predict_one_ex(const Mat& input) { // should be 1 x m only one sample
        return forward(input).to_vector(); // if input is 1 x m output mat should be m x 1
    }

    Vec predict_one_ex_bin(const Mat& input) {
        Vec v = predict_one_ex(input); 
        Vec res(v.dim);
        
        for (int i = 0; i < input.rows; i++) {
            res.comps[i] = v.comps[i] >= 0.5 ? 1 : 0;  
        }
        return res; 
    }

    Vec predict(const Vec& input) { 
        return forward(input.to_matrix()).to_vector(); 
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

//uint8_t is unsigned 8 bit integer 0-255
pair<Mat, Mat> convert_data(const mnist::MNIST_dataset<uint8_t, uint8_t>& dataset, int num_samples) {
    if (dataset.training_labels.empty())  throw invalid_argument("MNIST failed to load. Check file paths and filenames.");

    Mat X(num_samples, 784);
    Mat Y(num_samples, 1);

    const auto& training_images = dataset.training_images;
    const auto& training_labels = dataset.training_labels;
    // auto test_images = dataset.test_images;
    // auto test_labels = dataset.test_labels;
    
    int count = 0;

    for (int i = 0; i < training_labels.size() && count < num_samples; i++) {
        if (training_labels[i] == 1 || training_labels[i] == 0) {
            for (int j = 0; j < training_images[0].size(); j++) {
                 X.entries[count][j] = training_images[i][j] / 255.0; // use x[count] because i gets skipped sometimes
            }

            if (training_labels[i] == 1) {
                Y.entries[count][0] = 1;
            } else  {
                Y.entries[count][0] = 0;
            }   
            
            count++;
        }     
    }

    return {X, Y};
}


void display(vector<int> layout) { 
    const int screenWidth = 1200; 
    const int screenHeight = 1000; 

    const int neuronsize = max(2, 20 - layout[0] / 4);

    const int clean_vertical = screenHeight - 200;
    const int clean_horizontal = screenWidth - 400;

    const int input_amnt = layout[0]; 

    
    
    InitWindow(screenWidth, screenHeight, "Neural Network"); 
    SetTargetFPS(60); 

    while (!WindowShouldClose()) { 
        vector<pair<double, double>> circle_coords;
        BeginDrawing(); 
        ClearBackground(GRAY); 
        
        DrawText("input neurons", 250, 100, 25, GREEN);

        for (int i = 0; i < layout.size(); i++) {
            int y_spacing = (int)(1.0 * clean_vertical / (layout[i] + 1));         
            int x_spacing = (1.0 * clean_horizontal / (layout.size()));         
            for (int j = 1; j <= layout[i]; j++) {   
                int xPos = 50 + x_spacing * (i + 1);
                int yPos = 150 + y_spacing * j;

                circle_coords.push_back({xPos, yPos});

                DrawCircle(xPos, yPos, neuronsize, BLACK); 
            } 
        } 

        for (int i = 0; i < layout[0]; i++) {
            for (int j = 0; j < layout[1]; j++) {
                auto circle1 = circle_coords[i];
                auto circle2 = circle_coords[j + layout[0]];

                double startx = circle1.first;
                double starty = circle1.second;
                double endx = circle2.first;
                double endy = circle2.second;
                DrawLine(startx, starty, endx, endy, BLACK);
            }
        }

        for (int i = 0; i < layout[1]; i++) {
            for (int j = 0; j < layout[2]; j++) {
                auto circle1 = circle_coords[i + layout[0]];
                auto circle2 = circle_coords[j + layout[0] + layout[1]];

                double startx = circle1.first;
                double starty = circle1.second;
                double endx = circle2.first;
                double endy = circle2.second;
                DrawLine(startx, starty, endx, endy, BLACK);
            }
        }


        DrawText("FPS: 60", 20, 15, 20, GREEN); 
        EndDrawing(); 
    } 

    CloseWindow(); 
}



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


// ======= MAIN 2 =======


// int main() {
//     try {
//         srand(0);

//         // auto print_results = [](NeuralNet& N, const Mat& X, const Mat& y) {
//         //     Mat p = N.forward(X);
//         //     cout << "Loss: " << N.loss(y) << endl;

//         //     for (int i = 0; i < p.rows; i++) {
//         //         double prob = p.entries[i][0];
//         //         int pred = prob >= 0.5 ? 1 : 0;

//         //         cout << X.entries[i][0] << ", " << X.entries[i][1]
//         //              << " -> prob: " << prob
//         //              << " pred: " << pred
//         //              << " actual: " << y.entries[i][0]
//         //              << endl;
//         //     }
//         // };

//         // Mat X({
//         //     {0.0, 0.0},
//         //     {0.0, 1.0},
//         //     {1.0, 0.0},
//         //     {1.0, 1.0}
//         // });

//         // Mat y_and({
//         //     {0.0},
//         //     {0.0},
//         //     {0.0},
//         //     {1.0}
//         // });

//         // Mat y_xor({
//         //     {0.0},
//         //     {1.0},
//         //     {1.0},
//         //     {0.0}
//         // });

//         // cout << "================ AND TEST ================" << endl;

//         // NeuralNet and_net({2, 3, 1});

//         // cout << "\nBefore training:" << endl;
//         // print_results(and_net, X, y_and);

//         // cout << "\nTraining AND..." << endl;
//         // and_net.train(X, y_and, 50000, 0.1);

//         // cout << "\nAfter training:" << endl;
//         // print_results(and_net, X, y_and);


//         // cout << "\n\n================ XOR TEST ================" << endl;

//         // NeuralNet xor_net({2, 4, 1});

//         // cout << "\nBefore training:" << endl;
//         // print_results(xor_net, X, y_xor);

//         // cout << "\nTraining XOR..." << endl;
//         // xor_net.train(X, y_xor, 100000, 0.5);

//         // cout << "\nAfter training:" << endl;
//         // print_results(xor_net, X, y_xor);

//         // cout << "XOR accuracy: " << xor_net.accuracy_binary(X, y_xor) << endl;

//         auto dataset = mnist::read_dataset<uint8_t, uint8_t>();

// cout << "Training images: " << dataset.training_images.size() << endl;
// cout << "Training labels: " << dataset.training_labels.size() << endl;
// cout << "Test images: " << dataset.test_images.size() << endl;
// cout << "Test labels: " << dataset.test_labels.size() << endl;

// if (!dataset.training_labels.empty()) {
//     cout << "First label: " << (int)dataset.training_labels[0] << endl;
//     cout << "First image pixel count: " << dataset.training_images[0].size() << endl;
// } else {
//     cout << "MNIST failed to load. Check file paths and filenames." << endl;
// }
//     }
//     catch (const std::exception& e) {
//         cout << "Runtime error: " << e.what() << endl;
//     }

//     return 0;
// }

// int main() {
//     try {
//         auto dataset = mnist::read_dataset<uint8_t, uint8_t>();

//         cout << "Raw MNIST loaded:" << endl;
//         cout << "Training images: " << dataset.training_images.size() << endl;
//         cout << "Training labels: " << dataset.training_labels.size() << endl;
//         cout << "Test images: " << dataset.test_images.size() << endl;
//         cout << "Test labels: " << dataset.test_labels.size() << endl;

//         auto converted = convert_data(dataset, 1000);

//         Mat X = converted.first;
//         Mat Y = converted.second;

//         cout << "\nConverted binary dataset:" << endl;
//         cout << "X: " << X.rows << " x " << X.cols << endl;
//         cout << "Y: " << Y.rows << " x " << Y.cols << endl;

//         cout << "\nFirst 10 labels after conversion:" << endl;
//         for (int i = 0; i < 10; i++) {
//             cout << Y.entries[i][0] << " ";
//         }
//         cout << endl;

//         cout << "\nFirst image first 20 pixels:" << endl;
//         for (int j = 0; j < 20; j++) {
//             cout << X.entries[0][j] << " ";
//         }
//         cout << endl;

//         double minPixel = X.entries[0][0];
//         double maxPixel = X.entries[0][0];

//         for (int i = 0; i < X.rows; i++) {
//             for (int j = 0; j < X.cols; j++) {
//                 if (X.entries[i][j] < minPixel) minPixel = X.entries[i][j];
//                 if (X.entries[i][j] > maxPixel) maxPixel = X.entries[i][j];
//             }
//         }

//         cout << "\nPixel range:" << endl;
//         cout << "min: " << minPixel << endl;
//         cout << "max: " << maxPixel << endl;

//         int zeros = 0;
//         int ones = 0;

//         for (int i = 0; i < Y.rows; i++) {
//             if (Y.entries[i][0] == 0.0) zeros++;
//             else if (Y.entries[i][0] == 1.0) ones++;
//             else cout << "Bad label at row " << i << ": " << Y.entries[i][0] << endl;
//         }

//         cout << "\nLabel counts:" << endl;
//         cout << "zeros: " << zeros << endl;
//         cout << "ones: " << ones << endl;
//     }
//     catch (const exception& e) {
//         cout << "Runtime error: " << e.what() << endl;
//     }

//     return 0;
// }

// int main() {
//     try {
//         auto dataset = mnist::read_dataset<uint8_t, uint8_t>();

//         auto train = convert_data(dataset, 50);
//         Mat X_train = train.first;
//         Mat y_train = train.second;

//         NeuralNet N({784, 32, 1});

//         cout << "Before training:" << endl;
//         N.forward(X_train);
//         cout << "Loss: " << N.loss(y_train) << endl;
//         cout << "Accuracy: " << N.accuracy_binary(X_train, y_train) << endl;

//         cout << "\nTraining..." << endl;
//         N.train(X_train, y_train, 1000, 0.1);

//         cout << "\nAfter training:" << endl;
//         N.forward(X_train);
//         cout << "Loss: " << N.loss(y_train) << endl;
//         cout << "Accuracy: " << N.accuracy_binary(X_train, y_train) << endl;
//     }
//     catch (const exception& e) {
//         cout << "Runtime error: " << e.what() << endl;
//     }

//     return 0;
// }

int main() {
    display({20, 10, 5});
    return 0;
}



