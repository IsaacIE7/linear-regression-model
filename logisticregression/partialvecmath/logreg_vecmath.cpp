#include <iostream>
#include <cmath>
#include <utility>
#include <vector>

using namespace std;

double compute_z(vector<double> weights, double b, vector<double> features) {
    if (weights.size() != features.size()) {
    throw invalid_argument("weights and features must match in size");
    }
    double total = 0;
    int i = 0;
    for (double w: weights) {
        total += w * (features[i]);
        i++;
    }
    return total + b;
}

double sigmoid(vector<double> weights, double b, vector<double> features) {
    return 1 / (1 + exp(-1 * compute_z(weights, b, features)));
}
double predict(vector<double> weights, double b, vector<double> features) {
    return sigmoid(weights, b, features) >= 0.5 ? 1 : 0;
}


double error(vector<double> weights, double b,  vector<pair<vector<double>, double>> &data) {
    double totalError = 0;
    for (auto& point : data) {
        vector<double> features = point.first;
        double y = point.second;
        totalError += -1 * (y * log(sigmoid(weights, b, features)) + (1-y) * log(1 - sigmoid(weights,b,features)));
    }
    return totalError / data.size();
}

// double error_point(vector<double> weights, double b, double x, double y) {
//     return -1 * (y * log(sigmoid(weights, b, x)) + (1-y) * log(1 - sigmoid(weights ,b,x)));
// }

vector<double> partial_numerical(vector<double> weights, double b, vector<pair<vector<double>, double>> &data) {
    double dm = 0.00001;
    vector<double> partials;
    for (double m: weights) {
        partials.push_back((error(weights, b, data) - error(weights, b, data)) / dm);
    }
    return partials; 
} 

vector<double> partial(vector<double> weights, double b,  vector<pair<vector<double>, double>> &data) {
    int n = weights.size();
    vector<double> gradient(weights.size());
    double total;
    total = 0;
        

    for (auto point: data) {
        auto features = point.first;
        double y = point.second;
        double p = sigmoid(weights, b, features);
             
        for (int i = 0; i < weights.size(); i++) { 
            double xi = features[i];
            gradient[i] += (p - y) * xi;
        }
    }

    for (int i = 0; i < weights.size(); i++) {
        gradient[i] /= data.size();
    }

    return gradient; 
}

double sum_partials(vector<double> weights, double b,  vector<pair<vector<double>, double>> &data) {
    double tot = 0;
    vector<double> partials = partial(weights, b, data);
    for (double p: partials) {
        tot += abs(p);
    }
    return tot;
}

double partialB(vector<double> weights, double b,  vector<pair<vector<double>, double>> &data) {
    double total = 0;
    for (auto point: data) {
        auto features = point.first;
        double y = point.second;
        total += (sigmoid(weights, b, features) - y);
    }
    return total/data.size(); 
}


pair<vector<double>, double> train(vector<double> weights, double b, vector<pair<vector<double>, double>> data){
    vector<double> weightsP = weights;
    vector<double> partialsM = partial(weights, b, data);
    double parB = partialB(weights, b, data);
    double bp = b;
   

    int i = 0;
    while (error(weightsP, bp, data) > 0.1) {
        partialsM = partial(weightsP, bp, data);
        parB = partialB(weightsP, bp, data);

        if (abs(parB) + sum_partials(weightsP, bp, data) < 0.01) break;


        int j = 0;
        for (double& mp: weightsP) {
            mp -= (0.001  * partialsM[j]);
            j++;
        }

        bp -= (0.001  * parB);
        i++;
    }

    return {weightsP, bp};
}



pair<vector<double>, double> train_display(vector<double> weights, double b, vector<pair<vector<double>, double>> data){
    vector<double> weightsP = weights;
    vector<double> partialsM = partial(weights, b, data);
    double parB = partialB(weights, b, data);
    double bp = b;
   

    int i = 0;
    while (error(weightsP, bp, data) > 0.1) {
        partialsM = partial(weightsP, bp, data);
        parB = partialB(weightsP, bp, data);

        if (i % 250000 == 0) cout << "Loss: " << error(weightsP, bp, data) << "\n";

        if (abs(parB) + sum_partials(weightsP, bp, data) < 0.01) break;


        int j = 0;
        for (double& mp: weightsP) {
            mp -= (0.001  * partialsM[j]);
            j++;
        }

        bp -= (0.001  * parB);
        i++;
    }

    return {weightsP, bp};
}

void printPredic(vector<double> weights, double b, vector<pair<vector<double>, double>> data, vector<double> features) {
    // Create a temporary data point with the given features
    vector<pair<vector<double>, double>> tempData = {{features, 0}};
    cout << "Features: ";
    for (double f : features) cout << f << " ";
    cout << endl;
}




// Old main method:
// int main() {
//
//     // vector<pair<double,double>> data = {{1, 3}, {2, 6}, {3, 9}, {4, 12}}; 
//     vector<pair<double,double>> data = {
//     {3, 4},
//     {6, 9},
//     {2, 3},
//     {4, 7},
//     {5, 6}
// };
//
//     double m = 0;
//     double b = 0;
//
//     pair<double, double> res = minimize_err_display(m, b, data);
//     cout << "m = " << res.first << ", b = " << res.second << endl;
//     cout << "prediction at x = 3 is " << predict(res.first, res.second, 3) 
//     << " actual value is " << "unavailable" <<  endl;
// }

int main() {

    // Test data with 2 features per sample
    vector<pair<vector<double>, double>> data = {
        {{1.0, 2.0}, 0},
        {{2.0, 3.0}, 1},
        {{1.5, 2.5}, 0},
        {{3.0, 4.0}, 1},
        {{0.5, 1.5}, 0}
    };

    // Initialize weights and bias
    vector<double> weights = {0.1, 0.1};
    double b = 0.0;

    cout << "Initial error: " << error(weights, b, data) << endl;
    cout << "Initial sigmoid: " << sigmoid(weights, b, {1, 2}) << endl;

    // Train the model
    pair<vector<double>, double> res = train_display(weights, b, data);
    
    cout << "Final bias: " << res.second << endl;
    cout << "Final error: " << error(res.first, res.second, data) << endl;
    cout << "Final sigmoid: " << predict(res.first, res.second, {1, 2}) << endl;
}