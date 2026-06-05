#include <iostream>
#include <cmath>
#include <utility>
#include <vector>

using namespace std;

// double m = 2;
// double b = 2;


double predict_point(vector<double> weights, double b, vector<pair<vector<double>, double>> data, int index) {
    double total = 0;
    int i = 0;
    for (double m: weights) {
        total += m * (data[i].first[i]);
        i++;
    }
    return total + b;
}

double predict(vector<double> weights, double b, vector<pair<vector<double>, double>> data) {
    double total = 0;
    int i = 0;
    for (double m: weights) {
        total += m * (data[i].first[i]);
        i++;
    }
    return total + b;
}

double sigmoid(vector<double> weights, double b, vector<pair<vector<double>, double>> data) {
    return 1 / (1 + exp(-1 * predict(weights, b, data)));
}

double error(vector<double> weights, double b,  vector<pair<vector<double>, double>> data) {
    double totalError = 0;
    int i = 0;
    for (auto& point : data) {
        double x = point.first[i];
        double y = point.second;
        totalError += -1 * (y * log(sigmoid(weights, b, data)) + (1-y) * log(1 - sigmoid(weights,b,data)));
        i++;
    }
    return totalError / data.size();
}

// double error_point(vector<double> weights, double b, double x, double y) {
//     return -1 * (y * log(sigmoid(weights, b, x)) + (1-y) * log(1 - sigmoid(weights ,b,x)));
// }

vector<double> partial_numerical(vector<double> weights, double b, vector<pair<vector<double>, double>> data) {
    double dm = 0.00001;
    vector<double> partials;
    for (double m: weights) {
        partials.push_back((error(weights, b, data) - error(weights, b, data)) / dm);
    }
    return partials; 
} 

// double partialM_numerical(double m, double b,  vector<pair<double,double>> data) {
//     double dm = 0.00001;
//     return (error(m + dm, b, data) - error(m, b, data)) / dm; 
// } 

// double partialM(double m, double b,  vector<pair<double,double>> data) {
//     double totalError = 0;
//     for (auto& point : data) {
//         double x = point.first;
//         double y = point.second;
//         totalError += 2 * (y - (m * x + b)) * (-x);
//     }
//     return totalError / data.size(); 
// } 

vector<double> partial(vector<double> weights, double b,  vector<pair<vector<double>, double>> data) {
    double totalError = 0;
    vector<double> partials;
    for (int i = 0; i < data.size(); i++) {
        auto point = data[i];
        vector<double> pointVars = data[i].first;
        double y = data[i].second;
        double x;
        for (int j = 0; j < pointVars.size(); j++) {
            x = pointVars[j];
            totalError += 2 * (y - (predict(weights, b, data))) * (-x);
        }
        partials.push_back(totalError / data.size());
    }
    return partials; 
}

double sum_partials(vector<double> weights, double b,  vector<pair<vector<double>, double>> data) {
    double tot = 0;
    vector<double> partials = partial(weights, b, data);
    for (double p: partials) {
        tot += abs(p);
    }
    return tot;
}

// double partialB_numerical(vector<double> weights, double b,  vector<pair<double,double>> data) {
//     double db = 0.00001;
//     return (error(weights, b + db, data) - error(weights, b, data)) / db; 
// } 

double partialB(vector<double> weights, double b,  vector<pair<vector<double>, double>> data) {
    double totalError = 0;
    for (int i = 0; i < data.size(); i++) {
        auto point = data[i];
        vector<double> pointVars = data[i].first;
        double y = data[i].second;
        double x;
        for (int j = 0; j < pointVars.size(); j++) {
            x = pointVars[j];
            totalError += 2 * (y - (predict(weights, b, data))) * (-1);
        }
    }
    return totalError / data.size(); 
} 


// Old minimize_err function (commented out - using the one below with loss tracking instead)

pair<vector<double>, double> minimize_err(vector<double> weights, double b, vector<pair<vector<double>, double>> data){
    vector<double> weightsP = weights;
    vector<double> partialsM = partial(weights, b, data);
    double parB = partialB(weights, b, data);
    double bp = b;
   

    int i = 0;
    while (error(weightsP, bp, data) > 0.001) {
        partialsM = partial(weightsP, b, data);
        parB = partialB(weightsP, bp, data);

        if (abs(parB) + sum_partials(weightsP, b, data) < 1e-5) break;


        int* j = new int(0);
        for (double& mp: weightsP) {
            mp -= (0.001  * partialsM[*j]);
            j++;
        }
        delete j;

        bp -= (0.001  * parB);
        i++;
    }

    return {partialsM, bp};
}





pair<vector<double>, double> minimize_err_display(vector<double> weights, double b, vector<pair<vector<double>, double>> data){
    vector<double> weightsP = weights;
    vector<double> partialsM = partial(weights, b, data);
    double parB = partialB(weights, b, data);
    double bp = b;
   

    int i = 0;
    while (error(weightsP, bp, data) > 0.01) {
        partialsM = partial(weightsP, b, data);
        parB = partialB(weightsP, bp, data);

        if (i % 100000 == 0) cout << "Loss: " << error(weightsP, bp, data) << endl;

        if (abs(parB) + sum_partials(weightsP, b, data) < 1e-5) break;


        int* j = new int(0);
        for (double& mp: weightsP) {
            mp -= (0.001  * partialsM[*j]);
            j++;
        }
        delete j;

        bp -= (0.001  * parB);
        i++;
    }

    return {partialsM, bp};
}

void printPredic(vector<double> weights, double b, vector<pair<vector<double>, double>> data, vector<double> features) {
    // Create a temporary data point with the given features
    vector<pair<vector<double>, double>> tempData = {{features, 0}};
    cout << "Features: ";
    for (double f : features) cout << f << " ";
    cout << endl;
    cout << "Prediction: " << predict(weights, b, tempData) << endl;
    cout << "Sigmoid output: " << sigmoid(weights, b, tempData) << endl;
    cout << "Error on full dataset: " << error(weights, b, data) << endl;
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
    cout << "Initial sigmoid: " << sigmoid(weights, b, data) << endl;

    // Train the model
    pair<vector<double>, double> res = minimize_err(weights, b, data);
    
    cout << "Final bias: " << res.second << endl;
    cout << "Final error: " << error(res.first, res.second, data) << endl;
}