#include <iostream>
#include <cmath>
#include <utility>
#include <vector>

using namespace std;

// double m = 2;
// double b = 2;

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
            totalError += 2 * (y - (predict(weights, b, x))) * (-x);
        }
        partials.push_back(totalError / data.size());
    }
    return partials; 
}

double partialB_numerical(vector<double> weights, double b,  vector<pair<double,double>> data) {
    double db = 0.00001;
    return (error(weights, b + db, data) - error(weights, b, data)) / db; 
} 

double partialB(vector<double> weights, double b,  vector<pair<vector<double>, double>> data) {
    double totalError = 0;
    for (int i = 0; i < data.size(); i++) {
        auto point = data[i];
        vector<double> pointVars = data[i].first;
        double y = data[i].second;
        double x;
        for (int j = 0; j < pointVars.size(); j++) {
            x = pointVars[j];
            totalError += 2 * (y - (predict(weights, b, x))) * (-1);
        }
    }
    return totalError / data.size(); 
} 


//pair of weights and bias
pair<vector<double>, double> minimize_err(vector<double> weights, double b, vector<pair<vector<double>, double>> data){
    vector<double> weightsP = weights;
    vector<double> partialsM = partial(weights, b, data);
    double parB = partialB(weights, b, data);
    double bp = b;
   

    int i = 0;
    while (error(weightsP, bp, data) > 0.00001) {
        partialsM = partial_numerical(weightsP, b, data);
        parB = partialB(weightsP, bp, data);

        if (abs(parB) + abs(parM) < 1e-5) break;

        mp -= (0.0001  * parM);
        bp -= (0.0001  * parB);
        i++;
    }

    return {mp, bp};
}

pair<double, double> minimize_err_display(double m, double b, vector<pair<double,double>> data){
    double mp = m;
    double bp = b;
    double parM;
    double parB;

    int i = 0;
    while (error(mp, bp, data) > 0.00001) {
        
        if (i % 100000 == 0) cout << "Loss: " << error(mp, bp, data) << endl;
        parM = partialM(mp, bp, data);
        parB = partialB(mp, bp, data);

        if (abs(parB) + abs(parM) < 1e-5) break;

        mp -= (0.0001  * parM);
        bp -= (0.0001  * parB);
        i++;
    }

    return {mp, bp};
}

void printPredic(double m, double b, vector<pair<double,double>> data, double x) {
    cout << "at x = " << x << ", prediction is " << predict(m, b, x) << endl;
    
    cout << "err = " << error(m, b, data) << endl;
}



int main() {

    // vector<pair<double,double>> data = {{1, 3}, {2, 6}, {3, 9}, {4, 12}}; 
    vector<pair<double,double>> data = {
    {3, 4},
    {6, 9},
    {2, 3},
    {4, 7},
    {5, 6}
};

    double m = 0;
    double b = 0;

    pair<double, double> res = minimize_err_display(m, b, data);
    cout << "m = " << res.first << ", b = " << res.second << endl;
    cout << "prediction at x = 3 is " << predict(res.first, res.second, 3) 
    << " actual value is " << "unavailable" <<  endl;
}