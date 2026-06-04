#include <iostream>
#include <cmath>
#include <utility>
#include <vector>

using namespace std;

// double m = 2;
// double b = 2;

double predict_linear(double m, double b, double x) {
    return m * x + b;
}



double sigmoid(double m, double b, double x) {
    return 1 / (1 + exp(-1 * predict_linear(m, b, x)));
}

double predict(double m, double b, double x) {
    return (1 / (1 + exp(-1 * predict_linear(m, b, x))) >= 0.5) ? 1 : 0;
}

double error(double m, double b,  vector<pair<double,double>> data) {
    double totalError = 0;
    for (auto& point : data) {
        double x = point.first;
        double y = point.second;
        totalError += -1 * (y * log(sigmoid(m, b, x)) + (1-y) * log(1 - sigmoid(m,b,x)));
    }
    return totalError / data.size();
}

double error_point(double m, double b, double x, double y) {
    return -1 * (y * log(sigmoid(m, b, x)) + (1-y) * log(1 - sigmoid(m,b,x)));
}

double partialM_numerical(double m, double b,  vector<pair<double,double>> data) {
    double dm = 0.00001;
    return (error(m + dm, b, data) - error(m, b, data)) / dm; 
} 

double partialM(double m, double b,  vector<pair<double,double>> data) {
    double totalError = 0;
    for (auto& point : data) {
        double x = point.first;
        double y = point.second;
        totalError += (sigmoid(m, b, x) - y) * x;
    }
    return totalError / data.size(); 
} 

double partialB_numerical(double m, double b,  vector<pair<double,double>> data) {
    double db = 0.00001;
    return (error(m, b + db, data) - error(m, b, data)) / db; 
} 

double partialB(double m, double b,  vector<pair<double,double>> data) {
    double totalError = 0;
    for (auto& point : data) {
        double x = point.first;
        double y = point.second;
        totalError += (sigmoid(m, b, x) - y);
    }
    return totalError / data.size(); 
} 



pair<double, double> minimize_err(double m, double b, vector<pair<double,double>> data){
    double mp = m;
    double bp = b;
    double parM;
    double parB;

    int i = 0;
    while (error(mp, bp, data) > 0.00001) {
        parM = partialM(mp, bp, data);
        parB = partialB(mp, bp, data);

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
    while (error(mp, bp, data) > 0.001) {
        
        if (i % 100000 == 0) cout << "Loss: " << error(mp, bp, data) << endl;
        parM = partialM(mp, bp, data);
        parB = partialB(mp, bp, data);

        if (abs(parB) + abs(parM) < 1e-5) break;

        mp -= (0.01  * parM);
        bp -= (0.01  * parB);
        i++;
    }

    return {mp, bp};
}

void printPredic(double m, double b, vector<pair<double,double>> data, double x) {
    cout << "at x = " << x << ", prediction is " << sigmoid(m, b, x) << endl;
    
    cout << "err = " << error(m, b, data) << endl;
}



int main() {

    // vector<pair<double,double>> data = {{1, 3}, {2, 6}, {3, 9}, {4, 12}}; 
    vector<pair<double,double>> data = {
    {1, 0},
    {2.5, 0},
    {3, 0},
    {4, 0},
    {5, 1},
    {6, 1},
    {7, 1}
};

    double m = 0;
    double b = 0;

    pair<double, double> res = minimize_err_display(m, b, data);

    m = res.first;
    b = res.second;

    cout << "m = " << m << ", b = " << b << endl;
    cout << "prediction at x = 3.5 is " << predict(m, b, 3.5) 
    << " actual value is " << "unavailable" <<  endl;

    for(double x = 0; x <= 8; x += 0.5)
{
    cout << x << " -> "
         << predict(m,b,x) << endl;
}
}