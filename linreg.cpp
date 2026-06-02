#include <iostream>
#include <cmath>
#include <utility>
#include <vector>

using namespace std;

// double m = 2;
// double b = 2;

double predict(double m, double b, double x) {
    return m * x + b;
}

double error(double m, double b,  vector<pair<double,double>> data) {
    double totalError = 0;
    for (auto& point : data) {
        double x = point.first;
        double y = point.second;
        totalError += pow((y - (m * x + b)), 2);
    }
    return totalError / data.size();
}

double error_point(double m, double b, double x, double y) {
    return pow((y - (m * x + b)), 2);;
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
        totalError += 2 * (y - (m * x + b)) * (-x);
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
        totalError += 2 * (y - (m * x + b)) * -1;
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
    {1, 8},
    {2, 6},
    {3, 4},
    {4, 2},
    {5, 0}
};

    double m = 0;
    double b = 0;

    pair<double, double> res = minimize_err_display(m, b, data);
    cout << "m = " << res.first << ", b = " << res.second << endl;
    cout << "prediction at x = 3 is " << predict(res.first, res.second, 3) 
    << " actual value is " << "unavailable" <<  endl;
}