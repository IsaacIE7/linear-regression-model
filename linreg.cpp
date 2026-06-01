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

double error(double m, double b, double x, double y) {
    return pow((y - (m * x + b)), 2);
}

double errorCustom(double m, double b, double x, double y) {
    return pow((y - (m * x + b)), 2);
}

double partialM(double m, double b, double x, double y) {
    double dm = 0.00001;
    return (errorCustom(m + dm, b, x, y) - errorCustom(m, b, x, y)) / dm; 
} 

double partialB(double m, double b, double x, double y) {
    double db = 0.00001;
    return (errorCustom(m, b + db, x, y) - errorCustom(m, b, x, y)) / db; 
} 



pair<double, double> minimize_err(double m, double b, double x, double y){
    double mp = m;
    double bp = b;
    double parM;
    double parB;

    while (errorCustom(mp, bp, x, y) > 0.00001) {
        parM = partialM(mp, bp, x, y);
        parB = partialB(mp, bp, x, y);

        mp -= (0.0001  * parM);
        bp -= (0.0001  * parB);
    }

    return {mp, bp};
}

void printPredic(double m, double b, double x, double y) {
    cout << "at x = " << x << ", prediction is " << predict(m, b, x) << endl;
    
    cout << "err = " << error(m, b, x, y) << endl;
}



int main() {

    vector<pair<double,double>> data = {{1, 3}, {2, 5}, {3, 7}, {4, 9}}; 

    double m = 2;
    double b = 2;
    printPredic(m, b, 3, 7);
    pair<double, double> res = minimize_err(m, b, 3, 7);
    cout << "m = " << res.first << ", b = " << res.second << endl;
    cout << "prediction at x = 3 is " << predict(res.first, res.second, 3) 
    << " actual value is " << 7 <<  endl;
}