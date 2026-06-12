#include <vector>
#include <utility>
using namespace std;

struct Vec;

struct Mat {
    vector<vector<double>> entries;
    int rows;
    int cols;
    pair<int, int> dim;
    

    Mat(int rows, int cols); //all entries 0 m x n matrix

    Mat(vector<vector<double>> entries); // create matrix with initialized values


    Vec mat_mul(const Vec& v) const;

    Vec operator*(const Vec& v) const;

    Mat operator+(Mat m) const;

    Mat add_vec_to_row(const Vec& v);

    Mat operator+(double c) const;

    Mat operator-(Mat m) const;

    Mat operator-(double c) const;

    Mat operator*(double c) const;

    Vec row_vec(int row) const;

    Vec col_vec(int col) const;

    Mat operator*(const Mat& m);

    Mat Mat::sigmoid_element_wise() const;

    Mat Mat::log_element_wise() const;

    double Mat::sum_entries() const;

    Mat transpose () const; //lets compiler know that original Mat isnt edited

    Vec to_vector() const;

    
};