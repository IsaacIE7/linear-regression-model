#include <utility>
#include <vector>
#include <cmath>
#include <stdexcept>
#include "../matvec/vec.h"
using namespace std;


struct Mat {
    vector<vector<double>> entries;
    int rows;
    int cols;
    pair<int, int> dim;

    Mat(int rows, int cols): //all entries 0 m x n matrix
    entries(rows, vector<double> (cols ,0)), // to initialize vector without naming, vector<type>(size, value)
    dim({rows, cols}),
    rows(rows), cols(cols)
    {}

    Mat(vector<vector<double>> entries): // create matrix with initialized values
    entries(entries), 
    dim({entries.size(), entries[0].size()}),
    rows(entries.size()),
    cols(entries[0].size())
    {}


    Vec mat_mul(const Vec& v) const {
        vector<double> vals(rows, 0);
        Vec res(vals); 
        if (dim.second != v.dim) throw invalid_argument("matrix vector dimension mismatch");
        for (int i = 0; i < entries.size(); i++) {
            double rowSum = 0;
            for (int j = 0; j < entries[0].size(); j++) {
                rowSum += (v[j] * entries[i][j]);
            }
            res[i] = rowSum;
        }
        return res;
    }

    Vec operator*(const Vec& v) const {
        return mat_mul(v);
    }

    Mat operator+(Mat m) const {
        if (dim != m.dim) throw invalid_argument("matrix dimension mismatch");
        vector<vector<double>> vals(rows, vector<double>(cols, 0));
        Mat res(vals);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res.entries[i][j] = entries[i][j] + m.entries[i][j];
            }
        }
        return res;
    }

    Mat operator-(Mat m) const {
        if (dim != m.dim) throw invalid_argument("matrix dimension mismatch");
        vector<vector<double>> vals(rows, vector<double>(cols, 0));
        Mat res(vals);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res.entries[i][j] = entries[i][j] - m.entries[i][j];
            }
        }
        return res;
    }

    Mat operator*(double c) const {
        vector<vector<double>> vals(rows, vector<double>(cols, 0));
        Mat res(vals);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res.entries[i][j] = c * entries[i][j];
            }
        }
        return res;
    }

    Vec row_vec(int row) const {
        if (row < 0 || row >= rows) throw out_of_range("row index out of bounds");
        vector<double> vals(cols);
        for (int j = 0; j < cols; j++) {
            vals[j] = entries[row][j];
        }
        return Vec(vals);
    }

    Vec col_vec(int col) const {
        if (col < 0 || col >= cols) throw out_of_range("column index out of bounds");
        vector<double> vals(rows);
        for (int i = 0; i < rows; i++) {
            vals[i] = entries[i][col];
        }
        return Vec(vals);
    }

    Mat operator*(const Mat& m) {
        if (cols != m.rows) throw invalid_argument("matrix dimension incompatibility");
        Mat res(m.rows, m.cols);
        for (int j = 0; j < m.cols; j++) {
            for (int i = 0; i < rows; i++) {
                res.entries[i][j] = (*this).row_vec(i) * m.col_vec(j);
            }
        }
        return res;
    }


    Mat transpose () const { //lets compiler know that original Mat isnt edited
        Mat res(cols, rows);

        for (int j = 0; j < cols; j++ ) { //swapped loop order , more cache efficient for large datasets
            for (int i = 0; i < rows; i++) {
                res.entries[j][i] = entries[i][j];
            }
        }
        return res;
    }

    Vec to_vector() const {
        if (cols != 1) throw invalid_argument("cannot convert");
        vector<double> vals(rows);
        for (int i = 0; i < rows; i++) {
            vals[i] = entries[i][0];
        }
        return Vec(vals);
    }

    
};