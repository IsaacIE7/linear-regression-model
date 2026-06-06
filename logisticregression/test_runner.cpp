#include <iostream>
#include <vector>
#include <cmath>
#include <utility>
#include <random>
#include <chrono>
#include <iomanip>
#include <string>

using namespace std;

double compute_z(const vector<double>& weights, double b, const vector<double>& features) {
    if (weights.size() != features.size()) {
        throw invalid_argument("weights and features must match in size");
    }
    double total = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
        total += weights[i] * features[i];
    }
    return total + b;
}

double sigmoid(const vector<double>& weights, double b, const vector<double>& features) {
    return 1.0 / (1.0 + exp(-1.0 * compute_z(weights, b, features)));
}

vector<double> partial(const vector<double>& weights, double b, const vector<pair<vector<double>, double>>& data) {
    vector<double> gradient(weights.size());
    for (const auto& point : data) {
        const vector<double>& features = point.first;
        double y = point.second;
        double p = sigmoid(weights, b, features);
        for (size_t i = 0; i < weights.size(); ++i) {
            gradient[i] += (p - y) * features[i];
        }
    }
    for (size_t i = 0; i < gradient.size(); ++i) gradient[i] /= data.size();
    return gradient;
}

double partialB(const vector<double>& weights, double b, const vector<pair<vector<double>, double>>& data) {
    double total = 0;
    for (const auto& point : data) {
        const vector<double>& features = point.first;
        double y = point.second;
        total += (sigmoid(weights, b, features) - y);
    }
    return total / data.size();
}

double error(const vector<double>& weights, double b, const vector<pair<vector<double>, double>>& data) {
    double totalError = 0.0;
    for (const auto& point : data) {
        const vector<double>& features = point.first;
        double y = point.second;
        double s = sigmoid(weights, b, features);
        // clamp for safety
        s = min(max(s, 1e-12), 1.0 - 1e-12);
        totalError += -1.0 * (y * log(s) + (1 - y) * log(1 - s));
    }
    return totalError / data.size();
}

pair<vector<double>, double> train_gd(vector<double> weights, double b,
                                      const vector<pair<vector<double>, double>>& data,
                                      double lr, int max_iter, double tol) {
    for (int it = 0; it < max_iter; ++it) {
        vector<double> grad = partial(weights, b, data);
        double gb = partialB(weights, b, data);
        double max_change = 0.0;
        for (size_t i = 0; i < weights.size(); ++i) {
            double delta = lr * grad[i];
            weights[i] -= delta;
            max_change = max(max_change, fabs(delta));
        }
        double deltaB = lr * gb;
        b -= deltaB;
        max_change = max(max_change, fabs(deltaB));
        if (max_change < tol) break;
    }
    return {weights, b};
}

int main(int argc, char** argv) {
    // Usage: test_runner <num_params> <num_samples>
    int num_params = 2;
    int num_samples = 500;
    if (argc >= 2) num_params = stoi(argv[1]);
    if (argc >= 3) num_samples = stoi(argv[2]);

    // RNG
    std::mt19937_64 rng(123456);
    std::uniform_real_distribution<double> ud(-1.0, 1.0);

    // generate true weights and bias
    vector<double> true_w(num_params);
    for (int i = 0; i < num_params; ++i) true_w[i] = ud(rng);
    double true_b = ud(rng);

    // generate dataset
    vector<pair<vector<double>, double>> data;
    data.reserve(num_samples);
    for (int s = 0; s < num_samples; ++s) {
        vector<double> features(num_params);
        for (int j = 0; j < num_params; ++j) features[j] = ud(rng);
        double p = 1.0 / (1.0 + exp(- ( [&](){ double t=0; for (int k=0;k<num_params;++k) t+= true_w[k]*features[k]; return t + true_b; } )() ));
        double label = p >= 0.5 ? 1.0 : 0.0;
        data.push_back({features, label});
    }

    // init weights
    vector<double> weights(num_params, 0.0);
    double b = 0.0;

    cout << fixed << setprecision(6);
    cout << "Test: params=" << num_params << " samples=" << num_samples << "\n";
    cout << "True bias: " << true_b << "\n";
    cout << "Initial error: " << error(weights, b, data) << "\n";

    auto t0 = chrono::high_resolution_clock::now();
    auto res = train_gd(weights, b, data, 0.5, 2000, 1e-6);
    auto t1 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = t1 - t0;

    double final_err = error(res.first, res.second, data);
    int correct = 0;
    for (const auto& pt : data) {
        double pred = sigmoid(res.first, res.second, pt.first) >= 0.5 ? 1.0 : 0.0;
        if (pred == pt.second) correct++;
    }
    double acc = (double)correct / data.size();

    cout << "Training time (s): " << elapsed.count() << "\n";
    cout << "Final error: " << final_err << "\n";
    cout << "Accuracy: " << acc << "\n";
    cout << "Learned bias: " << res.second << "\n";
    cout << "---\n";
    return 0;
}
