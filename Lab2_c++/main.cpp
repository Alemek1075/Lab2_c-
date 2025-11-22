/*
 * Compiler: GCC 13.2.0
 * Standard: C++20
 * Student: Bazylevich Alex
 * Group: K-27
 * Variant: 12 (inclusive_scan)
 */

#include <iostream>
#include <vector>
#include <numeric>
#include <execution>
#include <chrono>
#include <random>
#include <thread>
#include <iomanip>
#include <functional>
#include <algorithm>
#include <cmath>

using namespace std;
using value_type = long long;

void measure_std_policies(const vector<value_type>& input, vector<value_type>& output) {
    cout << "--- Standard Library Policies ---\n";

    auto run_test = [&](string name, auto policy) {
        auto start = chrono::high_resolution_clock::now();
        inclusive_scan(policy, input.begin(), input.end(), output.begin());
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end - start;
        cout << left << setw(20) << name << ": " << duration.count() << " ms\n";
        };

    auto run_default = [&](string name) {
        auto start = chrono::high_resolution_clock::now();
        inclusive_scan(input.begin(), input.end(), output.begin());
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end - start;
        cout << left << setw(20) << name << ": " << duration.count() << " ms\n";
        };

    run_default("No Policy (Default)");
    run_test("seq", execution::seq);
    run_test("par", execution::par);
    run_test("par_unseq", execution::par_unseq);
}

void custom_parallel_inclusive_scan(const vector<value_type>& input, vector<value_type>& output, int K) {
    size_t n = input.size();
    if (n == 0) return;
    if (K <= 0) K = 1;
    if (K > (int)n) K = (int)n;

    size_t chunk_size = n / K;
    size_t remainder = n % K;

    vector<thread> threads;
    threads.reserve(K);

    vector<value_type> tail_sums(K);

    size_t current_start = 0;
    vector<size_t> starts(K);
    vector<size_t> ends(K);

    for (int i = 0; i < K; ++i) {
        starts[i] = current_start;
        size_t current_chunk = chunk_size + (i < remainder ? 1 : 0);
        ends[i] = starts[i] + current_chunk;
        current_start = ends[i];

        threads.emplace_back([&, i]() {
            inclusive_scan(input.begin() + starts[i], input.begin() + ends[i], output.begin() + starts[i]);
            tail_sums[i] = output[ends[i] - 1];
            });
    }

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }
    threads.clear();

    vector<value_type> offsets(K);
    exclusive_scan(tail_sums.begin(), tail_sums.end(), offsets.begin(), 0LL);

    for (int i = 1; i < K; ++i) {
        threads.emplace_back([&, i]() {
            value_type offset = offsets[i];
            for_each(output.begin() + starts[i], output.begin() + ends[i], [offset](value_type& val) {
                val += offset;
                });
            });
    }

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }
}

void run_experiment(size_t data_size) {
    cout << "\n==========================================================\n";
    cout << "Data Size: " << data_size << "\n";
    cout << "==========================================================\n";

    vector<value_type> input(data_size);
    vector<value_type> output(data_size);

    mt19937 gen(42);
    uniform_int_distribution<value_type> dist(1, 10);
    generate(input.begin(), input.end(), [&]() { return dist(gen); });

    measure_std_policies(input, output);

    cout << "\n--- Custom Parallel Algorithm ---\n";
    cout << setw(10) << "K" << setw(20) << "Time (ms)" << "\n";
    cout << string(30, '-') << "\n";

    unsigned int hw_threads = thread::hardware_concurrency();
    if (hw_threads == 0) hw_threads = 4;

    int best_k = 0;
    double best_time = 1e9;

    vector<int> k_values;
    for (int k = 1; k <= (int)hw_threads * 4; k = (k == 0 ? 1 : k * 2)) {
        k_values.push_back(k);
    }

    if (find(k_values.begin(), k_values.end(), hw_threads) == k_values.end()) {
        k_values.push_back(hw_threads);
        sort(k_values.begin(), k_values.end());
    }

    for (int k : k_values) {
        auto start = chrono::high_resolution_clock::now();
        custom_parallel_inclusive_scan(input, output, k);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end - start;

        cout << setw(10) << k << setw(20) << duration.count() << "\n";

        if (duration.count() < best_time) {
            best_time = duration.count();
            best_k = k;
        }
    }

    cout << "\nAnalysis:\n";
    cout << "Best K: " << best_k << "\n";
    cout << "Hardware Threads: " << hw_threads << "\n";
    cout << "Ratio (Best K / HW Threads): " << (double)best_k / hw_threads << "\n";
}

int main() {
    cout << "Bazylevich Alex\nGroup K-27\nVariant 12: inclusive_scan\n";

    vector<size_t> sizes = { 1'000'000, 10'000'000, 50'000'000 };

    for (size_t s : sizes) {
        run_experiment(s);
    }

    return 0;
}