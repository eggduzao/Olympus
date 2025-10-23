// quicksort_inplace.cpp
// In-place quicksort on an array (C++17).
//
// - Reads numbers (ints or floats) one-per-line from a file path (argv[1]) or stdin.
// - Ignores blank lines and lines starting with '#'
// - Uses iterative quicksort with Hoare partition + median-of-three pivot selection
//   and switches to insertion sort for small partitions.
// - Prints sorted values, formatting integers without trailing ".0".
//
// Build:
//   g++ -O3 -std=gnu++17 -Wall -Wextra -o quicksort_inplace quicksort_inplace.cpp
//
// Run:
//   ./quicksort_inplace numbers.txt
//   cat numbers.txt | ./quicksort_inplace

#include <bits/stdc++.h>
using namespace std;

/* -------------------- String helpers -------------------- */
static inline void ltrim(string &s) {
    size_t i = 0, n = s.size();
    while (i < n && isspace(static_cast<unsigned char>(s[i]))) ++i;
    if (i) s.erase(0, i);
}
static inline void rtrim(string &s) {
    while (!s.empty() && isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
}
static inline void trim(string &s) { ltrim(s); rtrim(s); }

/* -------------------- Input -------------------- */
static bool parse_double_strict(const string &line, double &out) {
    // Use std::stod but ensure full-consumption (except trailing spaces, already trimmed).
    try {
        size_t pos = 0;
        double v = stod(line, &pos);
        if (pos != line.size()) return false;
        if (!isfinite(v)) return false;
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

static vector<double> read_numbers(istream &in) {
    vector<double> v;
    v.reserve(1 << 10);
    string line;
    size_t lineno = 0;
    while (getline(in, line)) {
        ++lineno;
        trim(line);
        if (line.empty() || line[0] == '#') continue;
        double x;
        if (!parse_double_strict(line, x)) {
            cerr << "Invalid numeric line at " << lineno << ": " << line << "\n";
            exit(EXIT_FAILURE);
        }
        v.push_back(x);
    }
    return v;
}

/* -------------------- Sorting primitives -------------------- */

static size_t median_of_three(const vector<double> &a, size_t i, size_t j, size_t k) {
    const double ai = a[i], aj = a[j], ak = a[k];
    if (ai < aj) {
        if (aj < ak) return j;
        else if (ai < ak) return i;
        else return k;
    } else {
        if (ai < ak) return i;
        else if (aj < ak) return j;
        else return k;
    }
}

static void insertion_sort(vector<double> &a, size_t lo, size_t hi) {
    if (hi <= lo) return;
    for (size_t i = lo + 1; i <= hi; ++i) {
        double key = a[i];
        size_t j = i;
        while (j > lo && a[j - 1] > key) {
            a[j] = a[j - 1];
            --j;
        }
        a[j] = key;
    }
}

// Hoare partition: returns index p such that [lo..p] <= pivot and [p+1..hi] >= pivot
static size_t hoare_partition(vector<double> &a, size_t lo, size_t hi) {
    size_t mid = lo + (hi - lo) / 2;
    size_t pidx = median_of_three(a, lo, mid, hi);
    const double pivot = a[pidx];

    size_t i = lo - 1; // will pre-increment
    size_t j = hi + 1; // will pre-decrement
    for (;;) {
        do { ++i; } while (a[i] < pivot);
        do { --j; } while (a[j] > pivot);
        if (i >= j) return j;
        swap(a[i], a[j]);
    }
}

static void quicksort_inplace(vector<double> &a) {
    const size_t n = a.size();
    if (n < 2) return;

    const size_t SMALL = 24; // insertion sort cutoff
    struct Range { size_t lo, hi; };
    vector<Range> st;
    st.reserve(64);
    st.push_back({0, n - 1});

    while (!st.empty()) {
        auto [lo, hi] = st.back(); st.pop_back();
        if (hi <= lo) continue;

        if (hi - lo + 1 <= SMALL) {
            insertion_sort(a, lo, hi);
            continue;
        }

        size_t p = hoare_partition(a, lo, hi);
        size_t left_lo = lo, left_hi = p;
        size_t right_lo = p + 1, right_hi = hi;

        // Process smaller side first (simulate tail recursion elimination).
        size_t left_sz  = (left_hi >= left_lo) ? (left_hi - left_lo + 1) : 0;
        size_t right_sz = (right_hi >= right_lo) ? (right_hi - right_lo + 1) : 0;

        if (left_sz < right_sz) {
            if (right_lo < right_hi) st.push_back({right_lo, right_hi});
            if (left_lo  < left_hi)  st.push_back({left_lo, left_hi});
        } else {
            if (left_lo  < left_hi)  st.push_back({left_lo, left_hi});
            if (right_lo < right_hi) st.push_back({right_lo, right_hi});
        }
    }
}

/* -------------------- Output -------------------- */
static void print_numbers(const vector<double> &a) {
    constexpr double EPS = 1e-9;
    cout.setf(std::ios::fmtflags(0), std::ios::floatfield);
    cout.setf(std::ios::fixed, std::ios::floatfield);
    for (double x : a) {
        double rx = round(x);
        if (fabs(x - rx) < EPS) {
            cout.unsetf(std::ios::floatfield);
            cout << fixed << setprecision(0) << x << "\n";
        } else {
            cout.unsetf(std::ios::floatfield);
            // Use default formatting; or setprecision(15) to be explicit:
            cout << setprecision(15) << x << "\n";
        }
    }
}

/* -------------------- Main -------------------- */
int main(int argc, char **argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    vector<double> data;
    if (argc >= 2) {
        ifstream fin(argv[1]);
        if (!fin) {
            cerr << "Failed to open file: " << argv[1] << "\n";
            return EXIT_FAILURE;
        }
        data = read_numbers(fin);
    } else {
        data = read_numbers(cin);
    }

    quicksort_inplace(data);
    print_numbers(data);
    return 0;
}

