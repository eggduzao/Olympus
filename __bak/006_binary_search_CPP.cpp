// binary_search.cpp — Binary search in a sorted array (single-file C++17 program)
//
// Build:
//   g++ -O2 -Wall -Wextra -std=c++17 -o binary_search binary_search.cpp
//
// Run:
//   ./binary_search input.txt 1 9 15 2
//
// Behavior:
//   • Reads one integer per line from input.txt (ignores malformed lines).
//   • Sorts values ascending.
//   • For each target (CLI args after the file), prints its 1-based index if found; otherwise “not found”.
//   • If no targets are provided, uses demo targets: 1, 9, 15, 2.
//
// Notes:
//   • Time: O(n log n) to sort + O(log n) per query. Space: O(n) for values.
//   • Values are int64_t; adjust if you need bigger ranges.
//   • For very large inputs, consider memory-mapping or on-disk indices.

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// ---- trim helpers (minimal) ----
static inline std::string_view ltrim(std::string_view s) {
    while (!s.empty() && (s.front()==' ' || s.front()=='\t' || s.front()=='\r')) s.remove_prefix(1);
    return s;
}
static inline std::string_view rtrim(std::string_view s) {
    while (!s.empty() && (s.back()==' ' || s.back()=='\t' || s.back()=='\r')) s.remove_suffix(1);
    return s;
}
static inline std::string_view trim(std::string_view s) { return rtrim(ltrim(s)); }

// ---- parse int64 from a whole line; return std::nullopt if bad ----
static std::optional<int64_t> parse_i64_line(std::string_view line) {
    line = trim(line);
    if (line.empty()) return std::nullopt;

    // strtoll needs a c-string; make a copy
    std::string tmp(line);
    const char* p = tmp.c_str();
    char* end = nullptr;
    errno = 0;
    long long v = std::strtoll(p, &end, 10);
    if (errno != 0) return std::nullopt;

    // ensure no trailing junk (allow trailing spaces already trimmed)
    if (end && *end != '\0') return std::nullopt;

    return static_cast<int64_t>(v);
}

// ---- iterative binary search; return 1-based index if found, else -1 ----
static long bin_search(const std::vector<int64_t>& a, int64_t target) {
    if (a.empty()) return -1;
    std::size_t lo = 0, hi = a.size() - 1;
    while (lo <= hi) {
        std::size_t mid = lo + (hi - lo) / 2;
        int64_t v = a[mid];
        if (v == target) return static_cast<long>(mid + 1); // 1-based
        if (v < target)  lo = mid + 1;
        else {
            if (mid == 0) break; // prevent underflow
            hi = mid - 1;
        }
    }
    return -1;
}

// ---- load integers from file; ignore malformed lines (warn) ----
static bool load_values_sorted(const std::string& path, std::vector<int64_t>& out) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Error: cannot open '" << path << "'\n";
        return false;
    }
    std::string line;
    std::size_t lineno = 0;
    out.clear();
    out.reserve(4096);

    while (std::getline(in, line)) {
        ++lineno;
        auto maybe = parse_i64_line(line);
        if (maybe.has_value()) {
            out.push_back(*maybe);
        } else {
            // comment the next line if you prefer silence
            std::cerr << "[skip] line " << lineno << " not an integer: " << line << "\n";
        }
    }
    std::sort(out.begin(), out.end());
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr
            << "Usage: " << argv[0] << " <input.txt> [targets ...]\n"
            << "Example: " << argv[0] << " input.txt 1 9 15 2\n";
        return 2;
    }

    const std::string input_path = argv[1];
    std::vector<int64_t> values;
    if (!load_values_sorted(input_path, values)) return 1;

    std::cout << "Loaded " << values.size() << " numbers from " << input_path << "\n";

    // If no targets supplied, use demo set
    if (argc == 2) {
        const int64_t demo[] = {1, 9, 15, 2};
        for (int64_t t : demo) {
            long idx = bin_search(values, t);
            if (idx > 0) std::cout << "Target " << t << ": found at index " << idx << "\n";
            else         std::cout << "Target " << t << ": not found\n";
        }
        return 0;
    }

    // Parse targets from argv
    for (int i = 2; i < argc; ++i) {
        std::string_view sv(argv[i]);
        auto maybe = parse_i64_line(sv);
        if (!maybe) {
            std::cerr << "[skip target] not an integer: " << sv << "\n";
            continue;
        }
        long idx = bin_search(values, *maybe);
        if (idx > 0) std::cout << "Target " << *maybe << ": found at index " << idx << "\n";
        else         std::cout << "Target " << *maybe << ": not found\n";
    }
    return 0;
}
