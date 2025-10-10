/* binary_search.c — Binary search in a sorted array (single-file C program)
 *
 * Build:
 *   cc -O2 -Wall -Wextra -std=c11 -o binary_search binary_search.c
 *
 * Run:
 *   ./binary_search input.txt 1 9 15 2
 *
 * Behavior:
 *   - Reads one integer per line from input.txt (ignores malformed lines).
 *   - Sorts values ascending (qsort).
 *   - For each target (command-line args after the file), prints its index (1-based) or not found.
 *   - If no targets are provided, uses demo targets: 1, 9, 15, 2.
 *
 * Notes:
 *   • Time: O(n log n) to sort + O(log n) per query.
 *   • Space: O(n) for the array (64-bit integers). For multi-GB inputs, consider chunking,
 *     on-disk indices (SQLite/DuckDB), or memory mapping.
 */

#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ----- dynamic array of int64_t ----- */

typedef struct {
    int64_t *data;
    size_t   size;
    size_t   cap;
} vec_i64;

static void vec_init(vec_i64 *v) {
    v->data = NULL;
    v->size = 0;
    v->cap  = 0;
}

static void vec_free(vec_i64 *v) {
    free(v->data);
    v->data = NULL;
    v->size = v->cap = 0;
}

static bool vec_reserve(vec_i64 *v, size_t need) {
    if (need <= v->cap) return true;
    size_t new_cap = v->cap ? (v->cap + v->cap/2) : 4096;
    if (new_cap < need) new_cap = need;
    int64_t *p = (int64_t*)realloc(v->data, new_cap * sizeof(int64_t));
    if (!p) return false;
    v->data = p;
    v->cap  = new_cap;
    return true;
}

static bool vec_push(vec_i64 *v, int64_t x) {
    if (v->size == v->cap && !vec_reserve(v, v->size + 1)) return false;
    v->data[v->size++] = x;
    return true;
}

/* ----- parsing helpers ----- */

static bool parse_i64(const char *s, int64_t *out) {
    // Trim leading/trailing whitespace
    while (*s == ' ' || *s == '\t' || *s == '\r') s++;
    if (*s == '\0' || *s == '\n') return false;

    errno = 0;
    char *end = NULL;
    long long val = strtoll(s, &end, 10);
    if (errno != 0) return false;

    // Skip trailing spaces
    while (*end == ' ' || *end == '\t' || *end == '\r') end++;
    if (!(*end == '\0' || *end == '\n')) return false;

    *out = (int64_t)val;
    return true;
}

/* ----- qsort comparator ----- */
static int cmp_i64(const void *a, const void *b) {
    const int64_t x = *(const int64_t*)a;
    const int64_t y = *(const int64_t*)b;
    return (x > y) - (x < y);
}

/* ----- iterative binary search (1-based index on success, -1 on not found) ----- */
static long bin_search(const int64_t *arr, size_t n, int64_t target) {
    size_t lo = 0, hi = (n == 0 ? 0 : n - 1);
    while (n && lo <= hi) {
        size_t mid = lo + (hi - lo) / 2;
        int64_t v = arr[mid];
        if (v == target) return (long)(mid + 1); // 1-based
        if (v < target)  lo = mid + 1;
        else {
            if (mid == 0) break; // avoid size_t underflow
            hi = mid - 1;
        }
    }
    return -1;
}

/* ----- load integers from file, ignore bad lines ----- */
static bool load_sorted_file(const char *path, vec_i64 *out) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open '%s': %s\n", path, strerror(errno));
        return false;
    }
    // Larger stdio buffer can help on big inputs
    static char io_buf[1<<20];
    setvbuf(fp, io_buf, _IOFBF, sizeof io_buf);

    char *line = NULL;
    size_t cap = 0;
    ssize_t len = 0;
    size_t lineno = 0;

    while ((len = getline(&line, &cap, fp)) != -1) {
        lineno++;
        int64_t val;
        if (parse_i64(line, &val)) {
            if (!vec_push(out, val)) {
                fprintf(stderr, "Error: out of memory at line %zu\n", lineno);
                free(line);
                fclose(fp);
                return false;
            }
        } else {
            // Non-numeric line: report and skip
            // (Comment the next line if you prefer silence)
            fprintf(stderr, "[skip] line %zu not an integer: %.*s", lineno, (int)len, line);
        }
    }
    free(line);
    fclose(fp);

    // Sort ascending
    if (out->size > 1) qsort(out->data, out->size, sizeof(int64_t), cmp_i64);
    return true;
}

/* ----- main ----- */
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <input.txt> [targets ...]\n"
            "Example: %s input.txt 1 9 15 2\n", argv[0], argv[0]);
        return 2;
    }

    const char *input_path = argv[1];

    vec_i64 vec;
    vec_init(&vec);
    if (!load_sorted_file(input_path, &vec)) {
        vec_free(&vec);
        return 1;
    }
    fprintf(stdout, "Loaded %zu numbers from %s\n", vec.size, input_path);

    // Collect targets
    int arg_targets = argc - 2;
    if (arg_targets <= 0) {
        // Demo defaults
        int64_t demo[] = {1, 9, 15, 2};
        for (size_t i = 0; i < sizeof(demo)/sizeof(demo[0]); ++i) {
            long idx = bin_search(vec.data, vec.size, demo[i]);
            if (idx > 0) {
                printf("Target %" PRId64 ": found at index %ld\n", demo[i], idx);
            } else {
                printf("Target %" PRId64 ": not found\n", demo[i]);
            }
        }
        vec_free(&vec);
        return 0;
    }

    for (int i = 2; i < argc; ++i) {
        int64_t t;
        if (!parse_i64(argv[i], &t)) {
            fprintf(stderr, "[skip target] not an integer: %s\n", argv[i]);
            continue;
        }
        long idx = bin_search(vec.data, vec.size, t);
        if (idx > 0) {
            printf("Target %" PRId64 ": found at index %ld\n", t, idx);
        } else {
            printf("Target %" PRId64 ": not found\n", t);
        }
    }

    vec_free(&vec);
    return 0;
}
