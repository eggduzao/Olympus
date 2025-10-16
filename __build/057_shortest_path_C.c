// bfs_shortest_path.c
// Compile:  gcc -O2 -std=c11 -Wall -Wextra -o bfs bfs_shortest_path.c
// Usage:
//   ./bfs graph.txt
//   ./bfs --from A --to E graph.txt
//   ./bfs --directed graph.txt
//   cat graph.txt | ./bfs -
//
// Input format (any mix of spaces, tabs, or commas as delimiters):
//   A  B  F          # node A has neighbors B and F
//   B  A  C
//   ...
//   # A E            # query lines start with '#': find shortest path A -> E
//
// Notes:
//   • If there are no query lines, provide --from SRC --to DST on CLI.
//   • Default is UNDIRECTED; pass --directed for directed edges.
//   • Multiple query lines are supported; each prints one result.

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>

/* ---------------------------- tiny dynamic vector ---------------------------- */

typedef struct {
    int   *data;
    size_t len, cap;
} VecI;

static void veci_init(VecI *v) { v->data = NULL; v->len = v->cap = 0; }

static void veci_push(VecI *v, int x) {
    if (v->len == v->cap) {
        size_t ncap = v->cap ? v->cap * 2 : 4;
        int *nd = (int*)realloc(v->data, ncap * sizeof(int));
        if (!nd) { perror("realloc"); exit(1); }
        v->data = nd; v->cap = ncap;
    }
    v->data[v->len++] = x;
}

static void veci_free(VecI *v) { free(v->data); v->data = NULL; v->len = v->cap = 0; }

typedef struct {
    char *a, *b;
} PairStr;

typedef struct {
    PairStr *data;
    size_t len, cap;
} VecPair;

static void vecpair_init(VecPair *v) { v->data = NULL; v->len = v->cap = 0; }
static void vecpair_push(VecPair *v, PairStr p) {
    if (v->len == v->cap) {
        size_t ncap = v->cap ? v->cap * 2 : 4;
        PairStr *nd = (PairStr*)realloc(v->data, ncap * sizeof(PairStr));
        if (!nd) { perror("realloc"); exit(1); }
        v->data = nd; v->cap = ncap;
    }
    v->data[v->len++] = p;
}
static void vecpair_free(VecPair *v) {
    for (size_t i = 0; i < v->len; ++i) { free(v->data[i].a); free(v->data[i].b); }
    free(v->data); v->data = NULL; v->len = v->cap = 0;
}

/* ----------------------------- graph structure ------------------------------ */

typedef struct {
    char *name;
    VecI adj;
} Node;

typedef struct {
    Node  *nodes;
    size_t len, cap;
} Graph;

static void graph_init(Graph *g) { g->nodes = NULL; g->len = g->cap = 0; }

static int graph_find(const Graph *g, const char *name) {
    for (size_t i = 0; i < g->len; ++i)
        if (strcmp(g->nodes[i].name, name) == 0) return (int)i;
    return -1;
}

static int graph_add_node(Graph *g, const char *name) {
    if (g->len == g->cap) {
        size_t ncap = g->cap ? g->cap * 2 : 8;
        Node *nn = (Node*)realloc(g->nodes, ncap * sizeof(Node));
        if (!nn) { perror("realloc"); exit(1); }
        g->nodes = nn; g->cap = ncap;
    }
    g->nodes[g->len].name = strdup(name);
    veci_init(&g->nodes[g->len].adj);
    return (int)g->len++;
}

static int graph_get_or_add(Graph *g, const char *name) {
    int idx = graph_find(g, name);
    if (idx >= 0) return idx;
    return graph_add_node(g, name);
}

static void graph_add_edge(Graph *g, int u, int v, bool directed) {
    veci_push(&g->nodes[u].adj, v);
    if (!directed) veci_push(&g->nodes[v].adj, u);
}

static void graph_free(Graph *g) {
    for (size_t i = 0; i < g->len; ++i) {
        free(g->nodes[i].name);
        veci_free(&g->nodes[i].adj);
    }
    free(g->nodes); g->nodes = NULL; g->len = g->cap = 0;
}

/* ------------------------------- string utils ------------------------------- */

static char *str_trim(char *s) {
    while (isspace((unsigned char)*s)) s++;
    if (*s == 0) return s;
    char *end = s + strlen(s) - 1;
    while (end > s && isspace((unsigned char)*end)) end--;
    end[1] = '\0';
    return s;
}

static bool is_stdin_path(const char *p) { return p && strcmp(p, "-") == 0; }

/* tokenization: spaces, tabs, commas */
static const char *DELIMS = " \t,";

/* ---------------------------------- BFS ------------------------------------- */

static int* bfs_parent(const Graph *g, int s, int t) {
    size_t n = g->len;
    int *parent = (int*)malloc(n * sizeof(int));
    bool *seen   = (bool*)calloc(n, sizeof(bool));
    int *queue   = (int*)malloc(n * sizeof(int));
    if (!parent || !seen || !queue) { perror("malloc"); exit(1); }
    for (size_t i = 0; i < n; ++i) parent[i] = -1;

    size_t head = 0, tail = 0;
    queue[tail++] = s; seen[s] = true;

    while (head < tail) {
        int u = queue[head++];
        if (u == t) break;
        const VecI *adj = &g->nodes[u].adj;
        for (size_t i = 0; i < adj->len; ++i) {
            int v = adj->data[i];
            if (!seen[v]) {
                seen[v] = true;
                parent[v] = u;
                queue[tail++] = v;
                if ((size_t)v == (size_t)t) { head = tail; break; }
            }
        }
    }
    free(seen); free(queue);
    return parent; /* caller frees */
}

static void print_path(const Graph *g, int *parent, int s, int t) {
    if (s == t) { printf("PATH (0 edges): %s\n", g->nodes[s].name); return; }
    if (parent[t] == -1) { printf("NO PATH: %s -> %s\n", g->nodes[s].name, g->nodes[t].name); return; }

    /* reconstruct */
    size_t cap = 16, len = 0;
    int *stk = (int*)malloc(cap * sizeof(int));
    for (int cur = t; cur != -1; cur = parent[cur]) {
        if (len == cap) { cap *= 2; stk = (int*)realloc(stk, cap * sizeof(int)); }
        stk[len++] = cur;
        if (cur == s) break;
    }
    if (stk[len - 1] != s) { printf("NO PATH: %s -> %s\n", g->nodes[s].name, g->nodes[t].name); free(stk); return; }
    printf("PATH (%zu edges): ", len - 1);
    for (size_t i = len; i-- > 0;) {
        printf("%s", g->nodes[stk[i]].name);
        if (i) printf(" -> ");
    }
    printf("\n");
    free(stk);
}

/* ------------------------------- CLI parsing -------------------------------- */

typedef struct {
    bool directed;
    const char *input;
    const char *from;
    const char *to;
} Args;

static void usage_and_exit(const char *msg) {
    if (msg) fprintf(stderr, "ERROR: %s\n", msg);
    fprintf(stderr,
        "Usage: bfs [--directed] [--from SRC --to DST] <input path or '-'>\n");
    exit(msg ? 2 : 0);
}

static Args parse_args(int argc, char **argv) {
    Args a = { .directed = false, .input = "-", .from = NULL, .to = NULL };
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--directed") == 0) {
            a.directed = true;
        } else if (strcmp(argv[i], "--from") == 0) {
            if (++i >= argc) usage_and_exit("Missing value after --from");
            a.from = argv[i];
        } else if (strcmp(argv[i], "--to") == 0) {
            if (++i >= argc) usage_and_exit("Missing value after --to");
            a.to = argv[i];
        } else if (argv[i][0] == '-') {
            if (strcmp(argv[i], "-") == 0) {
                a.input = "-";
            } else {
                usage_and_exit("Unknown flag");
            }
        } else {
            a.input = argv[i];
        }
    }
    return a;
}

/* --------------------------------- main ------------------------------------- */

int main(int argc, char **argv) {
    Args args = parse_args(argc, argv);

    FILE *fp = NULL;
    if (is_stdin_path(args.input)) {
        fp = stdin;
    } else {
        fp = fopen(args.input, "r");
        if (!fp) { perror("fopen"); return 2; }
    }

    Graph g; graph_init(&g);
    VecPair queries; vecpair_init(&queries);

    char *line = NULL; size_t cap = 0; ssize_t nread;
    while ((nread = getline(&line, &cap, fp)) != -1) {
        char *s = str_trim(line);
        if (*s == '\0') continue;

        if (*s == '#') {
            /* query line: "# SRC DST" */
            s++;
            while (isspace((unsigned char)*s)) s++;
            if (*s == '\0') continue;
            char *tmp = strdup(s);
            char *tok = strtok(tmp, DELIMS);
            char *src = NULL, *dst = NULL;
            if (tok) { src = strdup(tok); tok = strtok(NULL, DELIMS); }
            if (tok) { dst = strdup(tok); }
            if (src && dst) vecpair_push(&queries, (PairStr){src, dst});
            else { free(src); free(dst); }
            free(tmp);
            continue;
        }

        /* edge line: "U V1 V2 ..." (or just "U" for isolated) */
        char *tmp = strdup(s);
        char *tok = strtok(tmp, DELIMS);
        if (!tok) { free(tmp); continue; }
        int u = graph_get_or_add(&g, tok);

        char *vstr = NULL;
        while ((vstr = strtok(NULL, DELIMS)) != NULL) {
            if (*vstr == '\0') continue;
            int v = graph_get_or_add(&g, vstr);
            graph_add_edge(&g, u, v, args.directed);
        }
        free(tmp);
    }
    free(line);
    if (fp != stdin) fclose(fp);

    /* If no inline queries, require --from/--to */
    if (queries.len == 0) {
        if (!args.from || !args.to) {
            usage_and_exit("No queries in input and --from/--to not provided");
        }
        vecpair_push(&queries, (PairStr){ strdup(args.from), strdup(args.to) });
    }

    /* Execute queries */
    for (size_t i = 0; i < queries.len; ++i) {
        PairStr q = queries.data[i];
        int s = graph_find(&g, q.a);
        int t = graph_find(&g, q.b);
        if (s < 0 || t < 0) {
            printf("NO PATH: %s -> %s (unknown node)\n", q.a, q.b);
            continue;
        }
        int *parent = bfs_parent(&g, s, t);
        print_path(&g, parent, s, t);
        free(parent);
    }

    /* cleanup */
    vecpair_free(&queries);
    graph_free(&g);
    return 0;
}

