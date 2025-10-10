// bfs_shortest_path.cpp
// Compile:  g++ -O2 -std=c++17 -Wall -Wextra -o bfs_cpp bfs_shortest_path.cpp
// Usage examples:
//   ./bfs_cpp graph.txt
//   ./bfs_cpp --from A --to E graph.txt
//   ./bfs_cpp --directed graph.txt
//   cat graph.txt | ./bfs_cpp -
//
// Input format (whitespace or commas as delimiters):
//   A  B  F          # node A has neighbors B and F
//   B  A  C
//   ...
//   #  A  E          # query lines start with '#': find shortest path A -> E
//
// Notes:
//   • If there are no query lines, provide --from SRC --to DST on CLI.
//   • Default is UNDIRECTED; pass --directed for directed edges.
//   • Multiple query lines are supported; each prints one result.

#include <bits/stdc++.h>
using namespace std;

struct Graph {
    vector<vector<int>> adj;
    vector<string> names;
    unordered_map<string,int> id;

    int get_or_add(const string& s) {
        auto it = id.find(s);
        if (it != id.end()) return it->second;
        int idx = (int)names.size();
        id.emplace(s, idx);
        names.push_back(s);
        adj.emplace_back();
        return idx;
    }
    int find(const string& s) const {
        auto it = id.find(s);
        return (it == id.end() ? -1 : it->second);
    }
};

struct Args {
    bool directed = false;
    string input  = "-";
    string from, to;
};

static void usage_and_exit(const string& msg = "") {
    if (!msg.empty()) cerr << "ERROR: " << msg << "\n";
    cerr << "Usage: bfs_cpp [--directed] [--from SRC --to DST] <input path or '-'>\n";
    exit(msg.empty() ? 0 : 2);
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        string t = argv[i];
        if (t == "--directed") a.directed = true;
        else if (t == "--from") {
            if (++i >= argc) usage_and_exit("Missing value after --from");
            a.from = argv[i];
        } else if (t == "--to") {
            if (++i >= argc) usage_and_exit("Missing value after --to");
            a.to = argv[i];
        } else if (t == "-") {
            a.input = "-";
        } else if (!t.empty() && t[0] == '-') {
            usage_and_exit("Unknown flag: " + t);
        } else {
            a.input = t;
        }
    }
    return a;
}

/// split on spaces/tabs/commas
static vector<string> split_tokens(const string& line) {
    vector<string> out;
    string cur;
    for (char c : line) {
        if (isspace((unsigned char)c) || c == ',') {
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
        } else cur.push_back(c);
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

static vector<int> bfs_parent(const Graph& g, int s, int t) {
    vector<int> parent(g.adj.size(), -1);
    vector<char> seen(g.adj.size(), 0);
    deque<int> q;
    q.push_back(s); seen[s] = 1;
    while (!q.empty()) {
        int u = q.front(); q.pop_front();
        if (u == t) break;
        for (int v : g.adj[u]) if (!seen[v]) {
            seen[v] = 1; parent[v] = u; q.push_back(v);
            if (v == t) { q.clear(); break; }
        }
    }
    return parent;
}

static void print_path(const Graph& g, const vector<int>& parent, int s, int t) {
    if (s == t) { cout << "PATH (0 edges): " << g.names[s] << "\n"; return; }
    if (parent[t] == -1) { 
        cout << "NO PATH: " << g.names[s] << " -> " << g.names[t] << "\n"; 
        return; 
    }
    vector<int> seq; 
    for (int cur = t; cur != -1; cur = parent[cur]) seq.push_back(cur);
    if (seq.back() != s) { 
        cout << "NO PATH: " << g.names[s] << " -> " << g.names[t] << "\n"; 
        return; 
    }
    reverse(seq.begin(), seq.end());
    cout << "PATH (" << (seq.size()-1) << " edges): ";
    for (size_t i = 0; i < seq.size(); ++i) {
        if (i) cout << " -> ";
        cout << g.names[seq[i]];
    }
    cout << "\n";
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Args args = parse_args(argc, argv);

    unique_ptr<istream> in_holder;
    istream* in = nullptr;
    if (args.input == "-") in = &cin;
    else {
        auto f = make_unique<ifstream>(args.input);
        if (!*f) { cerr << "ERROR: cannot open '" << args.input << "'\n"; return 2; }
        in_holder = move(f);
        in = in_holder.get();
    }

    Graph g;
    vector<pair<string,string>> queries;

    string raw;
    while (getline(*in, raw)) {
        // trim
        auto l = raw;
        // remove leading spaces
        size_t p = l.find_first_not_of(" \t\r\n");
        if (p == string::npos) continue;
        l.erase(0, p);
        // drop trailing spaces
        while (!l.empty() && isspace((unsigned char)l.back())) l.pop_back();
        if (l.empty()) continue;

        if (l[0] == '#') {
            auto toks = split_tokens(l.substr(1));
            if (toks.size() >= 2) queries.emplace_back(toks[0], toks[1]);
            continue;
        }

        auto toks = split_tokens(l);
        if (toks.empty()) continue;
        int u = g.get_or_add(toks[0]);
        for (size_t i = 1; i < toks.size(); ++i) {
            int v = g.get_or_add(toks[i]);
            g.adj[u].push_back(v);
            if (!args.directed) g.adj[v].push_back(u);
        }
    }

    if (queries.empty()) {
        if (args.from.empty() || args.to.empty()) {
            usage_and_exit("No queries in input and --from/--to not provided");
        }
        queries.emplace_back(args.from, args.to);
    }

    for (auto& q : queries) {
        int s = g.find(q.first);
        int t = g.find(q.second);
        if (s < 0 || t < 0) {
            cout << "NO PATH: " << q.first << " -> " << q.second << " (unknown node)\n";
            continue;
        }
        auto parent = bfs_parent(g, s, t);
        print_path(g, parent, s, t);
    }

    return 0;
}

