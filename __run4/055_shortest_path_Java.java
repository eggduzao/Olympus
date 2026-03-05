import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.regex.Pattern;

/**
 * Breadth-First Search (shortest path on an unweighted graph) — single-file Java program.
 *
 * Input format (flexible):
 * - Each non-empty, non-comment line lists a node followed by its neighbors.
 *   Delimiters: tabs / spaces / commas are all accepted.
 *     A   B   F
 *     B   A   C
 * - Query lines start with '#', followed by source and target:
 *     #  A  E
 *   (You can include multiple query lines; each will be answered.)
 *
 * CLI examples:
 *   javac BFSShortestPath.java && java BFSShortestPath graph.txt
 *   java BFSShortestPath --from A --to E graph.txt
 *   cat graph.txt | java BFSShortestPath -
 *
 * Notes:
 * - Defaults to UNDIRECTED graph. To use directed edges add --directed.
 * - If no query lines are present, you must provide --from and --to.
 */
public class BFSShortestPath {

    private static final Pattern DELIMS = Pattern.compile("[,\\t ]+");

    private static class Args {
        boolean directed = false;
        String input = "-"; // "-" means stdin
        String src = null;
        String dst = null;
    }

    private static Args parseArgs(String[] argv) {
        Args a = new Args();
        for (int i = 0; i < argv.length; i++) {
            String t = argv[i];
            switch (t) {
                case "--directed":
                    a.directed = true;
                    break;
                case "--from":
                    if (i + 1 >= argv.length) die("Missing value after --from");
                    a.src = argv[++i];
                    break;
                case "--to":
                    if (i + 1 >= argv.length) die("Missing value after --to");
                    a.dst = argv[++i];
                    break;
                default:
                    if (a.input.equals("-")) a.input = t;
                    else die("Unexpected argument: " + t);
            }
        }
        return a;
    }

    private static void die(String msg) {
        System.err.println("ERROR: " + msg);
        System.exit(2);
    }

    private static List<String> readAllLines(String path) {
        try {
            if (path.equals("-")) {
                BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
                List<String> lines = new ArrayList<>();
                for (String line; (line = br.readLine()) != null; ) lines.add(line);
                return lines;
            } else {
                return Files.readAllLines(Paths.get(path));
            }
        } catch (IOException e) {
            die("Failed to read input (" + path + "): " + e.getMessage());
            return Collections.emptyList(); // unreachable
        }
    }

    private static class Parsed {
        Map<String, Set<String>> graph = new HashMap<>();
        List<String[]> queries = new ArrayList<>();
    }

    private static Parsed parseGraphAndQueries(List<String> lines, boolean directed) {
        Parsed p = new Parsed();

        for (String raw : lines) {
            if (raw == null) continue;
            String line = raw.trim();
            if (line.isEmpty()) continue;

            if (line.startsWith("#")) {
                String rest = line.substring(1).trim();
                if (!rest.isEmpty()) {
                    String[] toks = DELIMS.split(rest);
                    if (toks.length >= 2) {
                        p.queries.add(new String[]{toks[0], toks[1]});
                    }
                }
                continue;
            }

            String[] toks = DELIMS.split(line);
            if (toks.length == 0) continue;

            String u = toks[0];
            ensureNode(p.graph, u);
            for (int i = 1; i < toks.length; i++) {
                String v = toks[i];
                if (v.isEmpty()) continue;
                ensureNode(p.graph, v);
                p.graph.get(u).add(v);
                if (!directed) p.graph.get(v).add(u);
            }
        }
        return p;
    }

    private static void ensureNode(Map<String, Set<String>> g, String n) {
        g.computeIfAbsent(n, k -> new LinkedHashSet<>());
    }

    /** Returns shortest path (src..dst) or null if none. */
    public static List<String> bfsShortestPath(Map<String, Set<String>> g, String src, String dst) {
        if (!g.containsKey(src) || !g.containsKey(dst)) return null;
        if (src.equals(dst)) return Collections.singletonList(src);

        Deque<String> q = new ArrayDeque<>();
        q.add(src);

        Set<String> seen = new HashSet<>();
        seen.add(src);

        Map<String, String> parent = new HashMap<>();

        while (!q.isEmpty()) {
            String u = q.removeFirst();
            for (String v : g.getOrDefault(u, Collections.emptySet())) {
                if (seen.contains(v)) continue;
                seen.add(v);
                parent.put(v, u);
                if (v.equals(dst)) {
                    return reconstruct(parent, src, dst);
                }
                q.addLast(v);
            }
        }
        return null;
    }

    private static List<String> reconstruct(Map<String, String> parent, String src, String dst) {
        List<String> path = new ArrayList<>();
        String cur = dst;
        while (cur != null) {
            path.add(cur);
            if (cur.equals(src)) break;
            cur = parent.get(cur);
        }
        if (!path.get(path.size() - 1).equals(src)) return null; // disconnected
        Collections.reverse(path);
        return path;
    }

    private static String formatPath(List<String> path) {
        return String.join(" -> ", path);
    }

    public static void main(String[] argv) {
        Args args = parseArgs(argv);
        List<String> lines = readAllLines(args.input);
        Parsed parsed = parseGraphAndQueries(lines, args.directed);

        List<String[]> queries = new ArrayList<>(parsed.queries);
        if (queries.isEmpty()) {
            if (args.src == null || args.dst == null) {
                die("No queries in input and --from/--to not provided.");
            }
            queries.add(new String[]{args.src, args.dst});
        }

        for (String[] q : queries) {
            String src = q[0], dst = q[1];
            List<String> path = bfsShortestPath(parsed.graph, src, dst);
            if (path == null) {
                System.out.println("NO PATH: " + src + " -> " + dst);
            } else {
                int hops = path.size() - 1;
                System.out.println("PATH (" + hops + " edges): " + formatPath(path));
            }
        }
    }
}

