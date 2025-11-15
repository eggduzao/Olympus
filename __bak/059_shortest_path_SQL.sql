-- BFS shortest path on an unweighted graph (PostgreSQL, via recursive CTE)
-- One-file, runnable script. Creates temp tables, loads the sample graph,
-- treats edges as UNDIRECTED, and answers one or more queries.

-- ===== 1) Schema (temporary) =====
DROP TABLE IF EXISTS edges;
CREATE TEMP TABLE edges (
  u TEXT NOT NULL,
  v TEXT NOT NULL,
  CONSTRAINT edges_pk PRIMARY KEY (u, v)
);

DROP TABLE IF EXISTS queries;
CREATE TEMP TABLE queries (
  src TEXT NOT NULL,
  dst TEXT NOT NULL
);

-- ===== 2) Load graph from the example (A B F means: A->B and A->F) =====
-- We insert both directions to make it undirected.
INSERT INTO edges(u, v) VALUES
  ('A','B'), ('B','A'),
  ('A','F'), ('F','A'),
  ('B','A'), ('A','B'),
  ('B','C'), ('C','B'),
  ('C','B'), ('B','C'),
  ('C','D'), ('D','C'),
  ('D','C'), ('C','D'),
  ('D','E'), ('E','D'),
  ('E','D'), ('D','E'),
  ('E','F'), ('F','E'),
  ('F','A'), ('A','F'),
  ('F','E'), ('E','F');

-- De-duplicate just in case (harmless if PK already enforced)
-- (No-op if the PK blocked duplicates.)

-- ===== 3) Load queries (lines that begin with '# A E' → src=A, dst=E) =====
INSERT INTO queries(src, dst) VALUES
  ('A','E');

-- You can add more queries, e.g.:
-- INSERT INTO queries(src, dst) VALUES ('B','E'), ('A','D');

-- ===== 4) BFS via recursive CTE =====
-- For each (src,dst), expand level-by-level while tracking the path.
-- We avoid revisiting nodes already on the current path to prevent cycles.
WITH RECURSIVE
bfs AS (
  -- seed: start at each query's source
  SELECT
    q.src,
    q.dst,
    q.src AS node,
    ARRAY[q.src] AS path,
    0::INT AS depth
  FROM queries q

  UNION ALL

  -- expand frontier
  SELECT
    b.src,
    b.dst,
    e.v AS node,
    b.path || e.v AS path,
    b.depth + 1 AS depth
  FROM bfs b
  JOIN edges e
    ON e.u = b.node
  -- don't revisit nodes already on the path (prevents cycles)
  WHERE NOT (e.v = ANY(b.path))
),

-- For each query, keep only the first time we hit the destination (shortest)
hits AS (
  SELECT
    src,
    dst,
    path,
    depth,
    ROW_NUMBER() OVER (PARTITION BY src, dst ORDER BY depth, path) AS rn
  FROM bfs
  WHERE node = dst
)

-- ===== 5) Results =====
-- A) Compact result: one row per (src,dst) with the path as text and hop count
SELECT
  src,
  dst,
  depth AS hops,
  ARRAY_TO_STRING(path, ' -> ') AS path_str
FROM hits
WHERE rn = 1
ORDER BY src, dst;

-- B) (Optional) Expanded path with 1-based positions:
-- SELECT src, dst, depth AS hops, pos, node
-- FROM (
--   SELECT h.src, h.dst, h.depth,
--          GENERATE_SUBSCRIPTS(h.path, 1) AS pos,
--          h.path[GENERATE_SUBSCRIPTS(h.path, 1)] AS node
--   FROM hits h
--   WHERE h.rn = 1
-- ) s
-- ORDER BY src, dst, pos;

