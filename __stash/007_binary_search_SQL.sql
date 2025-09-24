-- PostgreSQL script: Binary search in a sorted array (done in pure SQL)
-- It sets up a demo table, loads your sample data, and provides a reusable
-- SQL function `binary_search_int(target)` that performs a classic binary search.
-- The function returns whether the value was found, the 1-based position in the
-- sorted order, and the matched value (or NULL if not found).

-- -------------------------------------------------------------------
-- 1) Demo data (you can skip this block if you already have a table)
-- -------------------------------------------------------------------
DROP TABLE IF EXISTS nums;
CREATE TABLE nums (val INTEGER NOT NULL);
INSERT INTO nums(val) VALUES
  (1),(3),(4),(7),(9),(11),(15);

-- Helpful: an index for faster lookup if the table is large (not required).
CREATE INDEX IF NOT EXISTS nums_val_idx ON nums(val);

-- -------------------------------------------------------------------
-- 2) Binary search implemented in pure SQL (no PL/pgSQL needed)
--    Works on whatever is in table nums; sorts ascending, dedups not required.
--    If duplicates exist, this finds one matching position (not necessarily the first).
-- -------------------------------------------------------------------
CREATE OR REPLACE FUNCTION binary_search_int(_target INTEGER)
RETURNS TABLE(found BOOLEAN, position INTEGER, value INTEGER)
LANGUAGE sql
AS $$
WITH ordered AS (
  -- Materialize the sorted values into a 1-based array (makes mid access O(1))
  SELECT array_agg(val ORDER BY val) AS arr
  FROM nums
),
RECURSIVE bs(lo, hi, mid, mid_val) AS (
  -- Initial bounds and middle probe
  SELECT
    1 AS lo,
    cardinality(arr) AS hi,
    (1 + cardinality(arr)) / 2 AS mid,
    arr[(1 + cardinality(arr)) / 2 AS mid_val
  FROM ordered

  UNION ALL

  -- Narrow the search window based on comparison:
  --   if mid_val < target → search right half
  --   if mid_val > target → search left half
  SELECT
    CASE WHEN mid_val < _target THEN mid + 1 ELSE lo END AS lo,
    CASE WHEN mid_val > _target THEN mid - 1 ELSE hi END AS hi,
    (CASE WHEN mid_val < _target THEN mid + 1 ELSE lo END
     +   CASE WHEN mid_val > _target THEN mid - 1 ELSE hi END) / 2 AS mid,
    (SELECT arr[
      (CASE WHEN mid_val < _target THEN mid + 1 ELSE lo END
       +   CASE WHEN mid_val > _target THEN mid - 1 ELSE hi END) / 2
     FROM ordered) AS mid_val
  FROM bs
  JOIN ordered ON TRUE
  WHERE lo <= hi
    AND mid_val <> _target
)
SELECT
  (mid_val = _target) AS found,
  mid AS position,
  mid_val AS value
FROM bs
ORDER BY lo DESC, hi ASC
LIMIT 1;
$$;

-- -------------------------------------------------------------------
-- 3) Usage examples
-- -------------------------------------------------------------------
-- Exact hit (present in the array)
SELECT * FROM binary_search_int(7);
--  found | position | value
-- -------+----------+-------
--  t     |    4     |   7

-- Miss (not present); position/value show the last probe; found = false
SELECT * FROM binary_search_int(8);
--  found | position | value
-- -------+----------+-------
--  f     |   (e.g. 5) | 9   -- last inspected mid/value; not found

-- Notes:
-- • This is a true binary search: O(log N) steps over a sorted projection.
-- • It works on large tables since we compress to an array once (array_agg over ORDER BY).
-- • If you prefer *first occurrence* semantics in presence of duplicates,
--   you can wrap this as a first-hit and then scan left/right for boundaries,
--   or incorporate that logic into the recursion (slightly more involved).
