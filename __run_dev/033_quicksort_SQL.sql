-- quicksort_inplace.sql
-- In-place quicksort on an array… in SQL (PostgreSQL flavor).
-- We emulate an in-memory array with a working table (pos,val) and
-- rearrange positions inside contiguous [lo,hi] slices, iteratively.
--
-- How to run (psql):
--   \i quicksort_inplace.sql
--
-- What this script does:
--   1) Creates a temp array table "numbers(pos,val)".
--   2) Loads sample data (one number per line from the prompt).
--   3) Calls a PL/pgSQL function quicksort_numbers() that:
--        - maintains a stack of [lo,hi] subarrays,
--        - uses median-position pivot,
--        - rearranges rows in-place by rewriting pos in [lo,hi],
--        - switches to insertion-sort (stable reorder) for small slices.
--   4) Prints sorted values.
--
-- Notes:
--   - “In-place” here means we mutate the single table’s `pos` column;
--     we don’t create another numbers table to hold sorted rows.
--   - Handles duplicates stably (ties broken by previous pos).
--   - Works for large inputs (O(n log n) average), but keep in mind
--     SQL UPDATEs are set-oriented, not pointer swaps.

------------------------------------------------------------
-- 0) Cleanup (safe in a scratch session)
------------------------------------------------------------
DROP TABLE IF EXISTS numbers;
DROP FUNCTION IF EXISTS quicksort_numbers();

------------------------------------------------------------
-- 1) Working array table
------------------------------------------------------------
CREATE TEMP TABLE numbers (
  pos  INTEGER PRIMARY KEY,   -- 1-based array index
  val  NUMERIC                -- can be INT/FLOAT; NUMERIC is generic
) ON COMMIT DROP;

------------------------------------------------------------
-- 2) Load sample input (replace with COPY for big inputs)
------------------------------------------------------------
-- Example input from the prompt (order is the initial array):
WITH raw(line) AS (
  VALUES
    ('5'),
    ('3'),
    ('8'),
    ('1'),
    ('2')
),
numz AS (
  SELECT row_number() OVER () AS pos, (line::numeric) AS val
  FROM raw
)
INSERT INTO numbers(pos,val)
SELECT pos,val FROM numz;

-- For file-based loads, do e.g.:
-- CREATE TEMP TABLE staging(s text);
-- \copy staging FROM 'numbers.txt'
-- INSERT INTO numbers(pos,val)
-- SELECT row_number() OVER (), s::numeric FROM staging;

------------------------------------------------------------
-- 3) Quicksort function (in-place via UPDATE of positions)
------------------------------------------------------------
CREATE OR REPLACE FUNCTION quicksort_numbers()
RETURNS void
LANGUAGE plpgsql
AS $func$
DECLARE
  n          integer;
  lo         integer;
  hi         integer;
  mid        integer;
  pivot      numeric;
  cnt_lt     integer;
  cnt_eq     integer;
  cnt_gt     integer;
  SMALL_CUTOFF constant integer := 24;

BEGIN
  -- n = array length
  SELECT count(*) INTO n FROM numbers;
  IF n <= 1 THEN
    RETURN;
  END IF;

  -- Local stack to simulate iterative quicksort
  CREATE TEMP TABLE stack (
    lo integer,
    hi integer
  ) ON COMMIT DROP;

  -- Start with the whole array
  INSERT INTO stack VALUES (1, n);

  WHILE EXISTS (SELECT 1 FROM stack) LOOP
    -- Pop one range (LIFO)
    SELECT lo, hi INTO lo, hi
    FROM stack
    ORDER BY ctid DESC
    LIMIT 1;

    DELETE FROM stack
    WHERE ctid IN (SELECT ctid FROM stack ORDER BY ctid DESC LIMIT 1);

    IF hi <= lo THEN
      CONTINUE;
    END IF;

    -- Small partitions: do a stable "insertion sort" by rewriting pos
    IF (hi - lo + 1) <= SMALL_CUTOFF THEN
      WITH ordered AS (
        SELECT pos AS old_pos,
               row_number() OVER (ORDER BY val, pos) + (lo - 1) AS new_pos
        FROM numbers
        WHERE pos BETWEEN lo AND hi
      )
      UPDATE numbers AS t
      SET pos = o.new_pos
      FROM ordered AS o
      WHERE t.pos = o.old_pos;

      CONTINUE;
    END IF;

    -- Choose pivot at the middle position
    mid := (lo + hi) / 2;
    SELECT val INTO pivot FROM numbers WHERE pos = mid;

    -- Count groups in current window relative to pivot
    SELECT
      sum( CASE WHEN val <  pivot THEN 1 ELSE 0 END ),
      sum( CASE WHEN val =  pivot THEN 1 ELSE 0 END ),
      sum( CASE WHEN val >  pivot THEN 1 ELSE 0 END )
    INTO cnt_lt, cnt_eq, cnt_gt
    FROM numbers
    WHERE pos BETWEEN lo AND hi;

    -- Reorder current slice [lo..hi] so that:
    --   [< pivot][= pivot][> pivot]
    -- Stable within each block via secondary sort by previous pos.
    WITH ranked AS (
      SELECT
        pos AS old_pos,
        CASE
          WHEN val < pivot THEN 1
          WHEN val = pivot THEN 2
          ELSE 3
        END AS bucket,
        pos AS stable_tie
      FROM numbers
      WHERE pos BETWEEN lo AND hi
    ),
    ordered AS (
      SELECT
        old_pos,
        row_number() OVER (
          ORDER BY bucket, stable_tie
        ) AS rnk
      FROM ranked
    ),
    mapped AS (
      SELECT old_pos,
             (lo - 1) + rnk AS new_pos
      FROM ordered
    )
    UPDATE numbers AS t
    SET pos = m.new_pos
    FROM mapped AS m
    WHERE t.pos = m.old_pos;

    -- Push subranges for further partitioning:
    -- Left:  [lo .. lo+cnt_lt-1]
    -- Right: [hi-cnt_gt+1 .. hi]
    IF cnt_lt >= 2 THEN
      INSERT INTO stack VALUES (lo, lo + cnt_lt - 1);
    END IF;
    IF cnt_gt >= 2 THEN
      INSERT INTO stack VALUES (hi - cnt_gt + 1, hi);
    END IF;
    -- (No need to recurse into = pivot block)
  END LOOP;

  -- Normalize positions to 1..n exactly (already are, but be safe)
  WITH renum AS (
    SELECT pos AS old_pos,
           row_number() OVER (ORDER BY pos) AS new_pos
    FROM numbers
  )
  UPDATE numbers AS t
  SET pos = r.new_pos
  FROM renum AS r
  WHERE t.pos = r.old_pos;

END
$func$;

------------------------------------------------------------
-- 4) Execute sort and show result
------------------------------------------------------------
SELECT quicksort_numbers();

-- Sorted output (ascending):
TABLE (
  SELECT val
  FROM numbers
  ORDER BY pos
);

-- To inspect internal array positions (debug):
-- SELECT pos, val FROM numbers ORDER BY pos;

