WITH RECURSIVE
  params AS (
    SELECT 'kitten'::text AS s1, 'sitting'::text AS s2
  ),
  matrix(i, j, cost, dist) AS (
    -- Base case: first cell (0,0)
    SELECT 0, 0, 0, 0
    UNION ALL
    -- Fill first row
    SELECT i, j+1, 0, j+1
    FROM matrix, params
    WHERE i=0 AND j < length((SELECT s2 FROM params))
    UNION ALL
    -- Fill first column
    SELECT i+1, j, 0, i+1
    FROM matrix, params
    WHERE j=0 AND i < length((SELECT s1 FROM params))
    UNION ALL
    -- Fill rest of matrix
    SELECT i+1, j+1,
           CASE WHEN substr((SELECT s1 FROM params), i+1, 1) =
                     substr((SELECT s2 FROM params), j+1, 1)
                THEN 0 ELSE 1 END,
           LEAST(
             (SELECT dist FROM matrix m WHERE m.i=i AND m.j=j+1) + 1, -- deletion
             (SELECT dist FROM matrix m WHERE m.i=i+1 AND m.j=j) + 1, -- insertion
             (SELECT dist FROM matrix m WHERE m.i=i AND m.j=j) + 
             CASE WHEN substr((SELECT s1 FROM params), i+1, 1) =
                       substr((SELECT s2 FROM params), j+1, 1)
                  THEN 0 ELSE 1 END                                  -- substitution
           )
    FROM matrix, params
    WHERE i < length((SELECT s1 FROM params))
      AND j < length((SELECT s2 FROM params))
  )
SELECT dist AS levenshtein_distance
FROM matrix, params
WHERE i = length(s1) AND j = length(s2);

