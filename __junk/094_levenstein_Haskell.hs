import System.IO

levenshtein :: String -> String -> Int
levenshtein s1 s2 = dist (length s1) (length s2)
  where
    dist i j
      | i == 0 = j
      | j == 0 = i
      | otherwise =
          minimum [ dist (i - 1) j     + 1
                  , dist i     (j - 1) + 1
                  , dist (i - 1) (j - 1) + cost i j ]
    cost i j = if s1 !! (i - 1) == s2 !! (j - 1) then 0 else 1

-- Memoization wrapper using arrays
import Data.Array

levenshteinMemo :: String -> String -> Int
levenshteinMemo s1 s2 = table ! (m, n)
  where
    m = length s1
    n = length s2
    table = array ((0,0),(m,n))
      [((i,j), dist i j) | i <- [0..m], j <- [0..n]]
    dist i 0 = i
    dist 0 j = j
    dist i j =
      minimum [ table ! (i-1, j) + 1
              , table ! (i, j-1) + 1
              , table ! (i-1, j-1) + if s1 !! (i-1) == s2 !! (j-1) then 0 else 1 ]

main :: IO ()
main = do
  s1 <- getLine
  s2 <- getLine
  print (levenshteinMemo s1 s2)

