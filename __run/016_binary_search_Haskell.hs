-- BinarySearch.hs
-- Haskell (base only) — Binary search (first occurrence) on a sorted ascending array.
--
-- Build & run (GHC):
--   ghc -O2 BinarySearch.hs -o binsearch
--   ./binsearch <target> [path/to/file]
--   # or via STDIN:
--   printf "1\n3\n4\n7\n9\n11\n15\n" | ./binsearch 11
--
-- Behavior:
--   • O(log N) over a random-access array (Data.Array from base).
--   • Returns the FIRST index if duplicates exist (stable-left).
--   • On hit : prints  FOUND <target> at index <i> (1-based)
--   • On miss: prints  NOT FOUND <target>. Insertion index <i> (1-based), between <left> and <right>
--   • Validates non-decreasing order; exits with error if violated.
--
-- Input:
--   - One integer per line (ignored if not parseable; a warning is printed to stderr).
--
-- Notes:
--   - No external packages needed (uses Data.Array in base).
--   - Works for large inputs; memory use ~ O(N).

module Main (main) where

import System.Environment (getArgs)
import System.Exit (exitFailure)
import System.IO (hPutStrLn, stderr, withFile, IOMode(ReadMode))
import Data.Array (Array, (!), listArray, bounds)
import Data.Char (isSpace)
import Data.Maybe (mapMaybe)

-- ------------- Utilities ---------------

trim :: String -> String
trim = dropWhile isSpace . dropWhileEnd isSpace
  where
    dropWhileEnd p = reverse . dropWhile p . reverse

readMaybeInt :: String -> Maybe Integer
readMaybeInt s =
  case reads s :: [(Integer, String)] of
    [(v, rest)] | all isSpace rest -> Just v
    _ -> Nothing

warn :: String -> IO ()
warn = hPutStrLn stderr

die :: String -> IO a
die msg = do
  hPutStrLn stderr msg
  exitFailure

-- ------------- Input -------------------

readNumbersFromHandle :: String -> IO [Integer]
readNumbersFromHandle content = go (zip [1..] (lines content)) []
  where
    go [] acc = pure (reverse acc)
    go ((ln, raw):xs) acc =
      let s = trim raw
      in if null s
           then go xs acc
           else case readMaybeInt s of
                  Just v  -> go xs (v:acc)
                  Nothing -> warn ("WARN: skipping non-integer line " ++ show ln ++ ": " ++ s) >> go xs acc

readNumbers :: Maybe FilePath -> IO [Integer]
readNumbers (Just fp) = do
  withFile fp ReadMode $ \h -> do
    content <- readFile fp  -- small optimization: OS caches; fine for scripts
    readNumbersFromHandle content
readNumbers Nothing = do
  content <- getContents
  readNumbersFromHandle content

-- --------- Validation ------------------

ensureAscending :: [Integer] -> Either String ()
ensureAscending xs = go 1 xs
  where
    go _  []           = Right ()
    go _  [_]          = Right ()
    go ix (a:b:rest)
      | b < a   = Left ("ERROR: input not in ascending order at position " ++ show (ix+1) ++ ": " ++ show b ++ " < " ++ show a)
      | otherwise = go (ix+1) (b:rest)

-- --------- Binary search (first) -------

-- Returns: (index_1based, found?, insertion_index_1based)
binarySearchFirst :: Array Int Integer -> Integer -> (Int, Bool, Int)
binarySearchFirst arr target =
  let (lo0, hi0) = bounds arr
      -- search for leftmost occurrence; typical lower_bound
      go lo hi foundIx =
        if lo > hi
          then (foundIx, lo)  -- foundIx (-1 if none), and lower_bound = lo
          else
            let mid = lo + (hi - lo) `div` 2
                v   = arr ! mid
            in if v == target
                 then go lo (mid - 1) mid
                 else if v < target
                        then go (mid + 1) hi foundIx
                        else go lo (mid - 1) foundIx
      (foundIdx0, lb) = go lo0 hi0 (-1)
  in if foundIdx0 >= 0
       then (foundIdx0 + 1, True, lb + 1)
       else (0, False, lb + 1)

-- ---------- Pretty reporting -----------

reportResult :: [Integer] -> Integer -> IO ()
reportResult xs target = do
  case ensureAscending xs of
    Left err -> die err
    Right () -> do
      let n = length xs
          arr = listArray (0, n - 1) xs
          (idx1, found, ins1) = binarySearchFirst arr target
          leftStr  = if ins1 >= 2            then show (arr ! (ins1 - 2)) else "-inf"
          rightStr = if (ins1 - 1) < n       then show (arr ! (ins1 - 1)) else "+inf"
      if found
        then putStrLn $ "FOUND " ++ show target ++ " at index " ++ show idx1 ++ " (1-based)"
        else putStrLn $ "NOT FOUND " ++ show target
                      ++ ". Insertion index " ++ show ins1
                      ++ " (1-based), between " ++ leftStr ++ " and " ++ rightStr

usage :: IO ()
usage = do
  hPutStrLn stderr "Usage:"
  hPutStrLn stderr "  ./binsearch <target> [path/to/file]"
  hPutStrLn stderr "  printf \"1\\n3\\n4\\n7\\n9\\n11\\n15\\n\" | ./binsearch 11"

-- --------------- Main ------------------

main :: IO ()
main = do
  args <- getArgs
  case args of
    (t:rest) ->
      case readMaybeInt t of
        Nothing   -> die ("ERROR: target must be integer, got '" ++ t ++ "'")
        Just targ -> do
          nums <- readNumbers (case rest of { (fp:_) -> Just fp; _ -> Nothing })
          if null nums then die "ERROR: no numeric input provided."
                       else reportResult nums targ
    _ -> do
      die "ERROR: missing <target>\n"
      usage
