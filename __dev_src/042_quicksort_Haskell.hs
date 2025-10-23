-- QuicksortInPlace.hs
-- In-place quicksort on an array in Haskell (mutable vectors in the ST/IO monad).
--
-- • Reads integers (one per line) from a file passed as the first CLI arg,
--   or from STDIN if no file is given.
-- • Skips blank lines and lines starting with '#'.
-- • Sorts the numbers in place using an iterative quicksort with Hoare partition
--   and median-of-three pivot selection to mitigate worst cases.
-- • Prints the sorted numbers, one per line, to STDOUT.
--
-- Build (GHC):
--   ghc -O2 QuicksortInPlace.hs -o qsort
-- Run:
--   ./qsort input.txt
--   # or
--   printf "5\n3\n8\n1\n2\n" | ./qsort

{-# LANGUAGE ScopedTypeVariables #-}

module Main (main) where

import qualified Data.Vector as V
import qualified Data.Vector.Mutable as MV
import           Control.Monad (when)
import           Control.Monad.ST (ST, runST)
import           System.Environment (getArgs)
import           System.IO (hGetContents, stdin)
import           System.Exit (die)
import           Text.Read (readMaybe)

-- ---------- Main ----------
main :: IO ()
main = do
  args <- getArgs
  contents <- case args of
    (fp:_) -> readFile fp
    _      -> hGetContents stdin

  let parseLine (lnNo, raw) =
        let s = trim raw
        in if null s || head s == '#'
             then Right Nothing
             else case readMaybe s :: Maybe Integer of
                    Just x  -> Right (Just x)
                    Nothing -> Left $ "line " ++ show lnNo ++ ": not an integer: " ++ show s

      processed = traverse parseLine (zip [1..] (lines contents))
  case processed of
    Left err -> die err
    Right maybes -> do
      let xs = V.fromList [x | Just x <- maybes]
      if V.length xs <= 1
        then V.mapM_ print xs
        else V.mapM_ print (qsortInPlace xs)

-- ---------- In-place quicksort (returns an immutable vector view) ----------
qsortInPlace :: V.Vector Integer -> V.Vector Integer
qsortInPlace vec = runST $ do
  mv <- V.thaw vec
  quicksort mv 0 (MV.length mv - 1)
  V.freeze mv

-- Iterative quicksort using a manual stack to avoid deep recursion
quicksort :: MV.MVector s Integer -> Int -> Int -> ST s ()
quicksort mv lo0 hi0 = loop [(lo0, hi0)]
  where
    loop [] = pure ()
    loop ((lo, hi):stk)
      | lo >= hi  = loop stk
      | otherwise = do
          p <- hoarePartition mv lo hi
          -- Process smaller side first to keep stack shallow
          let leftLo = lo; leftHi = p
              rightLo = p + 1; rightHi = hi
              leftSize  = leftHi - leftLo + 1
              rightSize = rightHi - rightLo + 1
              stk' = if leftSize < rightSize
                       then pushIf rightLo rightHi (pushIf leftLo leftHi stk)
                       else pushIf leftLo leftHi (pushIf rightLo rightHi stk)
          loop stk'

    pushIf a b s = if a < b then (a,b):s else s

-- Hoare partition with median-of-three pivot value
hoarePartition :: MV.MVector s Integer -> Int -> Int -> ST s Int
hoarePartition mv lo hi = do
  let mid = lo + ((hi - lo) `div` 2)
  a <- MV.read mv lo
  b <- MV.read mv mid
  c <- MV.read mv hi
  let pivot = median3 a b c
  let go i j = do
        i' <- advanceL i
        j' <- advanceR j
        if i' >= j'
          then pure j'
          else MV.swap mv i' j' >> go i' j'
      -- advance left index while < pivot
      advanceL i = do
        let i1 = i + 1
        xi <- MV.read mv i1
        if xi < pivot then advanceL i1 else pure i1
      -- advance right index while > pivot
      advanceR j = do
        let j1 = j - 1
        xj <- MV.read mv j1
        if xj > pivot then advanceR j1 else pure j1
  go (lo - 1) (hi + 1)

median3 :: Ord a => a -> a -> a -> a
median3 x y z
  | x <= y && y <= z = y
  | z <= y && y <= x = y
  | y <= x && x <= z = x
  | z <= x && x <= y = x
  | otherwise        = z

-- ---------- Utils ----------
trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace
  where
    isSpace c = c `elem` [' ', '\t', '\r', '\n']
    dropWhileEnd p = reverse . dropWhile p . reverse

