-- BFSShortestPath.hs
-- Breadth-first search for shortest path(s) on an unweighted, UNDIRECTED graph.
--
-- INPUT FORMAT (whitespace-separated):
--   Edge lines : "U  V1 [V2 ...]"  => add undirected edges U—V1, U—V2, ...
--   Query line : "#  SRC DST"      => ask for a shortest path SRC -> DST
--
-- Example:
--   A   B   F
--   B   A   C
--   C   B   D
--   D   C   E
--   E   D   F
--   F   A   E
--   #   A   E
--
-- USAGE
--   runhaskell BFSShortestPath.hs < graph.tsv
--   # or compile:
--   ghc -O2 BFSShortestPath.hs && ./BFSShortestPath < graph.tsv
--   # or provide a file path:
--   runhaskell BFSShortestPath.hs path/to/graph.tsv

{-# LANGUAGE TupleSections #-}

import qualified Data.Map.Strict as M
import qualified Data.Set as S
import qualified Data.Sequence as Q
import           Data.Foldable  (toList)
import           System.Environment (getArgs)
import           System.IO (readFile)
import           Data.Maybe (fromMaybe)

-- ------------------------------- Types ----------------------------------

type Node  = String
type Graph = M.Map Node [Node]

-- ------------------------------- Main -----------------------------------

main :: IO ()
main = do
  args <- getArgs
  contents <- case args of
    (fp:_) -> readFile fp
    _      -> getContents
  let ls = filter (not . null) . map strip $ lines contents
      (g, queries) = parseInput ls
  if null queries
    then putStrLn "No queries found (expected lines like: \"# SRC DST\")."
    else mapM_ (answer g) queries

answer :: Graph -> (Node, Node) -> IO ()
answer g (s, t)
  | s == t = putStrLn s
  | not (M.member s g) = putStrLn $ "No path found from " ++ s ++ " to " ++ t ++ " (source not in graph)"
  | not (M.member t g) = putStrLn $ "No path found from " ++ s ++ " to " ++ t ++ " (destination not in graph)"
  | otherwise =
      case bfsPath g s t of
        Just path -> putStrLn (joinWith " -> " path)
        Nothing   -> putStrLn $ "No path found from " ++ s ++ " to " ++ t

-- ------------------------------ Parsing ---------------------------------

parseInput :: [String] -> (Graph, [(Node, Node)])
parseInput = finalize . foldl step (M.empty :: M.Map Node (S.Set Node), [] :: [(Node,Node)])
  where
    step (g, qs) ln =
      case words ln of
        []           -> (g, qs)
        ("#":a:b:_)  -> (g, qs ++ [(a,b)])
        (u:vs)       -> (foldl (\acc v -> addUndirected acc u v) (ensureNode (ensureNode g u) u) vs, qs)
        _            -> (g, qs)
    finalize (g, qs) = (M.map (toList) (M.map S.toAscList g), qs)

-- Add undirected edge u—v, keep neighbor sets deduplicated
addUndirected :: M.Map Node (S.Set Node) -> Node -> Node -> M.Map Node (S.Set Node)
addUndirected g u v =
  let g'  = ensureNode g u
      g'' = ensureNode g' v
  in M.adjust (S.insert v) u $ M.adjust (S.insert u) v g''

ensureNode :: M.Map Node (S.Set Node) -> Node -> M.Map Node (S.Set Node)
ensureNode g u = if M.member u g then g else M.insert u S.empty g

-- ------------------------------- BFS ------------------------------------

-- Return shortest path [src,...,dst] or Nothing if unreachable.
bfsPath :: Graph -> Node -> Node -> Maybe [Node]
bfsPath g src dst =
  let q0       = Q.singleton src
      visited0 = S.singleton src
      parent0  = M.empty :: M.Map Node Node
  in bfsLoop q0 visited0 parent0
  where
    bfsLoop q visited parent =
      case Q.viewl q of
        Q.EmptyL     -> Nothing
        u Q.:< qrest ->
          let nbrs = fromMaybe [] (M.lookup u g)
              step (q', vis', par', found) v
                | found               = (q', vis', par', True)
                | S.member v vis'     = (q', vis', par', False)
                | v == dst            = (q', S.insert v vis', M.insert v u par', True)
                | otherwise           = (q' Q.|> v, S.insert v vis', M.insert v u par', False)
              (q1, vis1, par1, hit) = foldl step (qrest, visited, parent, False) nbrs
          in if hit
               then Just (reconstruct par1 src dst)
               else bfsLoop q1 vis1 par1

reconstruct :: M.Map Node Node -> Node -> Node -> [Node]
reconstruct parent src dst = reverse (go dst [])
  where
    go cur acc
      | cur == src = src : acc
      | otherwise  = case M.lookup cur parent of
                       Just p  -> go p (cur:acc)
                       Nothing -> acc  -- shouldn't happen if called correctly

-- ------------------------------ Utilities --------------------------------

strip :: String -> String
strip = rstrip . lstrip
  where
    lstrip = dropWhile (`elem` [' ', '\t', '\r'])
    rstrip = reverse . lstrip . reverse

joinWith :: String -> [String] -> String
joinWith _ []     = ""
joinWith _ [x]    = x
joinWith sep (x:xs) = x ++ concatMap (sep ++) xs

