# bfs_shortest_path.exs
# Breadth-First Search (BFS) shortest path on an unweighted, UNDIRECTED graph.
#
# INPUT format (whitespace separated), one line per record:
#   Edge line  : "U  V1 [V2 ...]"  -> add undirected edges U—V1, U—V2, ...
#   Node line  : "U"               -> ensure node exists (even if isolated)
#   Query line : "#  SRC DST"      -> request one shortest path from SRC to DST
#
# Example (TSV-style; tabs or spaces both fine):
#   A   B   F
#   B   A   C
#   C   B   D
#   D   C   E
#   E   D   F
#   F   A   E
#   #   A   E
#
# RUN (choose one):
#   1) elixir bfs_shortest_path.exs < graph.tsv
#   2) elixir bfs_shortest_path.exs graph.tsv
#
# OUTPUT:
#   For each query line, either "A -> B -> C" or "No path found from SRC to DST".

defmodule BFSShortestPath do
  @moduledoc false

  # -------- Entry point --------

  def main(argv) do
    stream =
      case argv do
        [] -> IO.stream(:stdio, :line)
        [path] -> File.stream!(path, [], :line)
        _ ->
          IO.puts(:stderr, "Usage: elixir bfs_shortest_path.exs [graph.tsv]\n(Reads STDIN if no file given)")
          System.halt(2)
      end

    {graph, queries} = parse_stream(stream)

    if queries == [] do
      IO.puts("No queries found (expected lines beginning with: \"# SRC DST\").")
    else
      Enum.each(queries, fn {src, dst} ->
        case bfs_shortest_path(graph, src, dst) do
          {:ok, path} -> IO.puts(Enum.join(path, " -> "))
          :not_found  -> IO.puts("No path found from #{src} to #{dst}")
        end
      end)
    end
  end

  # -------- Parsing (single pass, streaming) --------

  # Graph: %{
  #   "Node" => MapSet.new(["Neighbor1","Neighbor2",...]),
  #   ...
  # }
  # Queries: [{src, dst}, ...]
  defp parse_stream(lines_enum) do
    Enum.reduce(lines_enum, {%{}, []}, fn raw, {graph, queries} ->
      tokens =
        raw
        |> String.trim()
        |> String.split(~r/\s+/, trim: true)

      case tokens do
        [] ->
          {graph, queries}

        ["#", src, dst | _] ->
          {graph, [{src, dst} | queries]}

        [u] ->
          {ensure_node(graph, u), queries}

        [u | vs] ->
          g = ensure_node(graph, u)

          g2 =
            Enum.reduce(vs, g, fn v, acc ->
              acc
              |> add_undirected_edge(u, v)
            end)

          {g2, queries}
      end
    end)
    |> then(fn {g, qs} -> {g, Enum.reverse(qs)} end)
  end

  defp ensure_node(graph, u), do: Map.put_new(graph, u, MapSet.new())

  defp add_undirected_edge(graph, u, v) do
    graph
    |> Map.update(u, MapSet.new([v]), &MapSet.put(&1, v))
    |> Map.update(v, MapSet.new([u]), &MapSet.put(&1, u))
  end

  # -------- BFS shortest path --------

  # Trivial same-node case
  defp bfs_shortest_path(graph, src, dst) when src == dst do
    if Map.has_key?(graph, src), do: {:ok, [src]}, else: :not_found
  end

  defp bfs_shortest_path(graph, src, dst) do
    cond do
      not Map.has_key?(graph, src) -> :not_found
      not Map.has_key?(graph, dst) -> :not_found
      true ->
        visited = MapSet.new([src])
        parents = %{} # child -> parent
        queue   = :queue.from_list([src])
        bfs_loop(graph, dst, queue, visited, parents)
    end
  end

  defp bfs_loop(_graph, _dst, queue, _visited, _parents) do
    case :queue.out(queue) do
      {:empty, _} ->
        :not_found

      {{:value, u}, q1} ->
        neighbors = Map.get(_graph, u, MapSet.new()) |> MapSet.to_list()

        case scan_neighbors(neighbors, u, q1, _visited, _parents, _graph, _dst) do
          {:found, parents2, dst} ->
            {:ok, reconstruct_path(parents2, dst)}

          {:cont, qn, vn, pn} ->
            bfs_loop(_graph, _dst, qn, vn, pn)
        end
    end
  end

  # Visit neighbors of current node `u`; enqueue unseen; capture parent pointers.
  defp scan_neighbors([], _u, q, visited, parents, _graph, _dst),
    do: {:cont, q, visited, parents}

  defp scan_neighbors([v | vs], u, q, visited, parents, graph, dst) do
    if MapSet.member?(visited, v) do
      scan_neighbors(vs, u, q, visited, parents, graph, dst)
    else
      visited1 = MapSet.put(visited, v)
      parents1 = Map.put(parents, v, u)

      if v == dst do
        {:found, parents1, dst}
      else
        scan_neighbors(vs, u, :queue.in(v, q), visited1, parents1, graph, dst)
      end
    end
  end

  # Reconstruct path from parents map (which contains dst->...->src chain).
  defp reconstruct_path(parents, dst) do
    dst
    |> Stream.unfold(fn cur ->
      case cur do
        nil -> nil
        node -> {node, Map.get(parents, node)}
      end
    end)
    |> Enum.reverse()
  end
end

# -- Script entrypoint (Elixir "traditional" for .exs) --
BFSShortestPath.main(System.argv())

