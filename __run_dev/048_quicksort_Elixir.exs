# in_place_quicksort.exs — Elixir — “In-place” quicksort on an integer array
#
# What “in-place” means here:
#   Elixir data are immutable; we simulate an in-place array using an ETS table
#   keyed by index (1..N). Swaps and partitions mutate ETS in O(1).
#
# Usage
# -----
#   # from a file (one integer per line)
#   elixir in_place_quicksort.exs input.txt
#
#   # from STDIN
#   elixir in_place_quicksort.exs
#   5
#   3
#   8
#   1
#   2
#   ^D
#
# Output: sorted numbers, one per line.
#
# Notes:
#   - Blank lines and surrounding whitespace are ignored.
#   - Handles large inputs (streams into ETS; iterative quicksort to avoid deep recursion).

defmodule InPlaceQuicksort do
  @table_opts [:set, :protected, {:read_concurrency, true}, {:write_concurrency, true}]

  # ---------- Public entry ----------

  def main(argv \\ System.argv()) do
    ints =
      case argv do
        [] -> read_all_ints_stdin()
        [path] -> read_all_ints_file(path)
        _ ->
          IO.puts(:stderr, "Usage: elixir in_place_quicksort.exs [input.txt]")
          System.halt(1)
      end

    n = length(ints)
    t = :ets.new(:qarr, @table_opts)
    load_ets(t, ints)

    if n > 1, do: quicksort_ets(t, 1, n)

    print_ets(t, n)
    :ets.delete(t)
  end

  # ---------- I/O helpers ----------

  defp read_all_ints_file(path) do
    path
    |> File.stream!([], 1_048_576) # 1 MB chunks for big files
    |> Stream.map(&String.trim/1)
    |> Stream.reject(&(&1 == ""))
    |> Stream.map(&String.to_integer/1)
    |> Enum.to_list()
  end

  defp read_all_ints_stdin() do
    IO.stream(:stdio, :line)
    |> Stream.map(&String.trim/1)
    |> Stream.reject(&(&1 == ""))
    |> Stream.map(&String.to_integer/1)
    |> Enum.to_list()
  end

  defp print_ets(_t, 0), do: :ok
  defp print_ets(t, n) do
    1..n
    |> Enum.each(fn i ->
      case :ets.lookup(t, i) do
        [{^i, v}] -> IO.puts(v)
        _ -> raise "Missing index #{i}"
      end
    end)
  end

  # ---------- ETS “array” ops ----------

  defp load_ets(t, ints) do
    _ =
      ints
      |> Enum.with_index(1)
      |> Enum.each(fn {v, i} -> :ets.insert(t, {i, v}) end)

    :ok
  end

  defp get(t, i) do
    case :ets.lookup(t, i) do
      [{^i, v}] -> v
      _ -> raise "Bad index #{i}"
    end
  end

  defp set(t, i, v), do: :ets.insert(t, {i, v})

  defp swap(_t, i, i), do: :ok
  defp swap(t, i, j) do
    vi = get(t, i)
    vj = get(t, j)
    set(t, i, vj)
    set(t, j, vi)
  end

  # ---------- Quicksort (Hoare partition, iterative) ----------

  defp median3(a, b, c) do
    cond do
      (a <= b and b <= c) or (c <= b and b <= a) -> b
      (b <= a and a <= c) or (c <= a and a <= b) -> a
      true -> c
    end
  end

  # Returns partition boundary j (Hoare scheme) for subarray [lo..hi]
  defp hoare_partition(t, lo, hi) do
    mid = lo + div(hi - lo, 2)
    pivot = median3(get(t, lo), get(t, mid), get(t, hi))
    i0 = lo - 1
    j0 = hi + 1
    hoare_step(t, pivot, i0, j0)
  end

  defp hoare_step(t, pivot, i, j) do
    i1 = next_i(t, pivot, i + 1)
    j1 = next_j(t, pivot, j - 1)

    if i1 >= j1 do
      j1
    else
      swap(t, i1, j1)
      hoare_step(t, pivot, i1, j1)
    end
  end

  defp next_i(t, pivot, i) do
    if get(t, i) < pivot, do: next_i(t, pivot, i + 1), else: i
  end

  defp next_j(t, pivot, j) do
    if get(t, j) > pivot, do: next_j(t, pivot, j - 1), else: j
  end

  # Iterative quicksort with explicit stack to avoid deep recursion on big inputs
  defp quicksort_ets(t, lo, hi) do
    quicksort_loop(t, [{lo, hi}])
  end

  defp quicksort_loop(_t, []), do: :ok
  defp quicksort_loop(t, [{lo, hi} | rest]) when lo < hi do
    p = hoare_partition(t, lo, hi)
    l_lo = lo; l_hi = p
    r_lo = p + 1; r_hi = hi

    l_len = l_hi - l_lo + 1
    r_len = r_hi - r_lo + 1

    # Push smaller segment first to keep stack shallow (tail recursion friendly)
    new_stack =
      if l_len <= r_len do
        s1 = if l_len > 1, do: [{l_lo, l_hi}], else: []
        s2 = if r_len > 1, do: [{r_lo, r_hi}], else: []
        s1 ++ s2 ++ rest
      else
        s1 = if r_len > 1, do: [{r_lo, r_hi}], else: []
        s2 = if l_len > 1, do: [{l_lo, l_hi}], else: []
        s1 ++ s2 ++ rest
      end

    quicksort_loop(t, new_stack)
  end

  defp quicksort_loop(t, [_ | rest]), do: quicksort_loop(t, rest)
end

# When run as a script: execute main/0 with current argv
InPlaceQuicksort.main()

