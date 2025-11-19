# binsearch.exs — Binary search (lower-bound) over a sorted ascending list.
# Usage:
#   elixir binsearch.exs -- 11 data.txt
#   printf "1\n3\n4\n7\n9\n11\n15\n" | elixir binsearch.exs -- 11
#
# Behavior:
#   • Reads newline-separated integers from a file or STDIN (blank/non-integer lines are skipped with a warning).
#   • Verifies non-decreasing (ascending) order.
#   • Performs lower-bound binary search:
#       - If target exists:  prints  "FOUND <target> at index <1-based>"
#       - Else:              prints  "NOT FOUND <target>. Insertion index <1-based>, between <L> and <R>"
#         (L/R become -inf/+inf at the extremes)

defmodule BinSearch do
  ## ===== Entry =====
  def main(argv) do
    {target_str, rest} = parse_after_dashdash(argv)

    with {:ok, target} <- to_int(target_str),
         {:ok, ints}   <- read_ints(rest)
    do
      ensure_ascending!(ints)

      # Use tuple for O(1) random access (good for large inputs)
      tup = List.to_tuple(ints)
      n   = tuple_size(tup)

      ins = lower_bound(tup, target, 1, n + 1) # half-open [lo,hi)
      found? = ins <= n and elem(tup, ins - 1) == target

      if found? do
        IO.puts("FOUND #{target} at index #{ins}")
      else
        left  = if ins > 1, do: "#{elem(tup, ins - 2)}", else: "-inf"
        right = if ins <= n, do: "#{elem(tup, ins - 1)}", else: "+inf"
        IO.puts("NOT FOUND #{target}. Insertion index #{ins} (1-based), between #{left} and #{right}")
      end
    else
      :no_dashdash ->
        err("Usage: elixir binsearch.exs -- <target-int> [path/to/file]\n" <>
              "       printf \"1\\n3\\n...\" | elixir binsearch.exs -- 11")
        System.halt(1)

      {:error, :no_target} ->
        err("ERROR: missing target integer after \"--\"")
        System.halt(1)

      {:error, {:bad_target, s}} ->
        err("ERROR: target must be an integer, got #{inspect(s)}")
        System.halt(1)

      {:error, {:read, path, reason}} ->
        err("ERROR: cannot read #{inspect(path)}: #{inspect(reason)}")
        System.halt(2)

      {:error, :empty_input} ->
        err("ERROR: no numeric input provided.")
        System.halt(3)

      {:error, {:not_ascending, a, b}} ->
        err("ERROR: input not in ascending order near #{a} then #{b}")
        System.halt(4)
    end
  end

  ## ===== CLI parsing: args after literal "--" =====
  defp parse_after_dashdash(argv) do
    case drop_until_dashdash(argv) do
      :no_dashdash -> {nil, :no_dashdash}
      []           -> {nil, []}
      [t | rest]   -> {t, rest}
    end
  end

  defp drop_until_dashdash([]), do: :no_dashdash
  defp drop_until_dashdash(["--" | tail]), do: tail
  defp drop_until_dashdash([_ | tail]), do: drop_until_dashdash(tail)

  ## ===== IO & parsing =====
  defp read_ints([]) do
    # Read from STDIN
    ints =
      IO.stream(:stdin, :line)
      |> Stream.with_index(1)
      |> Stream.map(fn {line, i} -> {String.trim(line), i} end)
      |> Stream.filter(fn {s, _i} -> s != "" end)
      |> Enum.reduce([], fn {s, i}, acc ->
        case to_int(s) do
          {:ok, v} -> [v | acc]
          {:error, _} ->
            warn("WARN: skipping non-integer line #{i}: #{inspect(s)}")
            acc
        end
      end)
      |> Enum.reverse()

    if ints == [], do: {:error, :empty_input}, else: {:ok, ints}
  end

  defp read_ints([path]) do
    case File.read(path) do
      {:ok, content} ->
        ints =
          content
          |> String.split("\n", trim: false)
          |> Enum.with_index(1)
          |> Enum.reduce([], fn {line, i}, acc ->
            s = String.trim(line)
            cond do
              s == "" -> acc
              true ->
                case to_int(s) do
                  {:ok, v} -> [v | acc]
                  {:error, _} ->
                    warn("WARN: skipping non-integer line #{i}: #{inspect(s)}")
                    acc
                end
            end
          end)
          |> Enum.reverse()

        if ints == [], do: {:error, :empty_input}, else: {:ok, ints}

      {:error, reason} ->
        {:error, {:read, path, reason}}
    end
  end

  defp read_ints(_too_many), do: {:error, :no_target}

  defp to_int(nil), do: {:error, :no_target}
  defp to_int(s) when is_binary(s) do
    s = String.trim(s)

    case Integer.parse(s) do
      {v, ""} -> {:ok, v}
      _       -> {:error, {:bad_target, s}}
    end
  end

  ## ===== Checks =====
  defp ensure_ascending!([]), do: :ok
  defp ensure_ascending!([_]), do: :ok
  defp ensure_ascending!([a, b | rest]) do
    if b < a do
      raise_args({:not_ascending, a, b})
    else
      ensure_ascending!([b | rest])
    end
  rescue
    e in ArgumentError -> reraise e, __STACKTRACE__
  end

  defp raise_args(reason), do: raise(ArgumentError, message: inspect(reason))

  ## ===== Binary search (lower_bound) on a tuple =====
  # lower_bound(tup, x, lo, hi) over half-open [lo, hi)
  # Returns first index i (1-based) with tup[i] >= x, or n+1 if none.
  defp lower_bound(_tup, _x, lo, hi) when lo >= hi, do: lo
  defp lower_bound(tup, x, lo, hi) do
    mid = div(lo + hi, 2)
    v   = elem_safe(tup, mid)

    cond do
      v < x  -> lower_bound(tup, x, mid + 1, hi)
      true   -> lower_bound(tup, x, lo, mid)
    end
  end

  defp elem_safe(tup, i) do
    # tuple is 1-based conceptually here; elem/2 is 0-based
    elem(tup, i - 1)
  end

  ## ===== STDERR helpers =====
  defp warn(msg), do: IO.write(:stderr, msg <> "\n")
  defp err(msg),  do: IO.write(:stderr, msg <> "\n")
end

# Run when invoked as a script:
BinSearch.main(System.argv())
