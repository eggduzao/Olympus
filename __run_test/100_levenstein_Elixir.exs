defmodule Levenshtein do
  def distance(s1, s2) do
    l1 = String.length(s1)
    l2 = String.length(s2)

    # Initialize DP matrix
    d0 =
      for i <- 0..l1, j <- 0..l2, into: %{} do
        cond do
          j == 0 -> {{i, j}, i}
          i == 0 -> {{i, j}, j}
          true -> {{i, j}, 0}
        end
      end

    d =
      Enum.reduce(1..l1, d0, fn i, acc1 ->
        Enum.reduce(1..l2, acc1, fn j, acc2 ->
          cost =
            if String.at(s1, i - 1) == String.at(s2, j - 1) do
              0
            else
              1
            end

          del = Map.get(acc2, {i - 1, j}) + 1
          ins = Map.get(acc2, {i, j - 1}) + 1
          sub = Map.get(acc2, {i - 1, j - 1}) + cost
          val = Enum.min([del, ins, sub])

          Map.put(acc2, {i, j}, val)
        end)
      end)

    Map.get(d, {l1, l2})
  end

  def main do
    s1 = IO.gets("") |> String.trim()
    s2 = IO.gets("") |> String.trim()
    IO.puts(distance(s1, s2))
  end
end

Levenshtein.main()

