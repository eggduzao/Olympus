(* Levenshtein edit distance in OCaml
   Reads two lines from stdin and prints the distance *)

let levenshtein (s1 : string) (s2 : string) : int =
  let m = String.length s1 in
  let n = String.length s2 in
  let dp = Array.make_matrix (m + 1) (n + 1) 0 in

  (* Base cases *)
  for i = 0 to m do
    dp.(i).(0) <- i
  done;
  for j = 0 to n do
    dp.(0).(j) <- j
  done;

  (* Fill DP table *)
  for i = 1 to m do
    for j = 1 to n do
      let cost = if s1.[i - 1] = s2.[j - 1] then 0 else 1 in
      let deletion    = dp.(i - 1).(j) + 1 in
      let insertion   = dp.(i).(j - 1) + 1 in
      let substitution= dp.(i - 1).(j - 1) + cost in
      dp.(i).(j) <- min deletion (min insertion substitution)
    done
  done;

  dp.(m).(n)

let () =
  let s1 = read_line () |> String.trim in
  let s2 = read_line () |> String.trim in
  let d = levenshtein s1 s2 in
  print_int d; print_newline ()

