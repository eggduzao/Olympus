(*
  InPlaceQuicksort.ml — OCaml — In-place quicksort on an int array.

  Features
  • Reads integers (whitespace-separated, one-per-line OK) from a file
    if a path is given as the first CLI arg; otherwise reads from STDIN.
  • In-place quicksort using an iterative stack + Hoare partition with
    median-of-three pivot selection to reduce worst-case behavior.
  • Prints sorted numbers, one per line, to STDOUT.

  Build & run (bytecode):
    ocamlc -o qsort InPlaceQuicksort.ml
    ./qsort input.txt
  Or with dune/ocamlopt as you prefer.

  Notes
  • Input tokens must parse as integers; otherwise Scanf will raise.
  • Handles empty input and very large inputs (iterative, not deep recursion).
*)

(* ---------- I/O: read all integers into a list ---------- *)
let read_all_ints (chan : in_channel) : int list =
  let rec loop acc =
    try
      (* " %d" skips arbitrary whitespace before the next integer *)
      let x = Scanf.fscanf chan " %d" (fun i -> i) in
      loop (x :: acc)
    with
    | End_of_file -> List.rev acc
  in
  loop []

(* ---------- Array helpers ---------- *)
let swap (a : int array) (i : int) (j : int) : unit =
  if i <> j then (
    let tmp = a.(i) in
    a.(i) <- a.(j);
    a.(j) <- tmp
  )

let median3 (x : int) (y : int) (z : int) : int =
  if (x <= y && y <= z) || (z <= y && y <= x) then y
  else if (y <= x && x <= z) || (z <= x && x <= y) then x
  else z

(* Hoare partition with median-of-three pivot selection.
   Returns j such that [lo..j] <= pivot and [j+1..hi] >= pivot. *)
let hoare_partition (a : int array) (lo : int) (hi : int) : int =
  let mid = lo + (hi - lo) / 2 in
  let pivot = median3 a.(lo) a.(mid) a.(hi) in
  let i = ref (lo - 1) in
  let j = ref (hi + 1) in
  let rec step () =
    (* advance i while a.(i) < pivot *)
    let rec next_i () =
      incr i;
      if a.(!i) < pivot then next_i ()
    in
    (* retreat j while a.(j) > pivot *)
    let rec next_j () =
      decr j;
      if a.(!j) > pivot then next_j ()
    in
    next_i ();
    next_j ();
    if !i >= !j then
      !j
    else (
      swap a !i !j;
      step ()
    )
  in
  step ()

(* Iterative quicksort using an explicit stack of (lo, hi) segments. *)
let quicksort_in_place (a : int array) : unit =
  let n = Array.length a in
  if n <= 1 then ()
  else
    let stack = ref [ (0, n - 1) ] in
    while !stack <> [] do
      let (lo, hi) = List.hd !stack in
      stack := List.tl !stack;
      if lo < hi then (
        let p = hoare_partition a lo hi in
        (* Push larger segment last (small-first) to reduce max stack size. *)
        let left_lo, left_hi = (lo, p) in
        let right_lo, right_hi = (p + 1, hi) in
        let left_len = left_hi - left_lo + 1 in
        let right_len = right_hi - right_lo + 1 in
        if left_len > 1 || right_len > 1 then
          if left_len < right_len then (
            if left_len > 1 then stack := (left_lo, left_hi) :: !stack;
            if right_len > 1 then stack := (right_lo, right_hi) :: !stack
          ) else (
            if right_len > 1 then stack := (right_lo, right_hi) :: !stack;
            if left_len > 1 then stack := (left_lo, left_hi) :: !stack
          )
      )
    done

(* ---------- Entrypoint ---------- *)
let () =
  let arr =
    if Array.length Sys.argv > 1 then (
      let path = Sys.argv.(1) in
      let ch = open_in path in
      let data =
        try read_all_ints ch
        finally close_in_noerr ch
      in
      Array.of_list data
    ) else
      Array.of_list (read_all_ints stdin)
  in
  quicksort_in_place arr;
  Array.iter (fun x -> Printf.printf "%d\n" x) arr

  