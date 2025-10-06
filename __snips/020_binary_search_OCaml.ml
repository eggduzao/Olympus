(* binsearch.ml — Binary search (lower-bound) on a sorted ascending list.
   Usage:
     ocamlopt -o binsearch binsearch.ml    # or: ocamlc -o binsearch binsearch.ml
     ./binsearch -- 11 data.txt            # reads integers (one per line) from file
     printf "1\n3\n4\n7\n9\n11\n15\n" | ./binsearch -- 11   # from STDIN

   Behavior:
     • Reads newline-separated integers (skips blank / non-integer lines with a warning).
     • Verifies ascending order (strictly non-decreasing).
     • Performs lower-bound binary search:
         - If target exists, prints:  FOUND <target> at index <1-based>
         - Else prints: NOT FOUND <target>. Insertion index <1-based>, between <L> and <R>
*)

open Printf

(* ---------- utils ---------- *)
let eprintff fmt = ksprintf (fun s -> output_string stderr s; flush stderr) fmt

let string_trim s =
  let is_sp = function ' ' | '\t' | '\r' | '\n' -> true | _ -> false in
  let n = String.length s in
  let i0 =
    let rec f i = if i < n && is_sp s.[i] then f (i + 1) else i in
    f 0
  in
  let i1 =
    let rec f i = if i >= 0 && is_sp s.[i] then f (i - 1) else i in
    f (n - 1)
  in
  if i1 < i0 then "" else String.sub s i0 (i1 - i0 + 1)

let int_of_string_opt s =
  try Some (int_of_string s) with _ -> None

(* ---------- IO ---------- *)
let read_ints_from_in ic =
  let rec loop ln acc =
    match input_line ic with
    | line ->
        let t = string_trim line in
        if t = "" then loop (ln + 1) acc
        else (
          match int_of_string_opt t with
          | Some v -> loop (ln + 1) (v :: acc)
          | None ->
              eprintff "WARN: skipping non-integer line %d: %s\n" ln line;
              loop (ln + 1) acc )
    | exception End_of_file -> List.rev acc
  in
  loop 1 []

let read_ints ?path () =
  match path with
  | None -> read_ints_from_in stdin
  | Some p ->
      let ic = open_in p in
      Fun.protect
        ~finally:(fun () -> close_in_noerr ic)
        (fun () -> read_ints_from_in ic)

(* ---------- checks & search ---------- *)
let ensure_ascending arr =
  let n = Array.length arr in
  for i = 0 to n - 2 do
    if arr.(i+1) < arr.(i) then (
      eprintff "ERROR: input not in ascending order at position %d: %d < %d\n"
        (i+2) arr.(i+1) arr.(i);
      exit 3
    )
  done

(* lower_bound: first i with arr.(i) >= x; returns n if all < x *)
let lower_bound arr x =
  let n = Array.length arr in
  let lo = ref 0 and hi = ref n in
  while !lo < !hi do
    let mid = (!lo + !hi) lsr 1 in
    if arr.(mid) < x then lo := mid + 1 else hi := mid
  done;
  !lo

let report arr target =
  let n = Array.length arr in
  let ins = lower_bound arr target in
  if ins < n && arr.(ins) = target then
    printf "FOUND %d at index %d\n%!" target (ins + 1) (* 1-based for humans *)
  else
    let left  = if ins >= 1 then string_of_int arr.(ins - 1) else "-inf" in
    let right = if ins < n  then string_of_int arr.(ins)       else "+inf" in
    printf "NOT FOUND %d. Insertion index %d (1-based), between %s and %s\n%!"
      target (ins + 1) left right

(* ---------- arg parsing & entrypoint ---------- *)
let usage () =
  eprintff "Usage: %s -- <target-int> [path/to/file]\n" Sys.argv.(0);
  eprintff "       printf \"1\\n3\\n...\" | %s -- 11\n" Sys.argv.(0)

let () =
  (* Find arguments after a literal "--" to be shell-friendly *)
  let after_dashdash =
    let a = Array.to_list Sys.argv in
    let rec drop_until = function
      | [] -> []
      | x :: xs when x = "--" -> xs
      | _ :: xs -> drop_until xs
    in
    drop_until a
  in
  match after_dashdash with
  | target_str :: path_opt ->
      begin match int_of_string_opt target_str with
      | None -> eprintff "ERROR: target must be an integer, got '%s'\n" target_str; usage (); exit 1
      | Some target ->
          let path =
            match path_opt with
            | [] -> None
            | [p] -> Some p
            | _ ->
                eprintff "ERROR: too many arguments after target.\n";
                usage (); exit 1
          in
          let nums = read_ints ?path () in
          if nums = [] then (eprintff "ERROR: no numeric input provided.\n"; exit 2);
          let arr = Array.of_list nums in
          ensure_ascending arr;
          report arr target
      end
  | _ ->
      usage (); exit 1
