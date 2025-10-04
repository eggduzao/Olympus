(*
 * bfs_shortest_path.ml
 * OCaml script: Breadth-First Search (BFS) shortest path on an unweighted, UNDIRECTED graph.
 *
 * INPUT (whitespace separated), one line per record:
 *   Edge line  : "U  V1 [V2 ...]"  -> add undirected edges U—V1, U—V2, ...
 *   Query line : "#  SRC DST"      -> request one shortest path from SRC to DST
 *
 * Example:
 *   A   B   F
 *   B   A   C
 *   C   B   D
 *   D   C   E
 *   E   D   F
 *   F   A   E
 *   #   A   E
 *
 * USAGE
 *   ocaml bfs_shortest_path.ml < graph.tsv
 *   # or (compiled) ./bfs_shortest_path graph.tsv
 *
 * OUTPUT
 *   For each query line, either a path like "A -> B -> C"
 *   or "No path found from SRC to DST"
 *)

(* -------- Utils -------- *)

let is_ws = function ' ' | '\t' | '\n' | '\r' -> true | _ -> false

let split_ws (s : string) : string list =
  let len = String.length s in
  let rec skip i = if i < len && is_ws s.[i] then skip (i + 1) else i in
  let rec take i j =
    if j < len && not (is_ws s.[j]) then take i (j + 1)
    else String.sub s i (j - i), j
  in
  let rec loop i acc =
    let i = skip i in
    if i >= len then List.rev acc
    else
      let tok, j = take i i in
      loop j (tok :: acc)
  in
  loop 0 []

let read_all_lines (ic : in_channel) : string list =
  let rec loop acc =
    match input_line ic with
    | line -> loop (line :: acc)
    | exception End_of_file -> List.rev acc
  in
  loop []

(* -------- Graph model -------- *)

module SMap = Hashtbl.Make (struct
  type t = string
  let equal = String.equal
  let hash = Hashtbl.hash
end)

type graph = (string, string list) SMap.t

let ensure_node (g : graph) (u : string) =
  if not (SMap.mem g u) then SMap.add g u []

let add_neighbor_nodup (g : graph) (u : string) (v : string) =
  let cur =
    match SMap.find_opt g u with
    | None -> []
    | Some xs -> xs
  in
  if List.exists (String.equal v) cur then ()
  else SMap.replace g u (v :: cur)

let add_undirected_edge (g : graph) (u : string) (v : string) =
  ensure_node g u; ensure_node g v;
  add_neighbor_nodup g u v;
  add_neighbor_nodup g v u

(* -------- BFS shortest path -------- *)

let bfs_shortest_path (g : graph) ~(src : string) ~(dst : string)
  : string list option =
  if String.equal src dst then Some [src]
  else
    match SMap.find_opt g src, SMap.find_opt g dst with
    | None, _ | _, None -> None
    | Some _, Some _ ->
      let visited : (string, bool) SMap.t = SMap.create 1024 in
      let parent  : (string, string) SMap.t = SMap.create 1024 in
      let q : string Queue.t = Queue.create () in
      SMap.replace visited src true;
      Queue.add src q;
      let rec loop () =
        if Queue.is_empty q then None
        else
          let u = Queue.take q in
          let neighs = match SMap.find_opt g u with Some xs -> xs | None -> [] in
          let rec scan = function
            | [] -> loop ()
            | v :: vs ->
              if SMap.mem visited v then scan vs
              else begin
                SMap.replace visited v true;
                SMap.replace parent  v u;
                if String.equal v dst then
                  (* reconstruct path from src -> ... -> dst *)
                  let rec build acc cur =
                    if String.equal cur src then src :: acc
                    else
                      match SMap.find_opt parent cur with
                      | None -> src :: acc (* should not happen *)
                      | Some p -> build (cur :: acc) p
                  in
                  Some (build [] v)
                else begin
                  Queue.add v q;
                  scan vs
                end
              end
          in
          scan neighs
      in
      loop ()

(* -------- Parsing -------- *)

type query = { src : string; dst : string }
type parsed = { g : graph; qs : query list }

let parse (lines : string list) : parsed =
  let g = SMap.create 1024 in
  let rec go qs = function
    | [] -> { g; qs = List.rev qs }
    | raw :: rest ->
      let line = String.trim raw in
      if line = "" then go qs rest
      else
        match split_ws line with
        | [] -> go qs rest
        | "#" :: src :: dst :: _ ->
          go ({ src; dst } :: qs) rest
        | u :: vs when vs <> [] ->
          ensure_node g u;
          List.iter (fun v -> add_undirected_edge g u v) vs;
          go qs rest
        | _single ->
          (* A single token line means "isolated node" – ensure presence *)
          (match split_ws line with
           | [u] -> ensure_node g u
           | _ -> ());
          go qs rest
  in
  go [] lines

(* -------- Formatting -------- *)

let join_with_arrow (xs : string list) : string =
  match xs with
  | [] -> ""
  | x :: rest ->
    List.fold_left (fun acc y -> acc ^ " -> " ^ y) x rest

(* -------- Main / entry point -------- *)

let run_channel (ic : in_channel) : unit =
  let lines = read_all_lines ic in
  let { g; qs } = parse lines in
  if qs = [] then
    print_endline "No queries found (expected lines beginning with: \"# SRC DST\")."
  else
    List.iter
      (fun { src; dst } ->
         match bfs_shortest_path g ~src ~dst with
         | Some path -> print_endline (join_with_arrow path)
         | None -> Printf.printf "No path found from %s to %s\n%!" src dst)
      qs

let () =
  match Array.to_list Sys.argv with
  | [_] ->
    (* No file argument: read from STDIN *)
    run_channel stdin
  | [_; file] ->
    let ic = open_in file in
    (try run_channel ic; close_in ic
     with e -> close_in_noerr ic; raise e)
  | _ ->
    prerr_endline "Usage: bfs_shortest_path [graph.tsv]\n\
                   (Reads STDIN if no file is given.)";
    exit 2