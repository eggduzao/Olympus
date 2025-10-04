%% bfs_shortest_path.erl
%% Breadth-First Search (BFS) shortest path on an unweighted, UNDIRECTED graph.
%%
%% INPUT format (whitespace separated), one line per record:
%%   Edge line  : "U  V1 [V2 ...]"  -> add undirected edges U—V1, U—V2, ...
%%   Query line : "#  SRC DST"      -> request one shortest path from SRC to DST
%%
%% Example:
%%   A   B   F
%%   B   A   C
%%   C   B   D
%%   D   C   E
%%   E   D   F
%%   F   A   E
%%   #   A   E
%%
%% RUN (choose one):
%%   1) escript bfs_shortest_path.erl < graph.tsv
%%   2) erlc bfs_shortest_path.erl && \
%%      erl -noshell -s bfs_shortest_path main graph.tsv -s init stop
%%
%% OUTPUT:
%%   For each query line, either "A -> B -> C" or "No path found from SRC to DST".

-module(bfs_shortest_path).
-export([main/1]).

%% ---------- Public entrypoint ----------

main(Args) ->
    Lines =
        case Args of
            []       -> read_all_lines(stdin);
            [File]   -> read_all_lines({file, File});
            _Other   ->
                io:format(standard_error,
                          "Usage: escript bfs_shortest_path.erl [graph.tsv]~n"
                          "       (reads STDIN if no file given)~n", []),
                halt(2)
        end,
    {Graph, Queries} = parse(Lines),
    case Queries of
        [] ->
            io:format("No queries found (expected lines beginning with: \"# SRC DST\").~n");
        _  ->
            lists:foreach(
              fun({Src, Dst}) ->
                  case bfs_shortest_path(Graph, Src, Dst) of
                      {ok, Path} ->
                          io:format("~s~n", [join_with_arrow(Path)]);
                      not_found  ->
                          io:format("No path found from ~s to ~s~n", [Src, Dst])
                  end
              end,
              Queries)
    end.

%% ---------- I/O helpers ----------

read_all_lines(stdin) ->
    read_from_device(standard_io, []);
read_all_lines({file, Path}) ->
    case file:open(Path, [read]) of
        {ok, IoDev} ->
            Lines = read_from_device(IoDev, []),
            ok = file:close(IoDev),
            Lines;
        {error, Reason} ->
            io:format(standard_error, "Failed to open ~s: ~p~n", [Path, Reason]),
            halt(3)
    end.

read_from_device(IoDev, Acc) ->
    case io:get_line(IoDev, "") of
        eof  -> lists:reverse(Acc);
        Line -> read_from_device(IoDev, [Line | Acc])
    end.

%% ---------- Parsing ----------

%% Graph is a map: #{ Node(string) => Neighbors(list of strings, no dups) }
parse(Lines) ->
    parse_lines(Lines, #{}, []).

parse_lines([], Graph, Queries) ->
    {Graph, lists:reverse(Queries)};
parse_lines([Raw | Rest], Graph0, Queries0) ->
    Tokens = string:tokens(Raw, " \t\r\n"),
    case Tokens of
        [] ->
            parse_lines(Rest, Graph0, Queries0);
        ["#", Src, Dst | _] ->
            parse_lines(Rest, Graph0, [{Src, Dst} | Queries0]);
        [U] ->
            %% single node line -> ensure presence
            Graph1 = ensure_node(Graph0, U),
            parse_lines(Rest, Graph1, Queries0);
        [U | Vs] ->
            Graph1 = ensure_node(Graph0, U),
            Graph2 = lists:foldl(fun(V, G) -> add_undirected_edge(G, U, V) end, Graph1, Vs),
            parse_lines(Rest, Graph2, Queries0)
    end.

ensure_node(G, U) ->
    case maps:is_key(U, G) of
        true  -> G;
        false -> maps:put(U, [], G)
    end.

add_undirected_edge(G0, U, V) ->
    G1 = ensure_node(G0, V),
    G2 = add_neighbor_nodup(G1, U, V),
    add_neighbor_nodup(G2, V, U).

add_neighbor_nodup(G, U, V) ->
    Ns = maps:get(U, G, []),
    case lists:member(V, Ns) of
        true  -> G;
        false -> maps:put(U, [V | Ns], G)
    end.

%% ---------- BFS shortest path ----------

bfs_shortest_path(Graph, Src, Dst) when Src =:= Dst ->
    case maps:is_key(Src, Graph) of
        true  -> {ok, [Src]};
        false -> not_found
    end;
bfs_shortest_path(Graph, Src, Dst) ->
    case {maps:is_key(Src, Graph), maps:is_key(Dst, Graph)} of
        {true, true} ->
            Vis0 = maps:put(Src, true, #{}),
            Par0 = #{},
            Q0   = queue:from_list([Src]),
            bfs_loop(Graph, Dst, Q0, Vis0, Par0);
        _ ->
            not_found
    end.

bfs_loop(_Graph, _Dst, Q, _Vis, _Par) when queue:is_empty(Q) ->
    not_found;
bfs_loop(Graph, Dst, Q0, Vis0, Par0) ->
    {{value, U}, Q1} = queue:out(Q0),
    Neighs = maps:get(U, Graph, []),
    case scan_neighbors(Neighs, Graph, U, Dst, Q1, Vis0, Par0) of
        {found, Path}       -> {ok, Path};
        {cont, QN, VN, PN}  -> bfs_loop(Graph, Dst, QN, VN, PN)
    end.

scan_neighbors([], _Graph, _U, _Dst, Q, Vis, Par) ->
    {cont, Q, Vis, Par};
scan_neighbors([V | Vs], Graph, U, Dst, Q, Vis, Par) ->
    case maps:is_key(V, Vis) of
        true ->
            scan_neighbors(Vs, Graph, U, Dst, Q, Vis, Par);
        false ->
            Vis1 = maps:put(V, true, Vis),
            Par1 = maps:put(V, U, Par),
            case V =:= Dst of
                true  -> {found, reconstruct_path(Par1, Src=U, Dst)};
                false -> scan_neighbors(Vs, Graph, U, Dst, queue:in(V, Q), Vis1, Par1)
            end
    end.

%% Reconstruct path from Src (implicit via parents) to Dst.
reconstruct_path(Parents, Src, Dst) ->
    Rev = build_rev(Parents, Dst, [Dst]),
    lists:reverse(Rev).

build_rev(Parents, Cur, Acc) ->
    case maps:get(Cur, Parents, undefined) of
        undefined -> Acc;      %% reached the source (which has no parent)
        P         -> build_rev(Parents, P, [P | Acc])
    end.

%% ---------- Formatting ----------

join_with_arrow([])      -> "";
join_with_arrow([H | T]) ->
    lists:foldl(fun(E, Acc) -> Acc ++ " -> " ++ E end, H, T).

