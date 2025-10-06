%% in_place_quicksort.erl — Erlang — In-place quicksort on an integer array.
%%
%% What “in-place” means in Erlang:
%%   Erlang data are immutable, but we can simulate in-place updates efficiently
%%   using an ETS table (index -> value). We partition and swap by mutating ETS.
%%
%% Build:
%%   erlc in_place_quicksort.erl
%%
%% Run (from shell):
%%   # from file
%%   erl -noshell -s in_place_quicksort main "input.txt" -s init stop
%%   # from STDIN (end with Ctrl-D/Ctrl-Z)
%%   erl -noshell -s in_place_quicksort main -s init stop
%%
%% Input format:
%%   One integer per line (blank lines ignored). Whitespace is ok.

-module(in_place_quicksort).
-export([main/0, main/1]).

-define(TABLE_OPTS, [set, protected, {read_concurrency, true}, {write_concurrency, true}]).

%% ---------- Entry points ----------

main() ->
    main([]).

main(Args) ->
    Ints =
        case Args of
            [] -> read_all_ints_stdin();
            [Filename] -> read_all_ints_file(Filename);
            _ -> usage_and_halt()
        end,
    N = length(Ints),
    T = ets:new(qarr, ?TABLE_OPTS),
    load_ets(T, Ints),              % 1..N
    quicksort_ets(T, 1, N),
    print_ets(T, N),
    ets:delete(T),
    ok.

usage_and_halt() ->
    io:format("Usage: erl -noshell -s in_place_quicksort main \"input.txt\" -s init stop~n"
              "   or: erl -noshell -s in_place_quicksort main -s init stop (read from STDIN)~n"),
    halt(1).

%% ---------- I/O helpers ----------

read_all_ints_file(Filename) ->
    {ok, Bin} = file:read_file(Filename),
    lines_to_ints(string:split(binary_to_list(Bin), "\n", all)).

read_all_ints_stdin() ->
    read_stdin_lines([]).

read_stdin_lines(Acc) ->
    case io:get_line("") of
        eof -> lists:reverse(Acc) |> lines_to_ints();
        Line -> read_stdin_lines([Line | Acc])
    end.

lines_to_ints(Lines) ->
    Clean = [string:trim(L) || L <- Lines],
    IntStrs = [L || L <- Clean, L =/= ""],
    [ list_to_integer(S) || S <- IntStrs ].

print_ets(T, N) ->
    lists:foreach(
      fun(I) ->
          {_, V} = ets:lookup_element(T, I, 2) orelse erlang:error({missing_index, I}),
          io:format("~p~n", [V])
      end,
      lists:seq(1, N)).

%% ---------- ETS array ops ----------

load_ets(T, Ints) ->
    _ = lists:foldl(
          fun(V, {Idx, Acc}) ->
                  ets:insert(T, {Idx, V}),
                  {Idx+1, Acc}
          end, {1, ok}, Ints),
    ok.

get(T, I) ->
    case ets:lookup(T, I) of
        [{_, V}] -> V;
        [] -> erlang:error({bad_index, I})
    end.

set(T, I, V) ->
    ets:insert(T, {I, V}).

swap(T, I, J) when I =:= J -> ok;
swap(T, I, J) ->
    Vi = get(T, I),
    Vj = get(T, J),
    set(T, I, Vj),
    set(T, J, Vi).

median3(A, B, C) ->
    case true of
        _ when (A =< B andalso B =< C) orelse (C =< B andalso B =< A) -> B;
        _ when (B =< A andalso A =< C) orelse (C =< A andalso A =< B) -> A;
        _ -> C
    end.

%% Hoare partition on ETS-backed array [Lo..Hi], returns J boundary index.
hoare_partition(T, Lo, Hi) ->
    Mid   = Lo + (Hi - Lo) div 2,
    Pivot = median3(get(T, Lo), get(T, Mid), get(T, Hi)),
    I0 = Lo - 1,
    J0 = Hi + 1,
    hoare_step(T, Pivot, I0, J0).

hoare_step(T, Pivot, I, J) ->
    I1 = next_i(T, Pivot, I+1),
    J1 = next_j(T, Pivot, J-1),
    if
        I1 >= J1 -> J1;
        true ->
            swap(T, I1, J1),
            hoare_step(T, Pivot, I1, J1)
    end.

next_i(T, Pivot, I) ->
    case get(T, I) < Pivot of
        true  -> next_i(T, Pivot, I+1);
        false -> I
    end.

next_j(T, Pivot, J) ->
    case get(T, J) > Pivot of
        true  -> next_j(T, Pivot, J-1);
        false -> J
    end.

%% Iterative quicksort using explicit stack of {Lo,Hi}, sorting ETS “in place”.
quicksort_ets(_T, _Lo, N) when N =< 1 -> ok;
quicksort_ets(T, Lo, Hi) ->
    quicksort_loop(T, [{Lo, Hi}]).

quicksort_loop(_T, []) -> ok;
quicksort_loop(T, [{Lo, Hi} | Rest]) when Lo < Hi ->
    P = hoare_partition(T, Lo, Hi),
    LLo = Lo,  LHi = P,
    RLo = P+1, RHi = Hi,
    LLen = LHi - LLo + 1,
    RLen = RHi - RLo + 1,
    %% push smaller segment first to keep stack shallow
    NewStack =
        case LLen =< RLen of
            true  ->
                S1 = (LLen > 1) andalso [{LLo, LHi}] orelse [],
                S2 = (RLen > 1) andalso [{RLo, RHi}] orelse [],
                S1 ++ S2 ++ Rest;
            false ->
                S1 = (RLen > 1) andalso [{RLo, RHi}] orelse [],
                S2 = (LLen > 1) andalso [{LLo, LHi}] orelse [],
                S1 ++ S2 ++ Rest
        end,
    quicksort_loop(T, NewStack);
quicksort_loop(T, [_ | Rest]) ->
    quicksort_loop(T, Rest).

