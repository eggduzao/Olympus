%% binsearch.erl — Binary search (lower-bound) on a sorted ascending list.
%% Usage (as escript-style entrypoint):
%%   escript binsearch.erl -- 11 data.txt
%%   printf "1\n3\n4\n7\n9\n11\n15\n" | escript binsearch.erl -- 11
%%
%% Behavior:
%%   • Reads newline-separated integers (skips blank/non-integer lines with a warning).
%%   • Verifies ascending (non-decreasing) order.
%%   • Performs lower-bound binary search:
%%       - If target exists:  "FOUND <target> at index <1-based>"
%%       - Else:              "NOT FOUND <target>. Insertion index <1-based>, between <L> and <R>"

-module(binsearch).
-export([main/1]).

%% ======= Entry =======
main(Args) ->
    {TargetStr, PathOpt} = parse_after_dashdash(Args),
    case to_int(TargetStr) of
        error ->
            eprintf("ERROR: target must be an integer, got '~s'~n", [TargetStr]),
            usage(), halt(1);
        {ok, Target} ->
            case PathOpt of
                [] ->
                    Ints = read_ints_stdin(),
                    run(Ints, Target);
                [Path] ->
                    case read_ints_file(Path) of
                        {error, Reason} ->
                            eprintf("ERROR: cannot read '~s': ~p~n", [Path, Reason]),
                            halt(2);
                        {ok, Ints} ->
                            run(Ints, Target)
                    end;
                _ ->
                    eprintf("ERROR: too many arguments after target.~n", []),
                    usage(), halt(1)
            end
    end.

usage() ->
    eprintf("Usage: escript binsearch.erl -- <target-int> [path/to/file]\n", []),
    eprintf("       printf \"1\\n3\\n...\" | escript binsearch.erl -- 11\n", []).

%% ======= Orchestration =======
run([], _Target) ->
    eprintf("ERROR: no numeric input provided.~n", []),
    halt(3);
run(Ints, Target) ->
    ensure_ascending(Ints),
    Tup = list_to_tuple(Ints),
    N = size(Tup),
    Ins = lower_bound(Tup, Target, 1, N + 1),  %% Erlang tuples 1-based; use half-open [lo,hi)
    case (Ins =< N) andalso (element(Ins, Tup) =:= Target) of
        true ->
            io:format("FOUND ~p at index ~p~n", [Target, Ins]);
        false ->
            Left  = if Ins > 1 -> integer_to_list(element(Ins-1, Tup)); true -> "-inf" end,
            Right = if Ins =< N -> integer_to_list(element(Ins, Tup));  true -> "+inf" end,
            io:format("NOT FOUND ~p. Insertion index ~p (1-based), between ~s and ~s~n",
                      [Target, Ins, Left, Right])
    end,
    ok.

%% ======= Parsing args after literal "--" =======
parse_after_dashdash(Args) ->
    case drop_until_dashdash(Args) of
        [] -> {"", []};
        [TargetStr | Rest] -> {TargetStr, Rest}
    end.

drop_until_dashdash([]) -> [];
drop_until_dashdash(["--" | Tail]) -> Tail;
drop_until_dashdash([_ | Tail]) -> drop_until_dashdash(Tail).

%% ======= IO & parsing =======
read_ints_stdin() ->
    read_ints_from_fun(fun() -> io:get_line('') end, 1, []).

read_ints_file(Path) ->
    case file:open(Path, [read]) of
        {ok, IoDev} ->
            F = fun() -> io:get_line(IoDev, '') end,
            Result = read_ints_from_fun(F, 1, []),
            file:close(IoDev),
            {ok, Result};
        Error -> Error
    end.

read_ints_from_fun(GetLineFun, Ln, Acc) ->
    case GetLineFun() of
        eof ->
            lists:reverse(Acc);
        Line ->
            T = trim(Line),
            case T of
                "" ->
                    read_ints_from_fun(GetLineFun, Ln+1, Acc);
                _ ->
                    case to_int(T) of
                        {ok, V} -> read_ints_from_fun(GetLineFun, Ln+1, [V|Acc]);
                        error ->
                            eprintf("WARN: skipping non-integer line ~p: ~s", [Ln, Line]),
                            read_ints_from_fun(GetLineFun, Ln+1, Acc)
                    end
            end
    end.

to_int(Str) ->
    %% Accept decimal integers (optional leading +/-)
    case string:to_integer(trim(Str)) of
        {Int, ""} -> {ok, Int};
        _ -> error
    end.

trim(S) ->
    string:trim(S, both, "\s\t\r\n").

eprintf(Fmt, Args) ->
    io:format(standard_error, Fmt, Args).

%% ======= Checks =======
ensure_ascending([]) -> ok;
ensure_ascending([_]) -> ok;
ensure_ascending([A,B|Rest]) ->
    case B < A of
        true ->
            eprintf("ERROR: input not in ascending order near '~p' then '~p'~n", [A, B]),
            halt(4);
        false ->
            ensure_ascending([B|Rest])
    end.

%% ======= Binary search (lower_bound) on a tuple =======
%% lower_bound(T, X, Lo, Hi):
%%   T is tuple (1..N), Lo/Hi define half-open search interval [Lo,Hi)
%%   Returns first index i s.t. T[i] >= X, or N+1 if none.
lower_bound(T, X, Lo, Hi) when Lo < Hi ->
    Mid = (Lo + Hi) div 2,
    case element_in_range(T, Mid) of
        {ok, V} when V < X ->
            lower_bound(T, X, Mid+1, Hi);
        {ok, _V} ->
            lower_bound(T, X, Lo, Mid);
        none -> Hi  %% shouldn't happen if bounds are valid
    end;
lower_bound(_T, _X, Lo, _Hi) ->
    Lo.

element_in_range(T, I) ->
    N = size(T),
    if I >= 1, I =< N -> {ok, element(I, T)};
       true           -> none
    end.
