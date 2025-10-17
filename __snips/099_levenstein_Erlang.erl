%% Levenshtein edit distance in Erlang
%% Reads two lines from stdin and prints the distance

-module(levenshtein).
-export([main/0, distance/2]).

distance(S1, S2) ->
    L1 = length(S1),
    L2 = length(S2),
    %% Initialize DP matrix as a map of {I,J} -> Value
    D0 = maps:from_list(
           [{{I,0}, I} || I <- lists:seq(0, L1)] ++
           [{{0,J}, J} || J <- lists:seq(0, L2)]
         ),
    D = fill_matrix(S1, S2, L1, L2, D0),
    maps:get({L1, L2}, D).

fill_matrix(S1, S2, L1, L2, D0) ->
    lists:foldl(
      fun(I, D1) ->
          lists:foldl(
            fun(J, D2) ->
                Cost = if lists:nth(I, S1) =:= lists:nth(J, S2) -> 0;
                          true -> 1
                       end,
                Del = maps:get({I-1,J}, D2) + 1,
                Ins = maps:get({I,J-1}, D2) + 1,
                Sub = maps:get({I-1,J-1}, D2) + Cost,
                Val = lists:min([Del, Ins, Sub]),
                maps:put({I,J}, Val, D2)
            end, D1, lists:seq(1, L2))
      end, D0, lists:seq(1, L1)).

main() ->
    {ok, [S1]} = io:fread("", "~s"),
    {ok, [S2]} = io:fread("", "~s"),
    D = distance(S1, S2),
    io:format("~p~n", [D]).

