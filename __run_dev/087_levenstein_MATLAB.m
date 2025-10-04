% Levenshtein edit distance (MATLAB script)
% Reads two lines (strings) from stdin and prints the distance

s1 = strtrim(input('', 's'));
s2 = strtrim(input('', 's'));

m = length(s1);
n = length(s2);

% DP matrix (m+1) x (n+1)
dp = zeros(m+1, n+1, 'uint32');
dp(:,1) = uint32(0:m);
dp(1,:) = uint32(0:n);

for i = 1:m
    for j = 1:n
        cost = uint32(s1(i) ~= s2(j));
        deletion    = dp(i,   j+1) + 1;
        insertion   = dp(i+1, j  ) + 1;
        substitution= dp(i,   j  ) + cost;
        dp(i+1, j+1) = min([deletion, insertion, substitution]);
    end
end

disp(dp(m+1, n+1))

