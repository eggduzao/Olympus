% bfs_unweighted.m
% Breadth-first search (BFS) shortest path on an unweighted, undirected graph.
% INPUT FORMAT (tab- or space-separated; one edge per line; queries start with '#'):
%   A   B   F
%   B   A   C
%   ...
%   #   SRC DST
%
% Run examples (in MATLAB/Octave Command Window):
%   % From a file:
%   run_bfs_unweighted('graph.tsv');
%   % From a string (e.g., copy-pasted sample):
%   txt = sprintf('A\tB\tF\nB\tA\tC\nC\tB\tD\nD\tC\tE\nE\tD\tF\nF\tA\tE\n#\tA\tE\n');
%   run_bfs_unweighted(txt);
%
% Outputs one line per query as: "SRC -> ... -> DST" or an explanatory message.

function run_bfs_unweighted(inputSource)
  if nargin < 1
    error('Usage: run_bfs_unweighted(inputSource) where inputSource is a file path OR a char/string with the contents.');
  end

  %---- Load lines ----------------------------------------------------------
  if ischar(inputSource) || (isstring(inputSource) && isscalar(inputSource))
    if isfile(inputSource)  % treat as path
      raw = fileread(inputSource);
    else                    % treat as literal contents
      raw = char(inputSource);
    end
  else
    error('inputSource must be a file path or a string with the file contents.');
  end
  raw = regexprep(raw, '\r\n?', '\n');             % normalize newlines
  L = regexp(raw, '\n', 'split');                  % cellstr lines

  %---- Parse into adjacency and queries -----------------------------------
  adj = containers.Map('KeyType','char','ValueType','any');  % node -> cellstr neighbors
  queries = {};  % N x 2 cell array of {src, dst}

  for i = 1:numel(L)
    line = strtrim(L{i});
    if isempty(line)
      continue
    end
    % split by any run of tabs/spaces
    toks = regexp(line, '[\t ]+', 'split');
    if isempty(toks)
      continue
    end

    if startsWith(toks{1}, '#')
      if numel(toks) < 3
        warning('Skipping malformed query line %d: "%s"', i, line);
        continue
      end
      src = toks{2}; dst = toks{3};
      queries(end+1,1:2) = {src, dst}; %#ok<AGROW>
    else
      % edge row: NODE NEI1 [NEI2]
      if numel(toks) < 2
        warning('Skipping malformed edge line %d: "%s"', i, line);
        continue
      end
      u = toks{1};
      v1 = toks{2};
      if ~isempty(v1)
        adj = add_undirected_edge(adj, u, v1);
      end
      if numel(toks) >= 3
        v2 = toks{3};
        if ~isempty(v2)
          adj = add_undirected_edge(adj, u, v2);
        end
      end
    end
  end

  if isempty(queries)
    fprintf('No queries found (lines starting with "#\\tSRC\\tDST"). Nothing to do.\n');
    return
  end

  %---- Answer queries with BFS --------------------------------------------
  for q = 1:size(queries,1)
    src = queries{q,1};
    dst = queries{q,2};

    if ~isKey(adj, src)
      fprintf('No path found from %s to %s (source not in graph)\n', src, dst);
      continue
    end
    if ~strcmp(src, dst) && ~isKey(adj, dst)
      fprintf('No path found from %s to %s (destination not in graph)\n', src, dst);
      continue
    end
    if strcmp(src, dst)
      fprintf('%s\n', src);
      continue
    end

    path = bfs_path(adj, src, dst);
    if isempty(path)
      fprintf('No path found from %s to %s\n', src, dst);
    else
      fprintf('%s\n', strjoin(path, ' -> '));
    end
  end
end

%==================== Helpers ====================%

function adj = add_undirected_edge(adj, u, v)
  % Add u<->v (unique neighbors)
  adj = add_directed_unique(adj, u, v);
  adj = add_directed_unique(adj, v, u);
end

function adj = add_directed_unique(adj, u, v)
  if ~isKey(adj, u)
    adj(u) = {v};
    return
  end
  lst = adj(u);
  % ensure uniqueness; treat as strings
  if ~any(strcmp(lst, v))
    lst{end+1} = v; %#ok<AGROW>
    adj(u) = lst;
  end
end

function path = bfs_path(adj, src, dst)
  % Classic BFS to reconstruct shortest unweighted path.
  % Uses cell arrays as queue for O(1) amortized via head index.
  visited = containers.Map('KeyType','char','ValueType','logical');
  parent  = containers.Map('KeyType','char','ValueType','char');

  queue = {src};
  head = 1;
  visited(src) = true;
  parent(src) = '';  % sentinel

  found = false;
  while head <= numel(queue)
    u = queue{head}; head = head + 1;

    if strcmp(u, dst)
      found = true;
      break
    end

    if isKey(adj, u)
      nei = adj(u);
      for k = 1:numel(nei)
        v = nei{k};
        if ~isKey(visited, v)
          visited(v) = true;
          parent(v)  = u;
          queue{end+1} = v; %#ok<AGROW>
        end
      end
    end
  end

  if ~found
    path = {};
    return
  end

  % Reconstruct from dst -> src
  rev = {dst};
  cur = dst;
  while true
    p = parent(cur);
    if isempty(p), break; end
    rev{end+1} = p; %#ok<AGROW>
    cur = p;
  end
  path = rev(end:-1:1);
end

