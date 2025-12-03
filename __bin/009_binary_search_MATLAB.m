
function binary_search_cli(target, filepath)
% binary_search_cli — Binary search over a sorted list of integers.
%
% USAGE (pick one):
%   % A) From MATLAB/Octave command line, using a file with one integer per line:
%   binary_search_cli(11, 'data.txt')
%
%   % B) From MATLAB/Octave, passing a numeric vector directly (no file):
%   binary_search_cli(11, [1 3 4 7 9 11 15])
%
%   % C) From system shell (if MATLAB/Octave is on PATH), non-interactive:
%   %   matlab -batch "binary_search_cli(11, 'data.txt')"
%   %   octave --quiet --eval "binary_search_cli(11, 'data.txt')"
%
% BEHAVIOR:
%   • Performs classic binary search (O(log N)).
%   • On success, prints: FOUND <target> at index <i> (1-based)
%   • On miss, prints:   NOT FOUND <target>. Insertion index <i> (1-based) ...
%   • If duplicates exist, reports the FIRST occurrence.
%
% NOTES:
%   • Input must be in NON-DECREASING (ascending) order.
%   • Non-integer values are allowed but will be rounded toward nearest integer
%     for display; the comparison itself is numeric (double).
%
% RETURNS:
%   This is a CLI-style function and does not return a value. For programmatic
%   use, see the local function [idx,found,ins] = binary_search(vec, target, firstOccurrence).

  % -------- Parse inputs / load vector --------
  if nargin < 2
    error('Usage: binary_search_cli(TARGET, FILEPATH or VECTOR)');
  end

  vec = [];
  if isnumeric(filepath)
    vec = double(filepath(:).');         % row vector
  elseif ischar(filepath) || isstring(filepath)
    fp = char(filepath);
    if ~isfile(fp)
      error('File not found: %s', fp);
    end
    % Robust file read: one number per line (ignores blanks, skips text)
    vec = read_numeric_column(fp);
  else
    error('Second argument must be a filename (char/string) or a numeric vector.');
  end

  if isempty(vec)
    error('No numeric data to search.');
  end

  % -------- Validate monotonicity (ascending) --------
  if any(diff(vec) < 0)
    error('Input data is not in ascending order (required for binary search).');
  end

  tgt = double(target);

  % -------- Perform binary search for FIRST occurrence --------
  [idx, found, ins] = binary_search(vec, tgt, true);

  % -------- Report --------
  if found
    fprintf('FOUND %g at index %d (1-based)\n', tgt, idx);
  else
    left  = '-inf';
    right = '+inf';
    if ins > 1,      left  = num2str(vec(ins-1)); end
    if ins <= numel(vec), right = num2str(vec(ins));   end
    fprintf('NOT FOUND %g. Insertion index %d (1-based), between %s and %s\n', tgt, ins, left, right);
  end
end


% === Local, programmatic API ===
function [idx, found, insertPos] = binary_search(vec, target, firstOccurrence)
% [idx, found, insertPos] = binary_search(vec, target, firstOccurrence)
%   vec             : ascending numeric vector
%   target          : numeric scalar
%   firstOccurrence : if true, return first index among duplicates
%
% RETURNS
%   idx       : index of first matching element, if found; undefined if not
%   found     : logical scalar
%   insertPos : insertion index (1-based) that would keep vec sorted (always set)
%
% Complexity: O(log N)

  n  = numel(vec);
  lo = 1;
  hi = n;
  idx = -1;

  while lo <= hi
    mid = floor((lo + hi) / 2);
    val = vec(mid);
    if val == target
      idx = mid;
      if firstOccurrence
        hi = mid - 1;   % shrink left to find first
      else
        break;
      end
    elseif val < target
      lo = mid + 1;
    else
      hi = mid - 1;
    end
  end

  found = (idx ~= -1);
  if found && firstOccurrence
    % idx already first by construction
  elseif ~found
    % lo is the insertion point (1..n+1)
    idx = NaN;
  end
  insertPos = lo;
end


function vec = read_numeric_column(fp)
% Read a single numeric column from text file `fp`. Skips blank lines and
% warns on garbage lines. Returns a row vector of doubles.

  fid = fopen(fp, 'r');
  if fid == -1, error('Could not open file: %s', fp); end
  c = onCleanup(@() fclose(fid));

  data = []; dataIdx = 0;
  lineNo = 0;
  while true
    t = fgetl(fid);
    if ~ischar(t), break; end
    lineNo = lineNo + 1;
    s = strtrim(t);
    if isempty(s), continue; end
    v = str2double(s);
    if ~isnan(v)
      dataIdx = dataIdx + 1;
      data(dataIdx,1) = v; %#ok<AGROW>
    else
      warning('Skipping non-numeric line %d: %s', lineNo, t);
    end
  end

  vec = double(data(:).');  % row
end

