/* binsearch.sas — Binary search (lower-bound) over a sorted ascending list.
   Usage (from SAS session):
     %binsearch(target=11, path=/full/path/to/data.txt);

   The file at &path must contain one integer per line. Blank or non-numeric
   lines are skipped with a warning. The macro:
     • Verifies ascending (non-decreasing) order
     • Performs lower-bound binary search
       - If found:   prints "FOUND <target> at index <1-based>"
       - If absent:  prints "NOT FOUND <target>. Insertion index <1-based>, between <L> and <R>"
         (L/R become -inf/+inf at the extremes)
*/

%macro binsearch(target=, path=);

  %local _have_path;
  %let _have_path = %sysevalf(%superq(path) ne, boolean);

  %if %superq(target)= %then %do;
    %put ERROR: Missing required parameter TARGET= (an integer).;
    %return;
  %end;

  /* -------- Read input into WORK._raw_nums -------- */
  data _raw_nums;
    length _line $32767;
    length value 8;
    retain _warned 0;
    %if &_have_path %then %do;
      infile "&path" lrecl=32767 truncover termstr=lf;
    %end;
    %else %do;
      %put NOTE: No PATH= provided. Expecting you to have an external file. ;
      %put NOTE: For a quick demo, copy your numbers into a file and pass PATH=. ;
      %put NOTE: Example file contents:;
      %put NOTE- 1 3 4 7 9 11 15 ;
      stop; /* nothing to read without PATH= */
    %end;

    input _line & $char32767.;
    _line = strip(_line);
    if missing(_line) then delete;

    /* Try numeric parse */
    _code = inputn(_line, 'BEST32.');
    if missing(_code) and notdigit(translate(_line, ' ', '+-')) then do;
      if _warned=0 then do;
        putlog "WARN: encountered non-integer line(s); they will be skipped.";
        _warned=1;
      end;
      delete;
    end;
    else do;
      /* allow integers like '+11' or '009' */
      if prxmatch('/^\s*[+-]?\d+\s*$/', _line) then value = input(_line, 32.);
      else do;
        /* numeric but not a clean integer token: skip */
        if _warned=0 then do;
          putlog "WARN: encountered non-integer line(s); they will be skipped.";
          _warned=1;
        end;
        delete;
      end;
    end;

    keep value;
  run;

  /* Empty? */
  %local _nobs;
  proc sql noprint;
    select count(*) into :_nobs trimmed from _raw_nums;
  quit;

  %if &_nobs = 0 %then %do;
    %put ERROR: No numeric input found in &path..;
    %return;
  %end;

  /* -------- Ensure ascending (non-decreasing) order -------- */
  data _chk;
    set _raw_nums end=last;
    retain _ok 1 _prev . _break_a . _break_b .;
    if not missing(_prev) and value < _prev then do;
      _ok = 0;
      _break_a = _prev; _break_b = value;
    end;
    _prev = value;
    if last then do;
      call symputx('_is_ok', _ok, 'L');
      call symputx('_break_a', _break_a, 'L');
      call symputx('_break_b', _break_b, 'L');
    end;
  run;

  %if &_is_ok = 0 %then %do;
    %put ERROR: Input is not in ascending order near &_break_a then &_break_b.;
    %return;
  %end;

  /* -------- Load into a temporary array and perform lower-bound search -------- */
  %local _tgt;
  %let _tgt = %sysfunc(inputn(&target, best.));
  %if %sysevalf(%superq(_tgt)=, boolean) %then %do;
    %put ERROR: TARGET= must be an integer, got %superq(target).;
    %return;
  %end;

  data _null_;
    /* Prepare an in-memory array sized to the number of observations */
    array a[&_nobs] _temporary_;
    retain i 0;

    /* Populate the array in order */
    do _n_ = 1 by 1 until (endf);
      set _raw_nums end=endf;
      i + 1;
      a[i] = value;
    end;

    length left right 32;
    length msg $200;

    /* Lower-bound binary search on 1-based indices over half-open [lo, hi) */
    lo = 1;
    hi = &_nobs + 1;
    tgt = &target;

    do while (lo < hi);
      mid = floor((lo + hi) / 2);
      v   = a[mid];
      if v < tgt then lo = mid + 1;
      else hi = mid;
    end;

    found = (lo <= &_nobs) and (a[lo] = tgt);

    if found then do;
      msg = cats("FOUND ", tgt, " at index ", put(lo, best.));
      put msg;
    end;
    else do;
      if lo > 1 then left  = strip(put(a[lo-1], best.)); else left  = "-inf";
      if lo <= &_nobs then right = strip(put(a[lo], best.)); else right = "+inf";
      msg = cats("NOT FOUND ", tgt,
                 ". Insertion index ", put(lo, best.),
                 " (1-based), between ", left, " and ", right);
      put msg;
    end;
  run;

%mend binsearch;

/* ------------------------------------------------------------------
   Example invocation (uncomment and adjust PATH= to your file):
   The file should contain one integer per line, ascending order:
     1
     3
     4
     7
     9
     11
     15
------------------------------------------------------------------- */
/*
%binsearch(target=11, path=/absolute/path/to/nums.txt);
%binsearch(target=8,  path=/absolute/path/to/nums.txt);
*/
