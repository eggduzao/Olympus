/* in_place_quicksort.sas — SAS — In-place quicksort on an integer “array”
 *
 * What “in-place” means here:
 *   We load the input numbers into a temporary ARRAY in a DATA step and sort
 *   them there by swapping elements in that array (no PROC SORT used).
 *
 * How to run
 * ----------
 *   Option A) Use the embedded sample (remove or replace the DATALINES as needed):
 *       sas in_place_quicksort.sas -nolog
 *
 *   Option B) Read from a text file (one integer per line):
 *       %let INPUT_PATH=/absolute/path/to/ints.txt;
 *       sas in_place_quicksort.sas -nolog
 *
 * Output
 * ------
 *   Prints the sorted numbers (one per row) and creates a WORK.SORTED dataset
 *   with a single numeric column VALUE.
 */

/* ===== 1) Load input into WORK.NUMS ====================================== */

%macro load_input;
%if %symexist(INPUT_PATH) and %length(&INPUT_PATH) %then %do;
data nums;
  infile "&INPUT_PATH" truncover;
  input value : best32.;
  if not missing(value);
run;
%end;
%else %do;
/* Default sample input; replace/remove as desired */
data nums;
  infile datalines truncover;
  input value : best32.;
  if not missing(value);
datalines;
5
3
8
1
2
;
run;
%end;
%mend;

%load_input;

/* Count rows to size the temporary arrays at compile time */
proc sql noprint;
  select count(*) into :N trimmed from nums;
quit;

%if &N = 0 %then %do;
  %put NOTE: No numbers found. Nothing to sort.;
  %goto done;
%end;

/* ===== 2) DATA step: iterative quicksort with Hoare partition ============ */

data sorted;
  length value 8;

  /* Temporary “in-place” array holding the numbers */
  array A[&N] _temporary_;

  /* Load A from dataset NUMS (point= for direct addressing) */
  do i = 1 to &N;
    set nums point=i nobs=nobs;
    A[i] = value;
  end;

  /* ---- Quicksort stacks (subranges) ---- */
  array L[&N] _temporary_;   /* low indexes */
  array H[&N] _temporary_;   /* high indexes */
  top = 0;

  /* push initial full range [1..N] */
  top + 1; L[top] = 1; H[top] = &N;

  /* -------- Utility routines as inlined code blocks -------- */

  /* median-of-three for pivot selection */
  length pivot 8;
  /* (We will compute pivot each partition using MEDIAN function) */

  /* Iterative quicksort loop */
  do while (top > 0);
    lo = L[top]; hi = H[top]; top + (-1);

    if lo < hi then do;
      /* ---- Hoare partition on A[lo..hi] ---- */
      mid   = lo + floor((hi - lo) / 2);
      pivot = median(A[lo], A[mid], A[hi]);

      i = lo - 1;
      j = hi + 1;
      do _ forever;
        /* advance i while A[i] < pivot */
        do i + 1 while (A[i] < pivot);
        end;
        /* retreat j while A[j] > pivot */
        do j - 1 while (A[j] > pivot);
        end;

        if i >= j then leave;

        /* swap A[i] <-> A[j] */
        temp = A[i]; A[i] = A[j]; A[j] = temp;
      end;
      p = j;                   /* partition boundary */

      /* Subranges: [lo..p] and [p+1..hi] */
      left_lo  = lo;  left_hi  = p;
      right_lo = p+1; right_hi = hi;

      left_len  = left_hi  - left_lo  + 1;
      right_len = right_hi - right_lo + 1;

      /* Push smaller range last → process it next (keeps stack shallow) */
      if left_len > 1 and right_len > 1 then do;
        if left_len <= right_len then do;
          top + 1; L[top] = right_lo; H[top] = right_hi;
          top + 1; L[top] = left_lo;  H[top] = left_hi;
        end;
        else do;
          top + 1; L[top] = left_lo;  H[top] = left_hi;
          top + 1; L[top] = right_lo; H[top] = right_hi;
        end;
      end;
      else if left_len > 1 then do;  top + 1; L[top] = left_lo;  H[top] = left_hi;  end;
      else if right_len > 1 then do; top + 1; L[top] = right_lo; H[top] = right_hi; end;
    end;
  end;

  /* Emit sorted values into WORK.SORTED */
  do k = 1 to &N;
    value = A[k];
    output;
  end;

  stop;
run;

/* ===== 3) Print results =================================================== */
title "Sorted integers (ascending)";
proc print data=sorted noobs; var value; run;
title;

%done:

