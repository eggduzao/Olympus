/*==============================================================================
  BFS Shortest Path on an Unweighted Graph (SAS, single file)
  ------------------------------------------------------------------------------
  INPUT format (whitespace- or tab-separated), one line per record:
    • Edge line  : "U  V1 [V2 ...]"    -> add UNDIRECTED edges U—V1, U—V2, ...
    • Node line  : "U"                 -> ensure node exists (even if isolated)
    • Query line : "#  SRC DST"        -> request shortest path from SRC to DST

  Example:
      A   B   F
      B   A   C
      C   B   D
      D   C   E
      E   D   F
      F   A   E
      #   A   E

  RUN:
    Option A) Use the demo datalines below: just SUBMIT.
    Option B) Comment the demo block and set:
        %let PATH=/full/path/to/graph.tsv;
      then run; it will read that file.

  OUTPUT:
    For each "# SRC DST" query, prints either "A -> B -> ... -> E"
    or "No path found from SRC to DST".

  Notes:
    • Uses DATA step hash objects with MULTIDATA to store adjacency lists.
    • BFS queue implemented as a hash keyed by an integer index.
    • Handles large graphs in a single pass; memory use ~ O(|V| + |E|).
==============================================================================*/

options nonotes nodate nosource;

/*--- Configure external input (optional) ---*/
%let PATH=;   /* leave empty to use demo datalines; set to a file path to read it */

/*--- Read input into WORK.RAW with tokens token1-tokenN ----------------------*/
%macro read_input;
  %if %length(&PATH) %then %do;
    data raw;
      length line $32767 token1-token32 $128; /* up to 32 tokens per line */
      infile "&PATH" lrecl=32767 truncover;
      input line $varying32767. _len;
      /* Split by any whitespace */
      n = 0;
      do i = 1 by 1 while(scan(line, i, '09'x'0A'x'0D'x'20'x) ne '');
        n+1;
        if n<=32 then call missing(token{i});
        if n<=32 then token{i} = scan(line, i, '09'x'0A'x'0D'x'20'x);
      end;
      drop i _len;
    run;
  %end;
  %else %do;
    data raw;
      length line $32767 token1-token32 $128;
      infile datalines lrecl=32767 truncover;
      input line $varying32767. _len;
      n = 0;
      do i = 1 by 1 while(scan(line, i, '09'x'0A'x'0D'x'20'x) ne '');
        n+1;
        if n<=32 then call missing(token{i});
        if n<=32 then token{i} = scan(line, i, '09'x'0A'x'0D'x'20'x);
      end;
      drop i _len;
      datalines;
A	B	F
B	A	C
C	B	D
D	C	E
E	D	F
F	A	E
#	A	E
;
    run;
  %end;
%mend read_input;

%read_input

/*--- Build EDGE list (undirected) and QUERY list -----------------------------*/
proc datasets lib=work nolist; delete edges queries nodes; quit;

data edges(keep=u v) queries(keep=src dst) nodes(keep=node);
  set raw;
  array t[32] $128 token1-token32;
  if n = . then n = 0;

  if n = 0 then delete;

  if t[1] = '#' then do;
    if n >= 3 then do; src = t[2]; dst = t[3]; output queries; end;
    return;
  end;

  /* Node existence even if isolated */
  node = t[1]; output nodes;

  /* For tokens t[2..n], emit undirected edges */
  length u v $128;
  u = t[1];
  do i = 2 to n while(t[i] ne '');
    v = t[i];
    output edges;             /* u -> v */
    /* also emit v node in case it appears only as neighbor */
    node = v; output nodes;
  end;
  drop i;
run;

/* Deduplicate nodes */
proc sort data=nodes nodupkey; by node; run;
/* Normalize/clean the edges (drop self-loops, dedup) */
data edges;
  set edges;
  if missing(u) or missing(v) then delete;
  if u = v then delete; /* ignore self-loops */
run;
proc sort data=edges nodupkey; by u v; run;

/*--- BFS for each query ------------------------------------------------------*/
data _null_;
  length u v src dst cur nbr node parent path $128;
  length found 8;
  /* Load queries */
  if 0 then set queries;

  /* Build adjacency hash: key=u, data=v; allow multiple v per u */
  declare hash adj(dataset:"edges", multidata:"y");
    adj.defineKey  ("u");
    adj.defineData ("u","v");
    adj.defineDone();

  /* Create an iterator to step through neighbors with same key */
  declare hiter adjit("adj");

  /* We'll iterate over queries with SET */
  do _q_ = 1 by 1 until (endq);
    set queries end=endq;

    /* Quick validations */
    found = 0;
    path  = "";

    /* Ensure SRC and DST exist as nodes */
    declare hash allnodes(dataset:"nodes");
      allnodes.defineKey("node");
      allnodes.defineDone();

    if allnodes.find(key:src) ne 0 or allnodes.find(key:dst) ne 0 then do;
      put "No path found from " src " to " dst;
      allnodes.delete();
      continue;
    end;
    allnodes.delete();

    /* Trivial case: src == dst */
    if src = dst then do;
      put src;
      continue;
    end;

    /* Visited set: key=node */
    declare hash visited();
      visited.defineKey("node");
      visited.defineDone();

    /* Parent map: key=child, data=parent */
    declare hash parentmap();
      parentmap.defineKey("node");
      parentmap.defineData("parent");
      parentmap.defineDone();

    /* FIFO queue implemented as hash: key=idx (numeric), data=node */
    length idx 8;
    declare hash q(ordered:"a");
      q.defineKey("idx");
      q.defineData("node");
      q.defineDone();

    /* enqueue src */
    idx = 1; node = src; rc = q.add();
    head = 1; tail = 1;
    node = src; rc = visited.add();

    /* ---------------- BFS loop ---------------- */
    do while (head <= tail and not found);
      /* dequeue */
      rc = q.find(key:head);
      cur = node;
      head + 1;

      /* For neighbors of cur: set key and iterate through all matching rows */
      u   = cur;
      rcF = adj.find(key:u);
      if rcF = 0 then do;
        /* We found at least one neighbor row; iter across all same-key rows */
        rcI = adjit.first();
        do while (rcI = 0);
          if u ne cur then leave; /* once iterator moves past current key block */
          nbr = v;

          /* If unseen, record and enqueue */
          node = nbr;
          if visited.find() ne 0 then do;
            /* mark visited */
            rc = visited.add();
            /* set parent */
            parent = cur;
            rc = parentmap.replace(key:node, data:parent);

            /* found destination? */
            if nbr = dst then do;
              found = 1;
              leave;
            end;

            /* enqueue nbr */
            tail + 1; idx = tail; node = nbr; rc = q.add();
          end;

          rcI = adjit.next();
        end;
      end;
    end;

    /* -------------- Reconstruct path or report none -------------- */
    if found then do;
      /* Build "src -> ... -> dst" by backtracking from DST */
      path = "";
      node = dst;
      do while (1);
        /* Prepend current node */
        if missing(path) then path = node;
        else path = catx(" -> ", node, path);

        /* Move to parent; if none, break */
        if parentmap.find(key:node) ne 0 then leave;
        if parent = "" then leave;
        node = parent;
      end;

      /* Ensure the path starts at SRC */
      if scan(path,1,'->') ne src then do;
        put "No path found from " src " to " dst;
      end;
      else put path;
    end;
    else do;
      put "No path found from " src " to " dst;
    end;

    /* Cleanup per-query transient hashes */
    visited.delete(); parentmap.delete(); q.delete();
  end;
run;

