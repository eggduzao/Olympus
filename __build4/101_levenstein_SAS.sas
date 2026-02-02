/* Levenshtein Distance implementation in SAS using a DATA step */
data levenshtein;
    length s1 s2 $50;
    s1 = "kitten";
    s2 = "sitting";

    l1 = length(s1);
    l2 = length(s2);

    /* Create an array to hold distances */
    array d[0:50, 0:50] _temporary_;

    /* Initialize first row and column */
    do i = 0 to l1;
        d[i,0] = i;
    end;
    do j = 0 to l2;
        d[0,j] = j;
    end;

    /* Fill the DP matrix */
    do i = 1 to l1;
        do j = 1 to l2;
            if substr(s1,i,1) = substr(s2,j,1) then cost = 0;
            else cost = 1;

            del = d[i-1,j] + 1;
            ins = d[i,j-1] + 1;
            sub = d[i-1,j-1] + cost;

            d[i,j] = min(del, ins, sub);
        end;
    end;

    distance = d[l1,l2];
    put "Levenshtein distance between '" s1 "' and '" s2 "' is " distance;
run;

