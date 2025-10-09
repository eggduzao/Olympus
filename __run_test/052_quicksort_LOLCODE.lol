HAI 1.3
CAN HAS STDIO?

BTW
BTW  In-place quicksort on an integer list read from STDIN (one number per line).
BTW  Usage:
BTW    $ lci quicksort.lol < input.txt
BTW
BTW  Notes:
BTW   - Input stops at EOF (Ctrl+D) or a blank line.
BTW   - We store numbers in a BUKKIT (hash) with numeric keys 0..N-1 and sort in place.
BTW

I HAS A A ITZ A BUKKIT       BTW array-like storage
I HAS A N ITZ 0              BTW length

HOW IZ I PUSH YR V
  A SRS N R V
  N R SUM OF N AN 1
IF U SAY SO

BTW Read integers until blank line or EOF
IM IN YR READIN
  I HAS A LINE
  GIMMEH LINE
  BOTH SAEM LINE AN NOOB, O RLY?
    YA RLY
      GTFO
    NO WAI
      BOTH SAEM LINE AN "", O RLY?
        YA RLY
          GTFO
        NO WAI
          I HAS A NUM
          NUM R MAEK LINE A NUMBAR
          I IZ PUSH YR NUM MKAY
      OIC
  OIC
IM OUTTA YR READIN

BTW ----------------------------
BTW Swap helper: swap A[IDX1], A[IDX2]
BTW ----------------------------
HOW IZ I SWAP YR IDX1 AN YR IDX2
  I HAS A TMP ITZ A SRS IDX1
  A SRS IDX1 R A SRS IDX2
  A SRS IDX2 R TMP
IF U SAY SO

BTW ------------------------------------------------------
BTW Lomuto partition: partitions A[LO..HI], returns pivot index
BTW ------------------------------------------------------
HOW IZ I PARTISHUN YR LO AN YR HI
  I HAS A PIV ITZ A SRS HI
  I HAS A STORE ITZ LO
  I HAS A ITR ITZ LO

  IM IN YR LOOP WILE BOTH SAEM SMALLR OF ITR AN HI AN ITR  BTW while ITR < HI
    BTW if A[ITR] <= PIV
    BOTH SAEM SMALLR OF A SRS ITR AN PIV AN A SRS ITR, O RLY?
      YA RLY
        I IZ SWAP YR STORE AN YR ITR MKAY
        STORE R SUM OF STORE AN 1
    OIC
    ITR R SUM OF ITR AN 1
  IM OUTTA YR LOOP

  I IZ SWAP YR STORE AN YR HI MKAY
  FOUND YR STORE
IF U SAY SO

BTW ----------------------------------------
BTW Recursive quicksort over range [LO..HI]
BTW ----------------------------------------
HOW IZ I QUICKSORT YR LO AN YR HI
  BTW if LO < HI
  BOTH SAEM SMALLR OF LO AN HI AN LO, O RLY?
    YA RLY
      DIFFRINT LO AN HI, O RLY?
        YA RLY
          I HAS A PIVIX ITZ I IZ PARTISHUN YR LO AN YR HI MKAY
          I IZ QUICKSORT YR LO AN YR DIFF OF PIVIX AN 1 MKAY
          I IZ QUICKSORT YR SUM OF PIVIX AN 1 AN YR HI MKAY
      OIC
  OIC
IF U SAY SO

BTW Kick off sort if we read anything
BOTH SAEM N AN 0, O RLY?
  YA RLY
    BTW nothing to sort
  NO WAI
    I IZ QUICKSORT YR 0 AN YR DIFF OF N AN 1 MKAY
OIC

BTW Output one number per line
I HAS A I ITZ 0
IM IN YR PRINTIN WILE BOTH SAEM SMALLR OF I AN N AN I
  VISIBLE A SRS I
  I R SUM OF I AN 1
IM OUTTA YR PRINTIN

KTHXBYE

