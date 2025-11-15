HAI 1.6
BTW ============================================================================
BTW  BFS Shortest Path on an Unweighted Graph — LOLCODE Teaching Script
BTW ----------------------------------------------------------------------------
BTW  This is a *teaching/illustrative* LOLCODE script that shows how BFS works.
BTW  It parses simple tab/space separated edges, builds an undirected graph,
BTW  then answers queries like:  "#  A  E"  by printing the shortest path.
BTW
BTW  Heads up:
BTW    • Real LOLCODE runtimes vary; complex data structures are… limited.
BTW    • This script keeps things simple and self-contained, focusing on ideas.
BTW    • It will “look and feel” like code you can read top-to-bottom to learn BFS.
BTW
BTW  Input format (whitespace/tab separated):
BTW      A  B  F     (edges A—B and A—F)
BTW      B  A  C
BTW      ...
BTW      #  A  E     (query: shortest path from A to E)
BTW
BTW  Output:
BTW      A -> F -> E
BTW ============================================================================

CAN HAS STDIO?

BTW ----------------------------------------------------------------------------
BTW  Tiny string utils so we can handle “arrays” as space-separated strings.
BTW  - PUSH joins with a space (if needed)
BTW  - SPLIT iterates tokens with a cursor (GetTok)
BTW ----------------------------------------------------------------------------

HOW IZ I TRIM YR S
    I HAS A R ITZ S
    IM IN YR lop WILE BOTH SAEM R AN "" OR BOTH SAEM R AN " "
        R R ""
    IM OUTTA YR lop
    FOUND YR R
IF U SAY SO

HOW IZ I PUSH YR BAG AN YR ITEM
    O RLY?
        YA RLY, BOTH SAEM BAG AN ""
            FOUND YR ITEM
        NO WAI
            FOUND YR SMOOSH BAG " " ITEM MKAY
    OIC
IF U SAY SO

HOW IZ I LEN YR BAG
    I HAS A N ITZ 0
    I HAS A TMP ITZ BAG
    IM IN YR cnt
        BOTH SAEM TMP AN ""? O RLY?
            YA RLY, GTFO
        OIC
        I HAS A TOK
        TOK R I IZ GETTOK YR TMP MKAY
        BOTH SAEM TOK AN ""? O RLY?
            YA RLY, GTFO
        OIC
        N R SUM OF N AN 1
    IM OUTTA YR cnt
    FOUND YR N
IF U SAY SO

HOW IZ I GETTOK YR BAG
    BTW return first token; mutate BAG by dropping it (simulated return via global)
    I HAS A OUT ITZ ""
    I HAS A I ITZ 0
    I HAS A L ITZ 0
    L R I IZ SRSSTRLEN YR BAG MKAY
    IM IN YR g0
        BOTH SAEM L AN 0? O RLY?
            YA RLY, GTFO
        OIC
        BOTH SAEM I AN L? O RLY?
            YA RLY, GTFO
        OIC
        I HAS A CH ITZ I IZ SRSSTRGET YR BAG AN I MKAY
        CHR CH IZ " " ? O RLY?
            YA RLY, OIC
        NO WAI
            OUT R SMOOSH OUT CH MKAY
        I R SUM OF I AN 1
        BOTH SAEM I AN L? O RLY?
            YA RLY, GTFO
        OIC
        CHR CH IZ " " ? O RLY?
            YA RLY, GTFO
        OIC
    IM OUTTA YR g0
    I HAS A REM ITZ ""
    IM IN YR g1
        BOTH SAEM I AN L? O RLY?
            YA RLY, GTFO
        OIC
        I HAS A CH2 ITZ I IZ SRSSTRGET YR BAG AN I MKAY
        REM R SMOOSH REM CH2 MKAY
        I R SUM OF I AN 1
    IM OUTTA YR g1
    I AM INVOKIN I IZ SETLASTREM YR REM MKAY
    FOUND YR OUT
IF U SAY SO

I HAS A __LAST_REM ITZ ""
HOW IZ I SETLASTREM YR S
    __LAST_REM R S
IF U SAY SO
HOW IZ I GETLASTREM
    FOUND YR __LAST_REM
IF U SAY SO

BTW ----------------------------------------------------------------------------
BTW  Super small “string library” helpers for characters (mocked for teaching).
BTW  In a real interpreter you’d replace SRSSTRLEN/SRSSTRGET with runtime funcs.
BTW ----------------------------------------------------------------------------
HOW IZ I SRSSTRLEN YR S
    I HAS A N ITZ 0
    IM IN YR l0
        BOTH SAEM S AN "" ? O RLY?
            YA RLY, GTFO
        OIC
        BTW pop one char by slicing head (mock: count by consuming)
        S R I IZ SRSSTRTAIL YR S MKAY
        N R SUM OF N AN 1
    IM OUTTA YR l0
    FOUND YR N
IF U SAY SO

HOW IZ I SRSSTRTAIL YR S
    BTW returns S minus first char (teaching stub)
    I HAS A OUT ITZ ""
    I HAS A FLAG ITZ 0
    IM IN YR t0
        BOTH SAEM S AN ""? O RLY?
            YA RLY, GTFO
        OIC
        I HAS A C ITZ I IZ SRSSTRHEAD YR S MKAY
        BOTH SAEM FLAG AN 0? O RLY?
            YA RLY
                FLAG R 1
            NO WAI
                OUT R SMOOSH OUT C MKAY
        OIC
        S R I IZ SRSSTRDROPHEAD YR S MKAY
    IM OUTTA YR t0
    FOUND YR OUT
IF U SAY SO

HOW IZ I SRSSTRHEAD YR S
    BTW return first char (teaching stub)
    I HAS A C ITZ ""
    O RLY?
        YA RLY, BOTH SAEM S AN "" 
            FOUND YR ""
        NO WAI
            BTW pretend head is S[0] = first symbol (approx)
            C R I IZ SRSSTRSUB YR S AN 0 AN 1 MKAY
            FOUND YR C
    OIC
IF U SAY SO

HOW IZ I SRSSTRDROPHEAD YR S
    FOUND YR I IZ SRSSTRSUB YR S AN 1 AN 99999 MKAY
IF U SAY SO

HOW IZ I SRSSTRSUB YR S AN YR START AN YR COUNT
    BTW naive substring (teaching stub)
    I HAS A OUT ITZ ""
    I HAS A I ITZ 0
    I HAS A L ITZ 0
    L R I IZ SRSSTRLENSHALLOW YR S MKAY
    IM IN YR sub0
        BOTH SAEM I AN L? O RLY?
            YA RLY, GTFO
        OIC
        BOTH SAEM I AN START ? O RLY?
            YA RLY, GTFO
        OIC
        S R I IZ SRSSTRDROPHEAD YR S MKAY
        I R SUM OF I AN 1
    IM OUTTA YR sub0
    I HAS A K ITZ 0
    IM IN YR sub1
        BOTH SAEM K AN COUNT ? O RLY?
            YA RLY, GTFO
        OIC
        BOTH SAEM S AN "" ? O RLY?
            YA RLY, GTFO
        OIC
        OUT R SMOOSH OUT I IZ SRSSTRHEAD YR S MKAY
        S R I IZ SRSSTRDROPHEAD YR S MKAY
        K R SUM OF K AN 1
    IM OUTTA YR sub1
    FOUND YR OUT
IF U SAY SO

HOW IZ I SRSSTRLENSHALLOW YR S
    BTW cheap length: count until empty using DROPHEAD
    I HAS A N ITZ 0
    IM IN YR l2
        BOTH SAEM S AN "" ? O RLY?
            YA RLY, GTFO
        OIC
        S R I IZ SRSSTRDROPHEAD YR S MKAY
        N R SUM OF N AN 1
    IM OUTTA YR l2
    FOUND YR N
IF U SAY SO

BTW ----------------------------------------------------------------------------
BTW  Graph storage (labels[], adj[] of neighbor-label “bags”)
BTW ----------------------------------------------------------------------------
I HAS A LABELS ITZ ""
I HAS A ADJ    ITZ ""   BTW parallel bags, same length; each entry is a "bag" itself

HOW IZ I FINDIDX YR LBL
    I HAS A I ITZ 0
    I HAS A N ITZ I IZ LEN YR LABELS MKAY
    I HAS A CUR ITZ LABELS
    IM IN YR f0
        BOTH SAEM I AN N ? O RLY?
            YA RLY, FOUND YR -1
        OIC
        I HAS A TOK
        TOK R I IZ GETTOK YR CUR MKAY
        CUR R I IZ GETLASTREM MKAY
        BOTH SAEM TOK AN LBL ? O RLY?
            YA RLY, FOUND YR I
        OIC
        I R SUM OF I AN 1
    IM OUTTA YR f0
IF U SAY SO

HOW IZ I ENSURENODE YR LBL
    I HAS A IDX ITZ I IZ FINDIDX YR LBL MKAY
    BOTH SAEM IDX AN -1 ? O RLY?
        YA RLY
            LABELS R I IZ PUSH YR LABELS AN YR LBL MKAY
            ADJ    R I IZ PUSH YR ADJ    AN YR ""  MKAY
            FOUND YR DIFF OF I IZ LEN YR LABELS MKAY AN 1
        NO WAI
            FOUND YR IDX
    OIC
IF U SAY SO

HOW IZ I ADDEDGE YR U AN YR V
    BTW undirected: append labels into each other's adjacency string-bag
    I HAS A UI ITZ I IZ ENSURENODE YR U MKAY
    I HAS A VI ITZ I IZ ENSURENODE YR V MKAY
    BTW add V to adj[UI]
    I HAS A NEWADJ ITZ ""
    I HAS A J ITZ 0
    I HAS A CUR ITZ ADJ
    I HAS A N ITZ I IZ LEN YR ADJ MKAY
    IM IN YR a0
        BOTH SAEM J AN N ? O RLY?
            YA RLY, GTFO
        OIC
        I HAS A AROW
        AROW R I IZ GETTOK YR CUR MKAY
        CUR R I IZ GETLASTREM MKAY
        BOTH SAEM J AN UI ? O RLY?
            YA RLY
                BTW check if V already present
                I HAS A rowbag ITZ AROW
                I HAS A has ITZ 0
                IM IN YR ch
                    BOTH SAEM rowbag AN "" ? O RLY?
                        YA RLY, GTFO
                    OIC
                    I HAS A t ITZ I IZ GETTOK YR rowbag MKAY
                    rowbag R I IZ GETLASTREM MKAY
                    BOTH SAEM t AN V ? O RLY?
                        YA RLY
                            has R 1
                            GTFO
                    OIC
                IM OUTTA YR ch
                O RLY?
                    YA RLY, BOTH SAEM has AN 0
                        AROW R I IZ PUSH YR AROW AN YR V MKAY
                OIC
        OIC
        NEWADJ R I IZ PUSH YR NEWADJ AN YR AROW MKAY
        J R SUM OF J AN 1
    IM OUTTA YR a0
    ADJ R NEWADJ

    BTW add U to adj[VI]
    I HAS A NEWADJ2 ITZ ""
    I HAS A J2 ITZ 0
    I HAS A CUR2 ITZ ADJ
    I HAS A N2 ITZ I IZ LEN YR ADJ MKAY
    IM IN YR a1
        BOTH SAEM J2 AN N2 ? O RLY?
            YA RLY, GTFO
        OIC
        I HAS A AROW2
        AROW2 R I IZ GETTOK YR CUR2 MKAY
        CUR2 R I IZ GETLASTREM MKAY
        BOTH SAEM J2 AN VI ? O RLY?
            YA RLY
                I HAS A rowbag2 ITZ AROW2
                I HAS A has2 ITZ 0
                IM IN YR ch2
                    BOTH SAEM rowbag2 AN "" ? O RLY?
                        YA RLY, GTFO
                    OIC
                    I HAS A t2 ITZ I IZ GETTOK YR rowbag2 MKAY
                    rowbag2 R I IZ GETLASTREM MKAY
                    BOTH SAEM t2 AN U ? O RLY?
                        YA RLY
                            has2 R 1
                            GTFO
                    OIC
                IM OUTTA YR ch2
                O RLY?
                    YA RLY, BOTH SAEM has2 AN 0
                        AROW2 R I IZ PUSH YR AROW2 AN YR U MKAY
                OIC
        OIC
        NEWADJ2 R I IZ PUSH YR NEWADJ2 AN YR AROW2 MKAY
        J2 R SUM OF J2 AN 1
    IM OUTTA YR a1
    ADJ R NEWADJ2
IF U SAY SO

BTW ----------------------------------------------------------------------------
BTW  BFS queue over label-strings. PARENT is “label->label” edges recorded
BTW ----------------------------------------------------------------------------
HOW IZ I BFS YR SRC AN YR DST
    BTW Early outs
    BOTH SAEM SRC AN DST ? O RLY?
        YA RLY
            FOUND YR SMOOSH SRC MKAY
    OIC

    I HAS A VIS ITZ ""        BTW bag of visited labels
    I HAS A Q ITZ SRC         BTW queue as space-separated labels
    I HAS A PARENT ITZ ""     BTW bag of "child:parent" pairs flattened as child parent

    IM IN YR loop
        BOTH SAEM Q AN "" ? O RLY?
            YA RLY, GTFO
        OIC
        I HAS A U ITZ I IZ GETTOK YR Q MKAY
        Q R I IZ GETLASTREM MKAY

        BTW if already visited, skip
        I HAS A already ITZ I IZ HASLABEL YR VIS AN YR U MKAY
        O RLY?
            YA RLY, BOTH SAEM already AN 1
                BTW skip
                NERF
            NO WAI
                VIS R I IZ PUSH YR VIS AN YR U MKAY

                BOTH SAEM U AN DST ? O RLY?
                    YA RLY, GTFO
                OIC

                BTW enqueue neighbors
                I HAS A UI ITZ I IZ FINDIDX YR U MKAY
                I HAS A row ITZ I IZ GETROW YR ADJ AN YR UI MKAY
                IM IN YR n0
                    BOTH SAEM row AN "" ? O RLY?
                        YA RLY, GTFO
                    OIC
                    I HAS A V ITZ I IZ GETTOK YR row MKAY
                    row R I IZ GETLASTREM MKAY
                    I HAS A vseen ITZ I IZ HASLABEL YR VIS AN YR V MKAY
                    O RLY?
                        YA RLY, BOTH SAEM vseen AN 0
                            Q R I IZ PUSH YR Q AN YR V MKAY
                            PARENT R I IZ PUSH YR PARENT AN YR V MKAY
                            PARENT R I IZ PUSH YR PARENT AN YR U MKAY
                    OIC
                IM OUTTA YR n0
        OIC
    IM OUTTA YR loop

    BTW reconstruct if reached
    I HAS A reached ITZ I IZ HASLABEL YR VIS AN YR DST MKAY
    BOTH SAEM reached AN 0 ? O RLY?
        YA RLY, FOUND YR "NO PATH"
    OIC

    I HAS A PATH ITZ ""
    I HAS A CUR ITZ DST
    IM IN YR rec
        PATH R I IZ PUSH YR PATH AN YR CUR MKAY
        BOTH SAEM CUR AN SRC ? O RLY?
            YA RLY, GTFO
        OIC
        CUR R I IZ GETPARENT YR PARENT AN YR CUR MKAY
        BOTH SAEM CUR AN "" ? O RLY?
            YA RLY, GTFO
        OIC
    IM OUTTA YR rec

    FOUND YR I IZ REVERSEBAG YR PATH MKAY
IF U SAY SO

HOW IZ I HASLABEL YR BAG AN YR L
    I HAS A cur ITZ BAG
    IM IN YR h0
        BOTH SAEM cur AN "" ? O RLY?
            YA RLY, FOUND YR 0
        OIC
        I HAS A t ITZ I IZ GETTOK YR cur MKAY
        cur R I IZ GETLASTREM MKAY
        BOTH SAEM t AN L ? O RLY?
            YA RLY, FOUND YR 1
        OIC
    IM OUTTA YR h0
IF U SAY SO

HOW IZ I GETROW YR BAG AN YR IDX
    I HAS A i ITZ 0
    I HAS A n ITZ I IZ LEN YR BAG MKAY
    I HAS A cur ITZ BAG
    IM IN YR r0
        BOTH SAEM i AN n ? O RLY?
            YA RLY, FOUND YR ""
        OIC
        I HAS A t ITZ I IZ GETTOK YR cur MKAY
        cur R I IZ GETLASTREM MKAY
        BOTH SAEM i AN IDX ? O RLY?
            YA RLY, FOUND YR t
        OIC
        i R SUM OF i AN 1
    IM OUTTA YR r0
IF U SAY SO

HOW IZ I GETPARENT YR FLAT AN YR CHILD
    BTW FLAT bag looks like: "V1 U1 V2 U2 ..."
    I HAS A cur ITZ FLAT
    IM IN YR p0
        BOTH SAEM cur AN "" ? O RLY?
            YA RLY, FOUND YR ""
        OIC
        I HAS A a ITZ I IZ GETTOK YR cur MKAY
        cur R I IZ GETLASTREM MKAY
        I HAS A b ITZ I IZ GETTOK YR cur MKAY
        cur R I IZ GETLASTREM MKAY
        BOTH SAEM a AN CHILD ? O RLY?
            YA RLY, FOUND YR b
        OIC
    IM OUTTA YR p0
IF U SAY SO

HOW IZ I REVERSEBAG YR BAG
    I HAS A out ITZ ""
    I HAS A stk ITZ BAG
    I HAS A tmp ITZ ""
    BTW collect into a list
    IM IN YR c0
        BOTH SAEM stk AN "" ? O RLY?
            YA RLY, GTFO
        OIC
        I HAS A t ITZ I IZ GETTOK YR stk MKAY
        stk R I IZ GETLASTREM MKAY
        tmp R I IZ PUSH YR tmp AN YR t MKAY
    IM OUTTA YR c0
    BTW now push back in reverse (walk tmp and prepend — emulate by rebuild)
    I HAS A n ITZ I IZ LEN YR tmp MKAY
    IM IN YR c1
        BOTH SAEM n AN 0 ? O RLY?
            YA RLY, GTFO
        OIC
        I HAS A cur ITZ ""
        I HAS A t2bag ITZ tmp
        I HAS A j ITZ 0
        IM IN YR c2
            BOTH SAEM j AN n ? O RLY?
                YA RLY, GTFO
            OIC
            I HAS A t2 ITZ I IZ GETTOK YR t2bag MKAY
            t2bag R I IZ GETLASTREM MKAY
            O RLY?
                YA RLY, BOTH SAEM j AN DIFF OF n AN 1
                    cur R t2
            OIC
            j R SUM OF j AN 1
        IM OUTTA YR c2
        out R I IZ PUSH YR out AN YR cur MKAY
        BTW drop last from tmp
        I HAS A rebuilt ITZ ""
        I HAS A t3bag ITZ tmp
        I HAS A jj ITZ 0
        IM IN YR c3
            BOTH SAEM jj AN DIFF OF n AN 1 ? O RLY?
                YA RLY, GTFO
            OIC
            I HAS A t3 ITZ I IZ GETTOK YR t3bag MKAY
            t3bag R I IZ GETLASTREM MKAY
            rebuilt R I IZ PUSH YR rebuilt AN YR t3 MKAY
            jj R SUM OF jj AN 1
        IM OUTTA YR c3
        tmp R rebuilt
        n R DIFF OF n AN 1
    IM OUTTA YR c1
    FOUND YR out
IF U SAY SO

BTW ----------------------------------------------------------------------------
BTW  MAIN: read lines, build graph, answer queries
BTW ----------------------------------------------------------------------------
I HAS A LINE
I HAS A QSRC
I HAS A QDST

IM IN YR readloop
    GIMMEH LINE
    BOTH SAEM LINE AN "" ? O RLY?
        YA RLY, GTFO
    OIC

    I HAS A LITZ ITZ I IZ TRIM YR LINE MKAY
    BOTH SAEM LITZ AN "" ? O RLY?
        YA RLY, CONTINYU
    OIC

    BTW Determine if it’s a query or an edge line.
    I HAS A FIRST ITZ I IZ HEADTOK YR LITZ MKAY
    BOTH SAEM FIRST AN "#" ? O RLY?
        YA RLY
            I HAS A REST ITZ I IZ TAILTOKS YR LITZ MKAY
            I HAS A SRC ITZ I IZ HEADTOK YR REST MKAY
            I HAS A REST2 ITZ I IZ TAILTOKS YR REST MKAY
            I HAS A DST ITZ I IZ HEADTOK YR REST2 MKAY

            I HAS A PATH ITZ I IZ BFS YR SRC AN YR DST MKAY
            BOTH SAEM PATH AN "NO PATH" ? O RLY?
                YA RLY
                    VISIBLE SMOOSH "No path found from " SRC " to " DST MKAY
                NO WAI
                    VISIBLE I IZ JOINARROWS YR PATH MKAY
            OIC
        NO WAI
            BTW parse edges: U V1 V2 ...
            I HAS A U ITZ FIRST
            I HAS A REST ITZ I IZ TAILTOKS YR LITZ MKAY
            O RLY?
                YA RLY, BOTH SAEM REST AN ""
                    BTW single node line (ensure existence)
                    I HAS A _ ITZ I IZ ENSURENODE YR U MKAY
                NO WAI
                    IM IN YR addn
                        BOTH SAEM REST AN "" ? O RLY?
                            YA RLY, GTFO
                        OIC
                        I HAS A V ITZ I IZ HEADTOK YR REST MKAY
                        REST R I IZ TAILTOKS YR REST MKAY
                        I IZ ADDEDGE YR U AN YR V MKAY
                    IM OUTTA YR addn
            OIC
    OIC
IM OUTTA YR readloop

BTW ----------------------------------------------------------------------------
BTW  Token helpers for a whitespace-separated line
BTW ----------------------------------------------------------------------------
HOW IZ I HEADTOK YR S
    I HAS A cur ITZ S
    I HAS A TOK ITZ I IZ GETTOK YR cur MKAY
    FOUND YR TOK
IF U SAY SO

HOW IZ I TAILTOKS YR S
    I HAS A cur ITZ S
    I HAS A _ ITZ I IZ GETTOK YR cur MKAY
    FOUND YR I IZ GETLASTREM MKAY
IF U SAY SO

HOW IZ I JOINARROWS YR BAG
    I HAS A out ITZ ""
    I HAS A cur ITZ BAG
    I HAS A first ITZ 1
    IM IN YR j0
        BOTH SAEM cur AN "" ? O RLY?
            YA RLY, GTFO
        OIC
        I HAS A t ITZ I IZ GETTOK YR cur MKAY
        cur R I IZ GETLASTREM MKAY
        O RLY?
            YA RLY, BOTH SAEM first AN 1
                out R t
                first R 0
            NO WAI
                out R SMOOSH out " -> " t MKAY
        OIC
    IM OUTTA YR j0
    FOUND YR out
IF U SAY SO

KTHXBYE

