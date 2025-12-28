;; bfs_shortest_path.scm
;; Scheme (Racket) script: Breadth-First Search (BFS) shortest path on an unweighted, UNDIRECTED graph.
;;
;; INPUT (tab/space separated), one item per line:
;;   Edge lines : "U  V1 [V2 ...]"  => add undirected edges U—V1, U—V2, ...
;;   Query line : "#  SRC DST"      => request one shortest path from SRC to DST
;;
;; Example:
;;   A   B   F
;;   B   A   C
;;   C   B   D
;;   D   C   E
;;   E   D   F
;;   F   A   E
;;   #   A   E
;;
;; USAGE
;;   racket bfs_shortest_path.scm < graph.tsv
;;   # or
;;   racket bfs_shortest_path.scm graph.tsv
;;
;; OUTPUT
;;   For each query line, either a path like "A -> B -> C"
;;   or "No path found from SRC to DST"

#lang racket

(require racket/string
         racket/list
         racket/port
         racket/pretty
         racket/set
         data/queue)

;; -----------------------------
;; Helpers
;; -----------------------------

(define (split-ws s)
  ;; Split S on runs of whitespace, dropping empties.
  (filter (λ (x) (and x (not (string-empty? x))))
          (regexp-split #px"\\s+" s)))

(define (ensure! h k [maker (λ () (mutable-set))])
  ;; Ensure hash table H has key K; initialize with (maker)
  (hash-ref! h k maker))

(define (add-undirected-edge! graph u v)
  ;; GRAPH : hash string -> (mutable-setof string)
  (define nu (ensure! graph u))
  (define nv (ensure! graph v))
  (set-add! nu v)
  (set-add! nv u)
  (void))

(define (read-all-lines in)
  (port->lines in))

;; -----------------------------
;; BFS shortest path
;; -----------------------------

(define (bfs-shortest-path graph src dst)
  ;; GRAPH: hash string -> neighbor-set (mutable set of string)
  ;; Return (list src ... dst) if reachable; #f otherwise.
  (cond
    [(string=? src dst) (list src)]
    [(or (not (hash-has-key? graph src))
         (not (hash-has-key? graph dst)))
     #f]
    [else
     (define visited (make-hash))          ; string -> #t
     (define parent  (make-hash))          ; string -> string
     (define q (make-queue))
     (hash-set! visited src #t)
     (enqueue! q src)
     (let loop ()
       (cond
         [(queue-empty? q) #f]
         [else
          (define u (dequeue! q))
          (for ([v (in-set (hash-ref graph u (mutable-set)))])
            (unless (hash-has-key? visited v)
              (hash-set! visited v #t)
              (hash-set! parent  v u)
              (when (string=? v dst)
                (define (reconstruct cur acc)
                  (if (string=? cur src)
                      (cons src acc)
                      (reconstruct (hash-ref parent cur) (cons cur acc))))
                (return-from loop (reconstruct v '()))))
            (void))
          (loop)]))]))

;; -----------------------------
;; Parsing + Main
;; -----------------------------

(struct parsed (graph queries) #:transparent)
;; queries: list of (cons src dst)

(define (parse-graph-and-queries lines)
  (define graph (make-hash)) ; node -> neighbor-set
  (define queries '())
  (for ([ln (in-list lines)])
    (define t (string-trim ln))
    (when (positive? (string-length t))
      (define parts (split-ws t))
      (when (pair? parts)
        (if (string=? (first parts) "#")
            (when (>= (length parts) 3)
              (set! queries (cons (cons (second parts) (third parts)) queries)))
            (let ([u (first parts)]
                  [vs (rest parts)])
              (void (ensure! graph u)) ; ensure node exists even if isolated
              (for ([v (in-list vs)])
                (add-undirected-edge! graph u v)))))))
  (parsed graph (reverse queries)))

(define (emit-path path)
  (displayln (string-join path " -> ")))

(define (run in)
  (define p (parse-graph-and-queries (read-all-lines in)))
  (define graph   (parsed-graph p))
  (define queries (parsed-queries p))
  (if (null? queries)
      (begin
        (displayln "No queries found (expected lines like: \"# SRC DST\").")
        (void))
      (for ([q (in-list queries)])
        (define src (car q))
        (define dst (cdr q))
        (define path (bfs-shortest-path graph src dst))
        (if path
            (emit-path path)
            (displayln (format "No path found from ~a to ~a" src dst))))))

;; -----------------------------
;; Entry point
;; -----------------------------

(define (main)
  (define args (current-command-line-arguments))
  (cond
    [(= (vector-length args) 0)
     (run (current-input-port))]
    [else
     (define file (vector-ref args 0))
     (call-with-input-file* file run)]))

(module+ main
  (main))

