;;;; bfs_shortest_path.lisp
;;;; Common Lisp script: Breadth-First Search (BFS) shortest path on an unweighted, UNDIRECTED graph.
;;;;
;;;; INPUT (whitespace/tab separated):
;;;;   Edge lines : "U  V1 [V2 ...]"  => add undirected edges U—V1, U—V2, ...
;;;;   Query line : "#  SRC DST"      => request one shortest path from SRC to DST
;;;;
;;;; Example:
;;;;   A   B   F
;;;;   B   A   C
;;;;   C   B   D
;;;;   D   C   E
;;;;   E   D   F
;;;;   F   A   E
;;;;   #   A   E
;;;;
;;;; USAGE
;;;;   sbcl --script bfs_shortest_path.lisp < graph.tsv
;;;;   # or
;;;;   sbcl --script bfs_shortest_path.lisp graph.tsv
;;;;
;;;; OUTPUT
;;;;   For each query line, either a path like "A -> B -> C" or "No path found from SRC to DST".

;;;; ------------------------------------------------------------
;;;; Utilities
;;;; ------------------------------------------------------------

(defun split-on-whitespace (line)
  "Return a list of non-empty tokens split on runs of whitespace in LINE."
  (let ((len (length line))
        (tokens '())
        (start nil))
    (labels ((whitespacep (ch)
               (or (char= ch #\Space)
                   (char= ch #\Tab)
                   (char= ch #\Newline)
                   (char= ch #\Return))))
      (loop for i from 0 below len
            for ch = (char line i) do
              (cond
                ((whitespacep ch)
                 (when start
                   (push (subseq line start i) tokens)
                   (setf start nil)))
                (t
                 (unless start (setf start i)))))
      (when start
        (push (subseq line start len) tokens))
      (nreverse tokens))))

(defun ensure (table key &key (value-maker (lambda () (make-hash-table :test #'equal))))
  "Ensure TABLE has KEY; if absent, set to (FUNCALL VALUE-MAKER) and return it.
For presence, return existing value."
  (multiple-value-bind (val presentp) (gethash key table)
    (if presentp
        val
        (setf (gethash key table) (funcall value-maker)))))

(defun add-undirected-edge (graph u v)
  "Add undirected edge u—v to GRAPH (a hash-table of node => neighbor-set (hash-table))."
  (let ((nu (ensure graph u))
        (nv (ensure graph v)))
    (setf (gethash v nu) t)
    (setf (gethash u nv) t)))

(defun read-all-lines (stream)
  "Read all lines from STREAM, return list of strings in order."
  (let ((lines '()))
    (loop for line = (read-line stream nil :eof)
          until (eq line :eof)
          do (push line lines))
    (nreverse lines)))

(defun get-cmd-args ()
  "Return command line arguments (excluding the script path) portably across common CLs.
If not detected, return NIL."
  (cond
    (#-sbcl nil #+sbcl (cdr sb-ext:*posix-argv*))
    (#-ccl nil #+ccl ccl:*unprocessed-command-line-arguments*)
    (#-ecl nil #+ecl (ext:command-args))
    (#-clisp nil #+clisp (ext:argv))
    (t nil)))

;;;; ------------------------------------------------------------
;;;; Queue (simple linked-list tail queue)
;;;; ------------------------------------------------------------

(defstruct (queue (:constructor make-queue))
  head
  tail)

(defun q-empty-p (q) (null (queue-head q)))

(defun q-enqueue (q item)
  "Enqueue ITEM in Q in O(1)."
  (let ((cell (list item)))
    (if (queue-tail q)
        (setf (cdr (queue-tail q)) cell
              (queue-tail q) cell)
        (setf (queue-head q) cell
              (queue-tail q) cell)))
  q)

(defun q-dequeue (q)
  "Dequeue and return next item from Q, or NIL if empty."
  (let ((h (queue-head q)))
    (when h
      (let ((item (car h)))
        (setf (queue-head q) (cdr h))
        (unless (queue-head q)
          (setf (queue-tail q) nil))
        item))))

;;;; ------------------------------------------------------------
;;;; BFS Shortest Path
;;;; ------------------------------------------------------------

(defun bfs-shortest-path (graph src dst)
  "Return shortest path as a list of node labels from SRC to DST inclusive, or NIL if unreachable.
GRAPH is a hash-table mapping node(string) => neighbor-set(hash-table)."
  (when (string= src dst)
    (return-from bfs-shortest-path (list src)))
  (unless (and (gethash src graph) (gethash dst graph))
    (return-from bfs-shortest-path nil))
  (let ((visited (make-hash-table :test #'equal))
        (parent  (make-hash-table :test #'equal))
        (q (make-queue)))
    (setf (gethash src visited) t)
    (q-enqueue q src)
    (loop until (q-empty-p q) do
      (let* ((u (q-dequeue q))
             (neighbors (gethash u graph)))
        (maphash
         (lambda (v _)
           (declare (ignore _))
           (unless (gethash v visited)
             (setf (gethash v visited) t
                   (gethash v parent)  u)
             (when (string= v dst)
               (return-from bfs-shortest-path
                 (labels ((reconstruct (cur acc)
                            (if (string= cur src)
                                (cons src acc)
                                (reconstruct (gethash cur parent) (cons cur acc)))))
                   (reconstruct v '()))))
             (q-enqueue q v)))
         neighbors)))
    nil))

(defun join-with (items sep)
  "Join list of strings ITEMS with separator SEP."
  (with-output-to-string (out)
    (loop for item in items
          for i from 0 do
            (when (> i 0) (princ sep out))
            (princ item out))))

;;;; ------------------------------------------------------------
;;;; Parsing + Main
;;;; ------------------------------------------------------------

(defun parse-graph-and-queries (lines)
  "Parse LINES (list of strings). Return (values GRAPH QUERIES).
GRAPH: hash-table node=>neighbor-set
QUERIES: list of (SRC . DST)."
  (let ((graph (make-hash-table :test #'equal))
        (queries '()))
    (dolist (ln lines)
      (let ((ln (string-trim '(#\Space #\Tab #\Return #\Newline) ln)))
        (when (plusp (length ln))
          (let ((parts (split-on-whitespace ln)))
            (when parts
              (if (string= (first parts) "#")
                  (when (>= (length parts) 3)
                    (push (cons (second parts) (third parts)) queries))
                  (let ((u (first parts)))
                    (ensure graph u) ; ensure node exists even if isolated
                    (dolist (v (rest parts))
                      (add-undirected-edge graph u v)))))))))
    (values graph (nreverse queries))))

(defun run (in-stream)
  (multiple-value-bind (graph queries)
      (parse-graph-and-queries (read-all-lines in-stream))
    (if (null queries)
        (format *error-output* "No queries found (expected lines like: \"# SRC DST\").~%")
        (dolist (q queries)
          (destructuring-bind (src . dst) q
            (let ((path (bfs-shortest-path graph src dst)))
              (if path
                  (format t "~a~%" (join-with path " -> "))
                  (format t "No path found from ~a to ~a~%" src dst))))))))

(defun main ()
  "Entry point: reads from file path given on CLI or from STDIN."
  (let* ((args (get-cmd-args))
         (file (and args (second args)))) ; first arg is the implementation path in some CLs
    (cond
      ((and file (not (string= file "")) (probe-file file))
       (with-open-file (in file :direction :input :external-format :utf-8)
         (run in)))
      (t
       (run *standard-input*)))))

;;;; ------------------------------------------------------------
;;;; Auto-run when used as a 'script'
;;;; ------------------------------------------------------------
;; Many CLs set *LOAD-TRUENAME* when loading; when run via --script, call MAIN.
(when (or (member :sbcl *features*) (member :ccl *features*) (member :ecl *features*) (member :clisp *features*))
  (when (and (boundp '*load-truename*) *load-truename*)
    ;; If being loaded as a script, try to run MAIN. This is harmless if loaded in REPL.
    (ignore-errors (main))))

