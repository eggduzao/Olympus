;;;; binsearch.lisp — Binary search (first occurrence) on a sorted ascending list.
;;;; Tested with SBCL/CLISP. Run:
;;;;   sbcl --script binsearch.lisp -- 11 data.txt
;;;;   # or via STDIN:
;;;;   printf "1\n3\n4\n7\n9\n11\n15\n" | sbcl --script binsearch.lisp -- 11
;;;; Behavior:
;;;;   • O(log N) binary search (lower-bound; first index of target if duplicates).
;;;;   • Prints human-friendly result; exits with non-zero status on malformed input.

(defpackage :binsearch
  (:use :cl)
  (:export :main))

(in-package :binsearch)

(defun eprintln (fmt &rest args)
  (apply #'format *error-output* (concatenate 'string fmt "~%") args))

(defun parse-int-or-nil (s)
  "Return integer parsed from trimmed string S or NIL if not a base-10 int."
  (let* ((t (string-trim '(#\Space #\Tab #\Newline #\Return) (or s ""))))
    (when (plusp (length t))
      (multiple-value-bind (n pos) (parse-integer t :junk-allowed t)
        (when (and n (= pos (length t))) n)))))

(defun read-numbers (path)
  "Read newline-separated integers from PATH or *STANDARD-INPUT* if PATH is NIL.
   Warns about non-integer lines, skips blanks."
  (let ((stream (if path
                    (open path :direction :input :if-does-not-exist nil)
                    *standard-input*)))
    (unless stream
      (eprintln "ERROR: cannot read file ~a" path)
      (uiop:quit 2))
    (unwind-protect
         (loop for line = (read-line stream nil :eof)
               for i from 1
               until (eq line :eof)
               for v = (parse-int-or-nil line)
               when (and (null v)
                         (plusp (length (string-trim '(#\Space #\Tab) line))))
                 do (eprintln "WARN: skipping non-integer line ~d: ~a" i line)
               when v collect v)
      (when path (close stream)))))

(defun ensure-ascending (vec)
  "Signal error (and exit) if VEC is not non-decreasing."
  (dotimes (i (1- (length vec)))
    (when (< (aref vec (1+ i)) (aref vec i))
      (eprintln "ERROR: input not in ascending order at position ~d: ~a < ~a"
                (1+ (1+ i)) (aref vec (1+ i)) (aref vec i))
      (uiop:quit 3))))

(defun lower-bound (vec target)
  "Return the smallest index i in [0..n] s.t. VEC[i] >= TARGET (binary search).
   If all elements < target, returns length(vec)."
  (let* ((n (length vec))
         (lo 0)
         (hi n))
    (loop while (< lo hi) do
      (let ((mid (truncate (+ lo hi) 2)))
        (if (< (aref vec mid) target)
            (setf lo (1+ mid))
            (setf hi mid))))
    lo))

(defun report (vec target)
  (let* ((n (length vec))
         (ins (lower-bound vec target)))
    (if (and (< ins n) (= (aref vec ins) target))
        (format t "FOUND ~a at index ~d (1-based)~%" target (1+ ins))
        (let ((left  (if (>= ins 1) (princ-to-string (aref vec (1- ins))) "-inf"))
              (right (if (< ins n)  (princ-to-string (aref vec ins))        "+inf")))
          (format t "NOT FOUND ~a. Insertion index ~d (1-based), between ~a and ~a~%"
                  target (1+ ins) left right)))))

(defun main ()
  "Entry point. Usage:
     sbcl --script binsearch.lisp -- <target> [path/to/file]
     printf \"...\\n\" | sbcl --script binsearch.lisp -- <target>"
  ;; Command line after '--' is in uiop:*command-line-arguments* if UIOP is present.
  (let* ((args (or (ignore-errors uiop:*command-line-arguments*)
                   ;; Fallback for implementations without UIOP parsing:
                   (rest (rest *posix-argv*))))
         (target-str (first args))
         (path (second args)))
    (unless target-str
      (eprintln "Usage: sbcl --script binsearch.lisp -- <target> [path/to/file]")
      (uiop:quit 1))
    (let ((target (parse-int-or-nil target-str)))
      (unless target
        (eprintln "ERROR: target must be an integer, got '~a'" target-str)
        (uiop:quit 1))
      (let* ((nums-list (read-numbers path)))
        (when (null nums-list)
          (eprintln "ERROR: no numeric input provided.")
          (uiop:quit 2))
        (let* ((vec (coerce nums-list 'vector)))
          (ensure-ascending vec)
          (report vec target))))))

;; Run when invoked as a script
(main)
