;;;; InPlaceQuicksort.lisp
;;;; Common Lisp — In-place quicksort on an array (simple-vector).
;;;;
;;;; • Reads integers (one per line) from a file given as the first CLI arg,
;;;;   or from STDIN if no file is provided.
;;;; • Skips blank lines and lines starting with '#'.
;;;; • Uses an iterative quicksort (manual stack) with Hoare partition
;;;;   and median-of-three pivot selection.
;;;; • Prints sorted numbers to STDOUT, one per line.
;;;;
;;;; Run (examples):
;;;;   sbcl --script InPlaceQuicksort.lisp data.txt
;;;;   cat data.txt | sbcl --script InPlaceQuicksort.lisp
;;;;   printf "5\n3\n8\n1\n2\n" | sbcl --script InPlaceQuicksort.lisp

(defpackage :inplace-qsort
  (:use :cl))
(in-package :inplace-qsort)

;; ---------- CLI argument helper (best-effort, cross-impl) ----------
(defun first-cli-arg ()
  "Return the first CLI argument if available, else NIL.
   Tries several implementation-specific variables."
  (or
   ;; SBCL
   #+sbcl (let ((argv sb-ext:*posix-argv*))
            (when (and argv (> (length argv) 1))
              (nth 1 argv)))
   ;; Clozure CL
   #+ccl (let ((argv ccl:*command-line-argument-list*))
           (when (and argv (> (length argv) 0))
             (first argv)))
   ;; ECL
   #+ecl (let ((argv (ext:command-args)))
           (when (and argv (> (length argv) 1))
             (second argv)))
   ;; Allegro (very rough)
   #+allegro (let ((argv (sys:command-line-arguments)))
               (when (and argv (> (length argv) 0))
                 (first argv)))
   ;; Fallback: no arg known
   nil))

;; ---------- Input parsing ----------
(defun read-int-lines (stream)
  "Read integers from STREAM (one per line), skipping blanks and lines starting with '#'.
   Returns a fresh simple-vector of integers."
  (let* ((capacity 1024)
         (vec (make-array capacity :element-type 'fixnum :adjustable t :fill-pointer 0)))
    (labels ((ensure-capacity ()
               (when (= (length vec) (array-total-size vec))
                 (adjust-array vec (* 2 (length vec)))))
             (push-int (n)
               (ensure-capacity)
               (vector-push-extend n vec)))
      (loop for line = (read-line stream nil nil)
            while line do
              (let* ((s (string-trim '(#\Space #\Tab #\Newline #\Return) line)))
                (when (and (> (length s) 0)
                           (char/= (char s 0) #\#))
                  (unless (every (lambda (c) (or (digit-char-p c) (find c "+-" :test #'char=))) s)
                    (error "Invalid integer line: ~S" s))
                  (push-int (parse-integer s)))))
      (coerce vec 'simple-vector))))

;; ---------- Quicksort (in place) ----------
(declaim (inline swap))
(defun swap (a i j)
  (declare (type (simple-vector) a)
           (type fixnum i j))
  (let ((tmp (svref a i)))
    (setf (svref a i) (svref a j)
          (svref a j) tmp)))

(defun median3 (x y z)
  "Median of three numbers (returns the middle value)."
  (if (or (and (<= x y) (<= y z))
          (and (<= z y) (<= y x)))
      y
      (if (or (and (<= y x) (<= x z))
              (and (<= z x) (<= x y)))
          x
          z)))

(defun hoare-partition (a l h)
  "Hoare partition with median-of-three pivot.
   Returns j such that all elements in [l..j] <= pivot and [j+1..h] >= pivot."
  (declare (type (simple-vector) a)
           (type fixnum l h))
  (let* ((m (+ l (truncate (- h l) 2)))
         (pivot (median3 (svref a l) (svref a m) (svref a h)))
         (i (1- l))
         (j (1+ h)))
    (declare (type fixnum m i j))
    (loop
      (loop do (incf i) while (< (svref a i) pivot))
      (loop do (decf j) while (> (svref a j) pivot))
      (when (>= i j) (return j))
      (swap a i j))))

(defun quicksort-in-place (a)
  "Iterative in-place quicksort on simple-vector A (ascending)."
  (declare (type (simple-vector) a))
  (let ((stack (list (cons 0 (1- (length a))))))
    (declare (type list stack))
    (loop while stack do
      (destructuring-bind (l . h) (pop stack)
        (declare (type fixnum l h))
        (when (< l h)
          (let ((p (hoare-partition a l h)))
            ;; Push larger side first? We'll push both; order doesn't matter much here.
            (when (< l p)       (push (cons l p) stack))
            (when (< (1+ p) h)  (push (cons (1+ p) h) stack)))))))
  a)

;; ---------- Output ----------
(defun print-ints (a)
  (declare (type (simple-vector) a))
  (loop for i from 0 below (length a) do
        (format t "~D~%" (svref a i))))

;; ---------- Entrypoint ----------
(defun main ()
  (let ((arg (first-cli-arg)))
    (let ((data (if arg
                    (with-open-file (in arg :direction :input)
                      (read-int-lines in))
                    (read-int-lines *standard-input*))))
      (quicksort-in-place data)
      (print-ints data))))

;; Auto-run when loaded as a script (not when compiled/loaded as a library).
(when (and *load-pathname* (not *compile-file-truename*))
  (main))

