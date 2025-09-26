;; binsearch.scm — Binary search (first occurrence) on a sorted ascending list.
;; Portable Scheme (tested on Guile/Chicken). Run examples:
;;   guile binsearch.scm -- 11 data.txt
;;   echo "1\n3\n4\n7\n9\n11\n15\n" | guile binsearch.scm -- 11
;;
;; Behavior:
;;   • O(log N) binary search (lower-bound): returns first index of TARGET if present.
;;   • Reads newline-separated integers from a file or STDIN.
;;   • Validates ascending order; warns on non-integer lines.

;; ---------- tiny utils ----------
(define (eprintln . xs)
  (let ((out (current-error-port)))
    (for-each (lambda (x) (display x out)) xs)
    (newline out)))

(define (string-trim s)
  (let* ((len (string-length s))
         (start (let loop ((i 0))
                  (if (and (< i len)
                           (or (char=? (string-ref s i) #\space)
                               (char=? (string-ref s i) #\tab)
                               (char=? (string-ref s i) #\newline)
                               (char=? (string-ref s i) #\return)))
                      (loop (+ i 1))
                      i)))
         (end (let loop ((i (- len 1)))
                (if (and (>= i 0)
                         (or (char=? (string-ref s i) #\space)
                             (char=? (string-ref s i) #\tab)
                             (char=? (string-ref s i) #\newline)
                             (char=? (string-ref s i) #\return)))
                    (loop (- i 1))
                    i))))
    (if (> end start)
        (substring s start (+ end 1))
        "")))

(define (parse-int-or-false s)
  (let* ((t (string-trim (or s ""))))
    (if (zero? (string-length t))
        #f
        (let ((n (string->number t)))
          (if (and n (integer? n)) (inexact->exact n) #f)))))

;; ---------- IO ----------
(define (read-numbers-from-port port)
  (let loop ((line (read-line port 'concat))
             (i 1)
             (acc '()))
    (if (eof-object? line)
        (reverse acc)
        (let ((v (parse-int-or-false line)))
          (cond
            ((and (not v)
                  (positive? (string-length (string-trim line))))
             (eprintln "WARN: skipping non-integer line " i ": " line)
             (loop (read-line port 'concat) (+ i 1) acc))
            (v (loop (read-line port 'concat) (+ i 1) (cons v acc)))
            (else (loop (read-line port 'concat) (+ i 1) acc)))))))

(define (read-numbers path)
  (if path
      (let ((p (open-input-file path)))
        (let ((xs (read-numbers-from-port p)))
          (close-input-port p)
          xs))
      (read-numbers-from-port (current-input-port))))

;; ---------- checks & search ----------
(define (ensure-ascending vec)
  (let ((n (vector-length vec)))
    (let loop ((i 0))
      (when (< i (- n 1))
        (when (< (vector-ref vec (+ i 1)) (vector-ref vec i))
          (eprintln "ERROR: input not in ascending order at position "
                    (+ i 2) ": "
                    (vector-ref vec (+ i 1)) " < " (vector-ref vec i))
          (exit 3))
        (loop (+ i 1))))))

;; lower-bound: first index i with vec[i] >= target; returns n if all < target
(define (lower-bound vec target)
  (let ((n (vector-length vec)))
    (let loop ((lo 0) (hi n))
      (if (< lo hi)
          (let* ((mid (quotient (+ lo hi) 2))
                 (vm (vector-ref vec mid)))
            (if (< vm target)
                (loop (+ mid 1) hi)
                (loop lo mid)))
          lo))))

(define (report vec target)
  (let* ((n (vector-length vec))
         (ins (lower-bound vec target)))
    (if (and (< ins n) (= (vector-ref vec ins) target))
        (begin
          (display "FOUND ") (display target)
          (display " at index ") (display (+ ins 1)) ; 1-based
          (newline))
        (let ((left  (if (>= ins 1) (number->string (vector-ref vec (- ins 1))) "-inf"))
              (right (if (< ins n)  (number->string (vector-ref vec ins))        "+inf")))
          (display "NOT FOUND ") (display target)
          (display ". Insertion index ") (display (+ ins 1))
          (display " (1-based), between ") (display left)
          (display " and ") (display right)
          (newline)))))

;; ---------- entrypoint ----------
(define (usage)
  (eprintln "Usage: scheme -q binsearch.scm -- <target-int> [path/to/file]")
  (eprintln "       echo \"1\\n3\\n...\" | scheme -q binsearch.scm -- 11"))

(define (main args)
  ;; args after '--': (<target> [path])
  (if (null? args)
      (begin (usage) (exit 1))
      (let* ((target-str (car args))
             (path (and (pair? (cdr args)) (cadr args)))
             (target (parse-int-or-false target-str)))
        (if (not target)
            (begin (eprintln "ERROR: target must be an integer, got '" target-str "'")
                   (exit 1))
            (let* ((nums (read-numbers path)))
              (when (null? nums)
                (eprintln "ERROR: no numeric input provided.") (exit 2))
              (let ((vec (list->vector nums)))
                (ensure-ascending vec)
                (report vec target)))))))

;; Many Schemes expose the full argv vector; we look for the conventional “--”.
(let* ((argv (command-line))                         ; e.g., ("guile" "binsearch.scm" "--" "11" "data.txt")
       (after-dashdash
        (let loop ((lst argv))
          (cond
            ((null? lst) '())
            ((string=? (car lst) "--") (cdr lst))
            (else (loop (cdr lst)))))))
  (main after-dashdash))

