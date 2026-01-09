;; Levenshtein edit distance in Scheme (R5RS-ish, assumes read-line is available)

(define (levenshtein s1 s2)
  (let* ((m (string-length s1))
         (n (string-length s2))
         (cols (+ n 1))
         (dp (make-vector (* (+ m 1) cols) 0)))
    (define (idx i j) (+ (* i cols) j))

    ;; Initialize first row and column
    (let loopi ((i 0))
      (when (<= i m)
        (vector-set! dp (idx i 0) i)
        (loopi (+ i 1))))
    (let loopj ((j 0))
      (when (<= j n)
        (vector-set! dp (idx 0 j) j)
        (loopj (+ j 1))))

    ;; Fill DP table
    (let outer ((i 1))
      (when (<= i m)
        (let inner ((j 1))
          (when (<= j n)
            (let* ((cost (if (char=? (string-ref s1 (- i 1))
                                     (string-ref s2 (- j 1)))
                             0 1))
                   (del (+ (vector-ref dp (idx (- i 1) j)) 1))
                   (ins (+ (vector-ref dp (idx i (- j 1))) 1))
                   (sub (+ (vector-ref dp (idx (- i 1) (- j 1))) cost)))
              (vector-set! dp (idx i j) (min del (min ins sub))))
            (inner (+ j 1))))
        (outer (+ i 1))))
    (vector-ref dp (idx m n))))

;; Read two lines from stdin and print distance
(let ((s1 (read-line))
      (s2 (read-line)))
  (display (levenshtein s1 s2))
  (newline))

