(defun levenshtein-distance (s1 s2)
  (let* ((m (length s1))
         (n (length s2))
         (dp (make-array (list (1+ m) (1+ n)) :initial-element 0)))
    ;; Initialize matrix
    (dotimes (i (1+ m))
      (setf (aref dp i 0) i))
    (dotimes (j (1+ n))
      (setf (aref dp 0 j) j))
    ;; Fill matrix
    (dotimes (i m)
      (dotimes (j n)
        (let ((cost (if (char= (aref s1 i) (aref s2 j)) 0 1))
          (setf (aref dp (1+ i) (1+ j))
                (min (+ (aref dp i (1+ j)) 1)      ;; Deletion
                     (+ (aref dp (1+ i) j) 1)      ;; Insertion
                     (+ (aref dp i j) cost)))))))  ;; Substitution
    (aref dp m n)))

;; Sample usage
(format t "~A~%" (levenshtein-distance "kitten" "sitting"))

