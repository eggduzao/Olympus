! quicksort.f90 — In-place quicksort on an integer array (reads from STDIN)
! Usage:
!   gfortran -O2 -std=f2008 -Wall -Wextra -o quicksort quicksort.f90
!   ./quicksort < input.txt
!
! Input format:
!   One integer per line (arbitrary length; we grow the buffer dynamically).
!
! Output:
!   The same integers, sorted ascending, one per line.

program quicksort_main
  implicit none
  integer, allocatable :: a(:)
  integer :: n, cap, val, ios, i

  ! ---- 1) Read all integers from STDIN into a dynamically grown array ----
  cap = 0; n = 0
  allocate(a(0))   ! start empty; we’ll grow as needed

  do
    read(*, *, iostat=ios) val
    if (ios /= 0) exit  ! EOF or read error ends input loop

    if (n == cap) then
      if (cap == 0) then
        cap = 16
      else
        cap = cap * 2
      end if
      call grow(a, n, cap)
    end if

    n = n + 1
    a(n) = val
  end do

  if (n == 0) stop  ! nothing to sort

  ! ---- 2) In-place quicksort (Hoare-style partition, median-of-three pivot) ----
  call quicksort(a, 1, n)

  ! ---- 3) Emit result ----
  do i = 1, n
    write(*,'(I0)') a(i)
  end do

contains

  subroutine grow(arr, n, newcap)
    ! Reallocate ARR to size NEWCAP preserving the first N elements.
    integer, allocatable, intent(inout) :: arr(:)
    integer, intent(in) :: n, newcap
    integer, allocatable :: tmp(:)

    if (newcap < n) stop "grow(): new capacity smaller than current length"

    allocate(tmp(newcap))
    if (n > 0) tmp(1:n) = arr(1:n)
    call move_alloc(tmp, arr)   ! O(1) ‘steal’ allocation, avoid copy-back
  end subroutine grow

  recursive subroutine quicksort(arr, lo, hi)
    implicit none
    integer, intent(inout) :: arr(:)
    integer, intent(in)    :: lo, hi
    integer :: i, j, mid, pivot, t

    if (lo >= hi) return

    mid   = lo + (hi - lo) / 2
    pivot = median3(arr(lo), arr(mid), arr(hi))

    ! Hoare partition (stable-enough for duplicates, minimizes swaps)
    i = lo
    j = hi
    do
      do while (arr(i) < pivot); i = i + 1; end do
      do while (arr(j) > pivot); j = j - 1; end do

      if (i <= j) then
        t = arr(i); arr(i) = arr(j); arr(j) = t
        i = i + 1; j = j - 1
      end if

      if (i > j) exit
    end do

    if (lo < j) call quicksort(arr, lo, j)
    if (i  < hi) call quicksort(arr, i,  hi)
  end subroutine quicksort

  integer function median3(a, b, c)
    ! Return the median of three integers (branching only; no sorting needed).
    implicit none
    integer, intent(in) :: a, b, c
    if ((a >= b .and. a <= c) .or. (a <= b .and. a >= c)) then
      median3 = a
    else if ((b >= a .and. b <= c) .or. (b <= a .and. b >= c)) then
      median3 = b
    else
      median3 = c
    end if
  end function median3

end program quicksort_main

