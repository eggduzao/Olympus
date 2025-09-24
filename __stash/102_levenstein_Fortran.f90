! Levenshtein edit distance in Fortran (Fortran 90+)
! Reads two lines from stdin and prints the distance.

program levenshtein
  implicit none
  character(len=:), allocatable :: s1, s2
  character(len=1024) :: buf
  integer :: m, n, i, j, cost
  integer, allocatable :: dp(:,:)

  ! Read first string (line)
  if (read_line(buf) == 0) then
     s1 = trim(buf)
  else
     stop 1
  end if

  ! Read second string (line)
  if (read_line(buf) == 0) then
     s2 = trim(buf)
  else
     stop 1
  end if

  m = len_trim(s1)
  n = len_trim(s2)

  allocate(dp(0:m,0:n))

  ! Base cases
  do i = 0, m
     dp(i,0) = i
  end do
  do j = 0, n
     dp(0,j) = j
  end do

  ! Fill DP table
  do i = 1, m
     do j = 1, n
        if (s1(i:i) == s2(j:j)) then
           cost = 0
        else
          cost = 1
        end if
        dp(i,j) = min( dp(i-1,j) + 1, min( dp(i,j-1) + 1, dp(i-1,j-1) + cost ) )
     end do
  end do

  print *, dp(m,n)

  contains

  ! Reads a line into buf; returns 0 on success, non-zero on EOF/error
  integer function read_line(b)
    character(len=*), intent(inout) :: b
    integer :: ios
    read_line = 1
    b = ''                ! clear
    read(unit=*, fmt='(A)', iostat=ios) b
    if (ios == 0) then
       read_line = 0
    end if
  end function read_line

end program levenshtein

