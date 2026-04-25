!===============================================================================
! BFS Shortest Path on an Unweighted Graph (Fortran 90, single file)
!-------------------------------------------------------------------------------
! INPUT (from stdin; whitespace/tab separated):
!   • Edge line : U  V1 [V2 ...]   -> add UNDIRECTED edges U—V1, U—V2, ...
!   • Node line : U                -> ensure node exists (even if isolated)
!   • Query line: #  SRC  DST     -> request shortest path from SRC to DST
!
! Example:
!   A   B   F
!   B   A   C
!   C   B   D
!   D   C   E
!   E   D   F
!   F   A   E
!   #   A   E
!
! RUN:
!   $ ./a.out < graph.tsv
!
! OUTPUT:
!   For each "# SRC DST" query, prints either "A -> B -> ... -> E"
!   or "No path found from SRC to DST".
!
! NOTES:
!   • Uses fixed-length labels (MAXLEN) and dynamic arrays.
!   • Builds CSR adjacency (offsets + neighbors) for O(|V|+|E|) BFS.
!   • Name→index mapping uses a linear scan for simplicity (fine for demos).
!     Replace with a hash map if you need to scale to millions of nodes.
!===============================================================================
program bfs_shortest_path
  implicit none

  integer, parameter :: MAXLEN = 64
  integer, parameter :: INIT_CAP_NODES = 64
  integer, parameter :: INIT_CAP_EDGES = 256
  integer, parameter :: LINE_MAX = 4096

  ! Node table
  character(len=MAXLEN), allocatable :: nodes(:)
  integer :: n_nodes = 0, cap_nodes = 0

  ! Edge list (temporary, store both directions)
  integer, allocatable :: eu(:), ev(:)
  integer :: n_edges = 0, cap_edges = 0

  ! Queries
  character(len=MAXLEN), allocatable :: qsrc(:), qdst(:)
  integer :: n_queries = 0, cap_queries = 0

  ! CSR adjacency
  integer, allocatable :: offs(:), adj(:)

  ! Scratch
  character(len=LINE_MAX) :: line
  logical :: eof
  integer :: ios, i

  call init_nodes()
  call init_edges()
  call init_queries()

  !----------------------------- Read & Parse ----------------------------------
  eof = .false.
  do
     read(*,'(A)', iostat=ios) line
     if (ios < 0) then
        exit   ! EOF
     else if (ios > 0) then
        cycle  ! Skip read errors
     end if

     if (is_blank(line)) cycle

     call process_line(trim(line))
  end do

  !--------------------------- Build adjacency (CSR) ---------------------------
  if (n_nodes == 0) then
     ! Nothing to do
     stop
  end if

  call build_csr(n_nodes, n_edges, eu, ev, offs, adj)

  !---------------------------- Answer queries ---------------------------------
  do i = 1, n_queries
     call answer_query(trim(qsrc(i)), trim(qdst(i)), n_nodes, offs, adj, nodes)
  end do

contains

  !------------------------------- Utilities -----------------------------------

  pure logical function is_ws(ch)
    character, intent(in) :: ch
    is_ws = (ch == ' ' .or. ch == achar(9) .or. ch == achar(10) .or. ch == achar(13))
  end function is_ws

  pure logical function is_blank(s)
    character(len=*), intent(in) :: s
    integer :: k, n
    n = len_trim(s)
    if (n == 0) then
       is_blank = .true.; return
    end if
    do k = 1, n
       if (.not. is_ws(s(k:k))) then
          is_blank = .false.; return
       end if
    end do
    is_blank = .true.
  end function is_blank

  subroutine split_ws(s, tokens, count)
    ! Split a string on whitespace into allocatable tokens(:)
    character(len=*), intent(in) :: s
    character(len=MAXLEN), allocatable, intent(out) :: tokens(:)
    integer, intent(out) :: count
    integer :: n, i, start, cap
    character(len=MAXLEN) :: tok

    n = len_trim(s)
    count = 0
    cap = 0
    allocate(tokens(0))
    i = 1
    do while (i <= n)
       ! skip whitespace
       do while (i <= n .and. is_ws(s(i:i)))
          i = i + 1
       end do
       if (i > n) exit
       start = i
       do while (i <= n .and. .not. is_ws(s(i:i)))
          i = i + 1
       end do
       tok = adjustl(s(start:min(i-1, start+MAXLEN-1)))
       call push_token(tokens, cap, count, tok)
    end do
  end subroutine split_ws

  subroutine push_token(tokens, cap, count, tok)
    character(len=MAXLEN), allocatable, intent(inout) :: tokens(:)
    integer, intent(inout) :: cap, count
    character(len=MAXLEN), intent(in) :: tok
    character(len=MAXLEN), allocatable :: tmp(:)
    if (count == cap) then
       cap = max(1, merge(2*cap, 1, cap>0))
       allocate(tmp(count))
       if (count > 0) tmp = tokens
       deallocate(tokens)
       allocate(tokens(cap))
       if (count > 0) tokens(1:count) = tmp
       if (allocated(tmp)) deallocate(tmp)
    end if
    count = count + 1
    tokens(count) = tok
  end subroutine push_token

  !---------------------------- Node dictionary --------------------------------

  subroutine init_nodes()
    cap_nodes = INIT_CAP_NODES
    allocate(nodes(cap_nodes))
    n_nodes = 0
  end subroutine init_nodes

  subroutine ensure_node_capacity()
    character(len=MAXLEN), allocatable :: tmp(:)
    if (n_nodes < cap_nodes) return
    cap_nodes = max(1, 2*cap_nodes)
    allocate(tmp(n_nodes))
    if (n_nodes > 0) tmp = nodes(1:n_nodes)
    deallocate(nodes)
    allocate(nodes(cap_nodes))
    if (n_nodes > 0) nodes(1:n_nodes) = tmp
    if (allocated(tmp)) deallocate(tmp)
  end subroutine ensure_node_capacity

  integer function get_or_add_node(name) result(idx)
    character(len=*), intent(in) :: name
    integer :: k
    character(len=MAXLEN) :: tname
    tname = adjustl(name(1:min(len_trim(name), MAXLEN)))
    ! linear search
    do k = 1, n_nodes
       if (trim(nodes(k)) == trim(tname)) then
          idx = k; return
       end if
    end do
    ! add new
    call ensure_node_capacity()
    n_nodes = n_nodes + 1
    nodes(n_nodes) = tname
    idx = n_nodes
  end function get_or_add_node

  integer function find_node(name) result(idx)
    character(len=*), intent(in) :: name
    integer :: k
    character(len=MAXLEN) :: tname
    tname = trim(adjustl(name(1:min(len_trim(name), MAXLEN))))
    do k = 1, n_nodes
       if (trim(nodes(k)) == tname) then
          idx = k; return
       end if
    end do
    idx = -1
  end function find_node

  !---------------------------- Edge collection --------------------------------

  subroutine init_edges()
    cap_edges = INIT_CAP_EDGES
    allocate(eu(cap_edges))
    allocate(ev(cap_edges))
    n_edges = 0
  end subroutine init_edges

  subroutine ensure_edge_capacity(extra)
    integer, intent(in) :: extra
    integer, allocatable :: tu(:), tv(:)
    integer :: need
    need = n_edges + extra
    if (need <= cap_edges) return
    do
       cap_edges = 2*cap_edges
       if (cap_edges >= need) exit
    end do
    allocate(tu(n_edges)); allocate(tv(n_edges))
    if (n_edges > 0) then
       tu = eu(1:n_edges); tv = ev(1:n_edges)
    end if
    deallocate(eu); deallocate(ev)
    allocate(eu(cap_edges)); allocate(ev(cap_edges))
    if (n_edges > 0) then
       eu(1:n_edges) = tu; ev(1:n_edges) = tv
    end if
    if (allocated(tu)) deallocate(tu)
    if (allocated(tv)) deallocate(tv)
  end subroutine ensure_edge_capacity

  subroutine add_undirected(u, v)
    integer, intent(in) :: u, v
    if (u == v) return
    call ensure_edge_capacity(2)
    n_edges = n_edges + 1; eu(n_edges) = u; ev(n_edges) = v
    n_edges = n_edges + 1; eu(n_edges) = v; ev(n_edges) = u
  end subroutine add_undirected

  !------------------------------- Queries -------------------------------------

  subroutine init_queries()
    cap_queries = 8
    allocate(qsrc(cap_queries))
    allocate(qdst(cap_queries))
    n_queries = 0
  end subroutine init_queries

  subroutine ensure_query_capacity()
    character(len=MAXLEN), allocatable :: ts(:), td(:)
    if (n_queries < cap_queries) return
    cap_queries = 2*cap_queries
    allocate(ts(n_queries)); allocate(td(n_queries))
    if (n_queries > 0) then
       ts = qsrc(1:n_queries); td = qdst(1:n_queries)
    end if
    deallocate(qsrc); deallocate(qdst)
    allocate(qsrc(cap_queries)); allocate(qdst(cap_queries))
    if (n_queries > 0) then
       qsrc(1:n_queries) = ts; qdst(1:n_queries) = td
    end if
    if (allocated(ts)) deallocate(ts)
    if (allocated(td)) deallocate(td)
  end subroutine ensure_query_capacity

  subroutine add_query(s, d)
    character(len=*), intent(in) :: s, d
    integer :: is, id
    is = get_or_add_node(s)  ! ensure nodes exist
    id = get_or_add_node(d)
    call ensure_query_capacity()
    n_queries = n_queries + 1
    qsrc(n_queries) = trim(s)
    qdst(n_queries) = trim(d)
  end subroutine add_query

  !------------------------------ Line parser ----------------------------------

  subroutine process_line(l)
    character(len=*), intent(in) :: l
    character(len=MAXLEN), allocatable :: t(:)
    integer :: m, i, u, v

    call split_ws(l, t, m)
    if (m <= 0) return

    if (trim(t(1)) == '#') then
       if (m >= 3) call add_query(trim(t(2)), trim(t(3)))
       return
    end if

    ! Ensure base node exists
    u = get_or_add_node(trim(t(1)))

    if (m == 1) then
       return  ! isolated node line
    end if

    ! Add undirected edges u—t(i)
    do i = 2, m
       v = get_or_add_node(trim(t(i)))
       call add_undirected(u, v)
    end do
  end subroutine process_line

  !------------------------------- CSR build -----------------------------------

  subroutine build_csr(nv, ne, eu, ev, offs, adj)
    integer, intent(in) :: nv, ne
    integer, intent(in) :: eu(:), ev(:)
    integer, allocatable, intent(out) :: offs(:), adj(:)
    integer, allocatable :: deg(:), nxt(:)
    integer :: i, u, v, total

    allocate(deg(nv)); deg = 0
    do i = 1, ne
       u = eu(i)
       if (u >= 1 .and. u <= nv) deg(u) = deg(u) + 1
    end do

    allocate(offs(nv+1))
    offs(1) = 1
    do i = 2, nv+1
       offs(i) = offs(i-1) + deg(i-1)
    end do
    total = offs(nv+1) - 1
    allocate(adj(total))

    allocate(nxt(nv))
    do i = 1, nv
       nxt(i) = offs(i)
    end do

    do i = 1, ne
       u = eu(i); v = ev(i)
       if (u<1 .or. u>nv) cycle
       adj(nxt(u)) = v
       nxt(u) = nxt(u) + 1
    end do

    if (allocated(deg)) deallocate(deg)
    if (allocated(nxt)) deallocate(nxt)
  end subroutine build_csr

  !------------------------------- BFS & print ---------------------------------

  subroutine answer_query(slabel, dlabel, nv, offs, adj, nodes)
    character(len=*), intent(in) :: slabel, dlabel
    integer, intent(in) :: nv
    integer, intent(in) :: offs(:), adj(:)
    character(len=MAXLEN), intent(in) :: nodes(:)

    integer, allocatable :: parent(:), queue(:)
    logical, allocatable :: vis(:)
    integer :: s, d, head, tail, cur, i, p, qsz
    integer :: startE, endE, nb
    character(len=:), allocatable :: out
    character(len=MAXLEN), allocatable :: rev(:)
    integer :: k, plen

    s = find_node(slabel)
    d = find_node(dlabel)

    if (s == -1 .or. d == -1) then
       write(*,'(A,1X,A,1X,A)') 'No path found from', trim(slabel), 'to '//trim(dlabel)
       return
    end if

    if (s == d) then
       write(*,'(A)') trim(nodes(s))
       return
    end if

    allocate(parent(nv)); parent = -1
    allocate(vis(nv)); vis = .false.
    allocate(queue(nv)); head = 1; tail = 0

    ! enqueue s
    tail = tail + 1; queue(tail) = s; vis(s) = .true.

    do while (head <= tail .and. .not. vis(d))
       cur = queue(head); head = head + 1
       startE = offs(cur)
       endE   = offs(cur+1) - 1
       do i = startE, endE
          nb = adj(i)
          if (.not. vis(nb)) then
             vis(nb) = .true.
             parent(nb) = cur
             if (nb == d) exit
             tail = tail + 1
             queue(tail) = nb
          end if
       end do
    end do

    if (.not. vis(d)) then
       write(*,'(A,1X,A,1X,A)') 'No path found from', trim(slabel), 'to '//trim(dlabel)
       deallocate(parent, vis, queue)
       return
    end if

    ! reconstruct path (reverse, then print forward)
    allocate(rev(nv)); plen = 0
    p = d
    do while (p /= -1)
       plen = plen + 1
        rev(plen) = trim(nodes(p))
       p = parent(p)
    end do

    ! print reverse order with " -> "
    out = ''
    do k = plen, 1, -1
       if (k /= plen) then
          out = out//' -> '//trim(rev(k))
       else
          out = out//trim(rev(k))
       end if
    end do
    write(*,'(A)') out

    deallocate(parent, vis, queue, rev)
  end subroutine answer_query

end program bfs_shortest_path

