
! dgaus8 integration routine
function dgaus8(f, a, b) result(integral)
    implicit none
    double precision :: a, b, integral
    double precision, external :: f
    double precision, dimension(8) :: xw 
    double precision, dimension(1:8) :: x
    integer :: i
    
    ! define the abscissas and weights
    xw = (/0.1012285362903763d0, 0.2223810344533745d0, 0.3137066458778873d0, 0.3626837833783619d0, &
           0.3626837833783619d0, 0.3137066458778873d0, 0.2223810344533745d0, 0.1012285362903763d0/)
    
    ! transform the abscissas to the interval [a,b]
    x = (b+a)/2.0d0 + (b-a)/2.0d0*xw
    
    ! evaluate the function at the abscissas and compute the integral
    integral = 0.0d0
    do i=1,8
        integral = integral + xw(i)*f(x(i))
    end do
    integral = (b-a)*integral/2.0d0
    
end function


