function quad16(f, a, b) result(integral)
    implicit none
    double precision :: a, b, integral
    double precision, external :: f
    integer, parameter :: n16 = 16
    double precision, dimension(n16) :: x16, w16 

    integer :: i
    double precision :: r, m, c

    ! Define the abscissas and weights for 16-point Gauss quadrature
    data x16 /-0.0950125098376374D0, 0.0950125098376374D0, -0.2816035507792589D0, 0.2816035507792589D0, &  
              -0.4580167776572274D0, 0.4580167776572274D0, -0.6178762444026438D0, 0.6178762444026438D0, &  
              -0.7554044083550030D0, 0.7554044083550030D0, -0.8656312023878318D0, 0.8656312023878318D0, &  
              -0.9445750230732326D0, 0.9445750230732326D0, -0.9894009349916499D0, 0.9894009349916499D0  /  
              
    data w16 /0.1894506104550685D0, 0.1894506104550685D0, 0.1826034150449236D0, 0.1826034150449236D0, &
              0.1691565193950025D0, 0.1691565193950025D0, 0.1495959888165767D0, 0.1495959888165767D0, &
              0.1246289712555339D0, 0.1246289712555339D0, 0.0951585116824928D0, 0.0951585116824928D0, &
              0.0622535239386479D0, 0.0622535239386479D0, 0.0271524594117541D0, 0.0271524594117541D0  /

    ! Compute the results using 16-point Gauss quadrature
    r = 0.D0
    m = (b-a)/2.D0
    c = (b+a)/2.D0
    integral = 0.0d0
    do i = 1, n16
        integral =  integral + w16(i) * f(m*x16(i) + c)
    end do
    integral = integral*m
end function
