function integrate(f, a, b, n) result(integral)
  implicit none
  
  integer, intent(in) :: n
  real, intent(in) :: a, b
  real :: dx, x, integral
  double precision, external :: f

  ! Calculate the step size
  dx = (b - a) / n
  
  ! Perform the integration using the trapezoidal rule
  integral = 0.5 * (f(a) + f(b))
  do x = a + dx, b - dx, dx
    integral = integral + f(x)
  end do
  integral = integral * dx
  
end function integrate