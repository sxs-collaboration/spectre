// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/MathFunctions/Sinusoid.hpp"

#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace MathFunctions {
template <typename Fr>
Sinusoid<1, Fr>::Sinusoid(const double amplitude, const double wavenumber,
                          const double phase) noexcept
    : amplitude_(amplitude), wavenumber_(wavenumber), phase_(phase) {}
template <typename Fr>
double Sinusoid<1, Fr>::operator()(const double& x) const noexcept {
  return apply_call_operator(x);
}

template <typename Fr>
DataVector Sinusoid<1, Fr>::operator()(const DataVector& x) const noexcept {
  return apply_call_operator(x);
}

template <typename Fr>
double Sinusoid<1, Fr>::first_deriv(const double& x) const noexcept {
  return apply_first_deriv(x);
}

template <typename Fr>
DataVector Sinusoid<1, Fr>::first_deriv(const DataVector& x) const noexcept {
  return apply_first_deriv(x);
}

template <typename Fr>
double Sinusoid<1, Fr>::second_deriv(const double& x) const noexcept {
  return apply_second_deriv(x);
}

template <typename Fr>
DataVector Sinusoid<1, Fr>::second_deriv(const DataVector& x) const noexcept {
  return apply_second_deriv(x);
}

template <typename Fr>
double Sinusoid<1, Fr>::third_deriv(const double& x) const noexcept {
  return apply_third_deriv(x);
}

template <typename Fr>
DataVector Sinusoid<1, Fr>::third_deriv(const DataVector& x) const noexcept {
  return apply_third_deriv(x);
}

template <typename Fr>
template <typename T>
T Sinusoid<1, Fr>::apply_call_operator(const T& x) const noexcept {
  return amplitude_ * sin(wavenumber_ * x + phase_);
}

template <typename Fr>
template <typename T>
T Sinusoid<1, Fr>::apply_first_deriv(const T& x) const noexcept {
  return wavenumber_ * amplitude_ * cos(wavenumber_ * x + phase_);
}

template <typename Fr>
template <typename T>
T Sinusoid<1, Fr>::apply_second_deriv(const T& x) const noexcept {
  return -amplitude_ * square(wavenumber_) * sin(wavenumber_ * x + phase_);
}

template <typename Fr>
template <typename T>
T Sinusoid<1, Fr>::apply_third_deriv(const T& x) const noexcept {
  return -amplitude_ * cube(wavenumber_) * cos(wavenumber_ * x + phase_);
}

template <typename Fr>
void Sinusoid<1, Fr>::pup(PUP::er& p) {
  MathFunction<1, Fr>::pup(p);
  p | amplitude_;
  p | wavenumber_;
  p | phase_;
}
}  // namespace MathFunctions

template MathFunctions::Sinusoid<1, Frame::Grid>::Sinusoid(
    const double amplitude, const double wavenumber,
    const double phase) noexcept;
template MathFunctions::Sinusoid<1, Frame::Inertial>::Sinusoid(
    const double amplitude, const double wavenumber,
    const double phase) noexcept;
template void MathFunctions::Sinusoid<1, Frame::Grid>::pup(PUP::er& p);
template void MathFunctions::Sinusoid<1, Frame::Inertial>::pup(PUP::er& p);

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                  \
  template DTYPE(data) MathFunctions::Sinusoid<1, FRAME(data)>::operator()(   \
      const DTYPE(data) & x) const noexcept;                                  \
  template DTYPE(data) MathFunctions::Sinusoid<1, FRAME(data)>::first_deriv(  \
      const DTYPE(data) & x) const noexcept;                                  \
  template DTYPE(data) MathFunctions::Sinusoid<1, FRAME(data)>::second_deriv( \
      const DTYPE(data) & x) const noexcept;                                  \
  template DTYPE(data) MathFunctions::Sinusoid<1, FRAME(data)>::third_deriv(  \
      const DTYPE(data) & x) const noexcept;                                  \
  template DTYPE(data)                                                        \
      MathFunctions::Sinusoid<1, FRAME(data)>::apply_call_operator(           \
          const DTYPE(data) & x) const noexcept;                              \
  template DTYPE(data)                                                        \
      MathFunctions::Sinusoid<1, FRAME(data)>::apply_first_deriv(             \
          const DTYPE(data) & x) const noexcept;                              \
  template DTYPE(data)                                                        \
      MathFunctions::Sinusoid<1, FRAME(data)>::apply_second_deriv(            \
          const DTYPE(data) & x) const noexcept;                              \
  template DTYPE(data)                                                        \
      MathFunctions::Sinusoid<1, FRAME(data)>::apply_third_deriv(             \
          const DTYPE(data) & x) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (Frame::Grid, Frame::Inertial),
                        (double, DataVector))
#undef DTYPE
#undef FRAME
#undef INSTANTIATE

