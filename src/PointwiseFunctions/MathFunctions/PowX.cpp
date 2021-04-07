// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/MathFunctions/PowX.hpp"

#include "DataStructures/DataVector.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace MathFunctions {

template <typename Fr>
PowX<1, Fr>::PowX(const int power) noexcept : power_(power) {}

template <typename Fr>
double PowX<1, Fr>::operator()(const double& x) const noexcept {
  return apply_call_operator(x);
}

template <typename Fr>
DataVector PowX<1, Fr>::operator()(const DataVector& x) const noexcept {
  return apply_call_operator(x);
}

template <typename Fr>
double PowX<1, Fr>::first_deriv(const double& x) const noexcept {
  return apply_first_deriv(x);
}

template <typename Fr>
DataVector PowX<1, Fr>::first_deriv(const DataVector& x) const noexcept {
  return apply_first_deriv(x);
}

template <typename Fr>
double PowX<1, Fr>::second_deriv(const double& x) const noexcept {
  return apply_second_deriv(x);
}

template <typename Fr>
DataVector PowX<1, Fr>::second_deriv(const DataVector& x) const noexcept {
  return apply_second_deriv(x);
}

template <typename Fr>
double PowX<1, Fr>::third_deriv(const double& x) const noexcept {
  return apply_third_deriv(x);
}

template <typename Fr>
DataVector PowX<1, Fr>::third_deriv(const DataVector& x) const noexcept {
  return apply_third_deriv(x);
}

template <typename Fr>
template <typename T>
T PowX<1, Fr>::apply_call_operator(const T& x) const noexcept {
  return pow(x, power_);
}

template <typename Fr>
template <typename T>
T PowX<1, Fr>::apply_first_deriv(const T& x) const noexcept {
  return 0 == power_ ? make_with_value<T>(x, 0.0) : power_ * pow(x, power_ - 1);
}

template <typename Fr>
template <typename T>
T PowX<1, Fr>::apply_second_deriv(const T& x) const noexcept {
  return 0 == power_ or 1 == power_
             ? make_with_value<T>(x, 0.0)
             : power_ * (power_ - 1) * pow(x, power_ - 2);
}

template <typename Fr>
template <typename T>
T PowX<1, Fr>::apply_third_deriv(const T& x) const noexcept {
  return 0 == power_ or 1 == power_ or 2 == power_
             ? make_with_value<T>(x, 0.0)
             : power_ * (power_ - 1) * (power_ - 2) * pow(x, power_ - 3);
}

template <typename Fr>
void PowX<1, Fr>::pup(PUP::er& p) {
  MathFunction<1, Fr>::pup(p);
  p | power_;
}

template MathFunctions::PowX<1, Frame::Grid>::PowX(const int power) noexcept;
template MathFunctions::PowX<1, Frame::Inertial>::PowX(
    const int power) noexcept;
template void MathFunctions::PowX<1, Frame::Grid>::pup(PUP::er& p);
template void MathFunctions::PowX<1, Frame::Inertial>::pup(PUP::er& p);

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                   \
  template DTYPE(data)                                                         \
      MathFunctions::PowX<1, FRAME(data)>::operator()(const DTYPE(data) & x)   \
          const noexcept;                                                      \
  template DTYPE(data)                                                         \
      MathFunctions::PowX<1, FRAME(data)>::first_deriv(const DTYPE(data) & x)  \
          const noexcept;                                                      \
  template DTYPE(data)                                                         \
      MathFunctions::PowX<1, FRAME(data)>::second_deriv(const DTYPE(data) & x) \
          const noexcept;                                                      \
  template DTYPE(data)                                                         \
      MathFunctions::PowX<1, FRAME(data)>::third_deriv(const DTYPE(data) & x)  \
          const noexcept;                                                      \
  template DTYPE(data)                                                         \
      MathFunctions::PowX<1, FRAME(data)>::apply_call_operator(                \
          const DTYPE(data) & x) const noexcept;                               \
  template DTYPE(data) MathFunctions::PowX<1, FRAME(data)>::apply_first_deriv( \
      const DTYPE(data) & x) const noexcept;                                   \
  template DTYPE(data)                                                         \
      MathFunctions::PowX<1, FRAME(data)>::apply_second_deriv(                 \
          const DTYPE(data) & x) const noexcept;                               \
  template DTYPE(data) MathFunctions::PowX<1, FRAME(data)>::apply_third_deriv( \
      const DTYPE(data) & x) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (Frame::Grid, Frame::Inertial),
                        (double, DataVector))
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
}  // namespace MathFunctions
