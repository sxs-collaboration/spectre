// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/MathFunctions/Constant.hpp"

#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace MathFunctions {

template <typename Fr>
Constant<1, Fr>::Constant(const double value) noexcept : value_(value) {}

template <typename Fr>
double Constant<1, Fr>::operator()(const double& x) const noexcept {
  return apply_call_operator(x);
}

template <typename Fr>
DataVector Constant<1, Fr>::operator()(const DataVector& x) const noexcept {
  return apply_call_operator(x);
}

template <typename Fr>
double Constant<1, Fr>::first_deriv(const double& x) const noexcept {
  return apply_first_deriv(x);
}

template <typename Fr>
DataVector Constant<1, Fr>::first_deriv(const DataVector& x) const noexcept {
  return apply_first_deriv(x);
}

template <typename Fr>
double Constant<1, Fr>::second_deriv(const double& x) const noexcept {
  return apply_second_deriv(x);
}

template <typename Fr>
DataVector Constant<1, Fr>::second_deriv(const DataVector& x) const noexcept {
  return apply_second_deriv(x);
}

template <typename Fr>
double Constant<1, Fr>::third_deriv(const double& x) const noexcept {
  return apply_third_deriv(x);
}

template <typename Fr>
DataVector Constant<1, Fr>::third_deriv(const DataVector& x) const noexcept {
  return apply_third_deriv(x);
}

template <typename Fr>
template <typename T>
T Constant<1, Fr>::apply_call_operator(const T& x) const noexcept {
  return make_with_value<T>(x, value_);
}

template <typename Fr>
template <typename T>
T Constant<1, Fr>::apply_first_deriv(const T& x) const noexcept {
  return make_with_value<T>(x, 0.0);
  ;
}

template <typename Fr>
template <typename T>
T Constant<1, Fr>::apply_second_deriv(const T& x) const noexcept {
  return make_with_value<T>(x, 0.0);
}

template <typename Fr>
template <typename T>
T Constant<1, Fr>::apply_third_deriv(const T& x) const noexcept {
  return make_with_value<T>(x, 0.0);
}

template <typename Fr>
void Constant<1, Fr>::pup(PUP::er& p) {
  MathFunction<1, Fr>::pup(p);
  p | value_;
}

template <size_t VolumeDim, typename Fr>
Constant<VolumeDim, Fr>::Constant(const double value) noexcept
    : value_(value) {}

template <size_t VolumeDim, typename Fr>
template <typename T>
Scalar<T> Constant<VolumeDim, Fr>::apply_call_operator(
    const tnsr::I<T, VolumeDim, Fr>& x) const noexcept {
  return make_with_value<Scalar<T>>(x, value_);
}

template <size_t VolumeDim, typename Fr>
template <typename T>
tnsr::i<T, VolumeDim, Fr> Constant<VolumeDim, Fr>::apply_first_deriv(
    const tnsr::I<T, VolumeDim, Fr>& x) const noexcept {
  return make_with_value<tnsr::i<T, VolumeDim, Fr>>(x, 0.0);
}

template <size_t VolumeDim, typename Fr>
template <typename T>
tnsr::ii<T, VolumeDim, Fr> Constant<VolumeDim, Fr>::apply_second_deriv(
    const tnsr::I<T, VolumeDim, Fr>& x) const noexcept {
  return make_with_value<tnsr::ii<T, VolumeDim, Fr>>(x, 0.0);
}

template <size_t VolumeDim, typename Fr>
template <typename T>
tnsr::iii<T, VolumeDim, Fr> Constant<VolumeDim, Fr>::apply_third_deriv(
    const tnsr::I<T, VolumeDim, Fr>& x) const noexcept {
  return make_with_value<tnsr::iii<T, VolumeDim, Fr>>(x, 0.0);
}

template <size_t VolumeDim, typename Fr>
Scalar<double> Constant<VolumeDim, Fr>::operator()(
    const tnsr::I<double, VolumeDim, Fr>& x) const noexcept {
  return apply_call_operator(x);
}
template <size_t VolumeDim, typename Fr>
Scalar<DataVector> Constant<VolumeDim, Fr>::operator()(
    const tnsr::I<DataVector, VolumeDim, Fr>& x) const noexcept {
  return apply_call_operator(x);
}

template <size_t VolumeDim, typename Fr>
tnsr::i<double, VolumeDim, Fr> Constant<VolumeDim, Fr>::first_deriv(
    const tnsr::I<double, VolumeDim, Fr>& x) const noexcept {
  return apply_first_deriv(x);
}
template <size_t VolumeDim, typename Fr>
tnsr::i<DataVector, VolumeDim, Fr> Constant<VolumeDim, Fr>::first_deriv(
    const tnsr::I<DataVector, VolumeDim, Fr>& x) const noexcept {
  return apply_first_deriv(x);
}

template <size_t VolumeDim, typename Fr>
tnsr::ii<double, VolumeDim, Fr> Constant<VolumeDim, Fr>::second_deriv(
    const tnsr::I<double, VolumeDim, Fr>& x) const noexcept {
  return apply_second_deriv(x);
}
template <size_t VolumeDim, typename Fr>
tnsr::ii<DataVector, VolumeDim, Fr> Constant<VolumeDim, Fr>::second_deriv(
    const tnsr::I<DataVector, VolumeDim, Fr>& x) const noexcept {
  return apply_second_deriv(x);
}

template <size_t VolumeDim, typename Fr>
tnsr::iii<double, VolumeDim, Fr> Constant<VolumeDim, Fr>::third_deriv(
    const tnsr::I<double, VolumeDim, Fr>& x) const noexcept {
  return apply_third_deriv(x);
}
template <size_t VolumeDim, typename Fr>
tnsr::iii<DataVector, VolumeDim, Fr> Constant<VolumeDim, Fr>::third_deriv(
    const tnsr::I<DataVector, VolumeDim, Fr>& x) const noexcept {
  return apply_third_deriv(x);
}

template <size_t VolumeDim, typename Fr>
void Constant<VolumeDim, Fr>::pup(PUP::er& p) {
  MathFunction<VolumeDim, Fr>::pup(p);
  p | value_;
}
}  // namespace MathFunctions

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATE(_, data)                                          \
  template MathFunctions::Constant<DIM(data), FRAME(data)>::Constant( \
      const double value) noexcept;                                   \
  template void MathFunctions::Constant<DIM(data), FRAME(data)>::pup( \
      PUP::er& p);

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3), (Frame::Grid, Frame::Inertial))
#undef DIM
#undef FRAME
#undef INSTANTIATE

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                 \
  template Scalar<DTYPE(data)>                                               \
  MathFunctions::Constant<DIM(data), FRAME(data)>::operator()(               \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& x) const noexcept; \
  template tnsr::i<DTYPE(data), DIM(data), FRAME(data)>                      \
  MathFunctions::Constant<DIM(data), FRAME(data)>::first_deriv(              \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& x) const noexcept; \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>                     \
  MathFunctions::Constant<DIM(data), FRAME(data)>::second_deriv(             \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& x) const noexcept; \
  template tnsr::iii<DTYPE(data), DIM(data), FRAME(data)>                    \
  MathFunctions::Constant<DIM(data), FRAME(data)>::third_deriv(              \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& x) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3), (Frame::Grid, Frame::Inertial),
                        (double, DataVector))
#undef FRAME
#undef DIM
#undef DTYPE
#undef INSTANTIATE

template MathFunctions::Constant<1, Frame::Grid>::Constant(
    const double value) noexcept;
template MathFunctions::Constant<1, Frame::Inertial>::Constant(
    const double value) noexcept;
template void MathFunctions::Constant<1, Frame::Grid>::pup(PUP::er& p);
template void MathFunctions::Constant<1, Frame::Inertial>::pup(PUP::er& p);

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                  \
  template DTYPE(data) MathFunctions::Constant<1, FRAME(data)>::operator()(   \
      const DTYPE(data) & x) const noexcept;                                  \
  template DTYPE(data) MathFunctions::Constant<1, FRAME(data)>::first_deriv(  \
      const DTYPE(data) & x) const noexcept;                                  \
  template DTYPE(data) MathFunctions::Constant<1, FRAME(data)>::second_deriv( \
      const DTYPE(data) & x) const noexcept;                                  \
  template DTYPE(data) MathFunctions::Constant<1, FRAME(data)>::third_deriv(  \
      const DTYPE(data) & x) const noexcept;                                  \
  template DTYPE(data)                                                        \
      MathFunctions::Constant<1, FRAME(data)>::apply_call_operator(           \
          const DTYPE(data) & x) const noexcept;                              \
  template DTYPE(data)                                                        \
      MathFunctions::Constant<1, FRAME(data)>::apply_first_deriv(             \
          const DTYPE(data) & x) const noexcept;                              \
  template DTYPE(data)                                                        \
      MathFunctions::Constant<1, FRAME(data)>::apply_second_deriv(            \
          const DTYPE(data) & x) const noexcept;                              \
  template DTYPE(data)                                                        \
      MathFunctions::Constant<1, FRAME(data)>::apply_third_deriv(             \
          const DTYPE(data) & x) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (Frame::Grid, Frame::Inertial),
                        (double, DataVector))
#undef DTYPE
#undef FRAME
#undef INSTANTIATE

/// \endcond
