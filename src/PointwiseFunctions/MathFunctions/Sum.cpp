// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/MathFunctions/Sum.hpp"

#include <cmath>
#include <memory>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace MathFunctions {

template <typename Fr>
Sum<1, Fr>::Sum(std::unique_ptr<MathFunction<1, Fr>> math_function_a,
                std::unique_ptr<MathFunction<1, Fr>> math_function_b) noexcept
    : math_function_a_(std::move(math_function_a)),
      math_function_b_(std::move(math_function_b)) {}

template <typename Fr>
double Sum<1, Fr>::operator()(const double& x) const noexcept {
  return apply_call_operator(x);
}

template <typename Fr>
DataVector Sum<1, Fr>::operator()(const DataVector& x) const noexcept {
  return apply_call_operator(x);
}

template <typename Fr>
double Sum<1, Fr>::first_deriv(const double& x) const noexcept {
  return apply_first_deriv(x);
}

template <typename Fr>
DataVector Sum<1, Fr>::first_deriv(const DataVector& x) const noexcept {
  return apply_first_deriv(x);
}

template <typename Fr>
double Sum<1, Fr>::second_deriv(const double& x) const noexcept {
  return apply_second_deriv(x);
}

template <typename Fr>
DataVector Sum<1, Fr>::second_deriv(const DataVector& x) const noexcept {
  return apply_second_deriv(x);
}

template <typename Fr>
double Sum<1, Fr>::third_deriv(const double& x) const noexcept {
  return apply_third_deriv(x);
}

template <typename Fr>
DataVector Sum<1, Fr>::third_deriv(const DataVector& x) const noexcept {
  return apply_third_deriv(x);
}

template <typename Fr>
template <typename T>
T Sum<1, Fr>::apply_call_operator(const T& x) const noexcept {
  return math_function_a_->operator()(x) + math_function_b_->operator()(x);
}

template <typename Fr>
template <typename T>
T Sum<1, Fr>::apply_first_deriv(const T& x) const noexcept {
  return math_function_a_->first_deriv(x) + math_function_b_->first_deriv(x);
}

template <typename Fr>
template <typename T>
T Sum<1, Fr>::apply_second_deriv(const T& x) const noexcept {
  return math_function_a_->second_deriv(x) + math_function_b_->second_deriv(x);
}

template <typename Fr>
template <typename T>
T Sum<1, Fr>::apply_third_deriv(const T& x) const noexcept {
  return math_function_a_->third_deriv(x) + math_function_b_->third_deriv(x);
}

template <typename Fr>
void Sum<1, Fr>::pup(PUP::er& p) {
  MathFunction<1, Fr>::pup(p);
  p | math_function_a_;
  p | math_function_b_;
}

template <size_t VolumeDim, typename Fr>
Sum<VolumeDim, Fr>::Sum(
    std::unique_ptr<MathFunction<VolumeDim, Fr>> math_function_a,
    std::unique_ptr<MathFunction<VolumeDim, Fr>> math_function_b) noexcept
    : math_function_a_(std::move(math_function_a)),
      math_function_b_(std::move(math_function_b)) {}

template <size_t VolumeDim, typename Fr>
template <typename T>
Scalar<T> Sum<VolumeDim, Fr>::apply_call_operator(
    const tnsr::I<T, VolumeDim, Fr>& x) const noexcept {
  Scalar<T> result = math_function_a_->operator()(x);
  get(result) += get(math_function_b_->operator()(x));
  return result;
}

template <size_t VolumeDim, typename Fr>
template <typename T>
tnsr::i<T, VolumeDim, Fr> Sum<VolumeDim, Fr>::apply_first_deriv(
    const tnsr::I<T, VolumeDim, Fr>& x) const noexcept {
  const tnsr::i<T, VolumeDim, Fr>& first_deriv_a =
      math_function_a_->first_deriv(x);
  const tnsr::i<T, VolumeDim, Fr>& first_deriv_b =
      math_function_b_->first_deriv(x);
  return TensorExpressions::evaluate<ti_a_t>(first_deriv_a(ti_a) +
                                             first_deriv_b(ti_a));
}

template <size_t VolumeDim, typename Fr>
template <typename T>
tnsr::ii<T, VolumeDim, Fr> Sum<VolumeDim, Fr>::apply_second_deriv(
    const tnsr::I<T, VolumeDim, Fr>& x) const noexcept {
  const tnsr::ii<T, VolumeDim, Fr>& second_deriv_a =
      math_function_a_->second_deriv(x);
  const tnsr::ii<T, VolumeDim, Fr>& second_deriv_b =
      math_function_b_->second_deriv(x);
  return TensorExpressions::evaluate<ti_a_t, ti_b_t>(
      second_deriv_a(ti_a, ti_b) + second_deriv_b(ti_a, ti_b));
}

template <size_t VolumeDim, typename Fr>
template <typename T>
tnsr::iii<T, VolumeDim, Fr> Sum<VolumeDim, Fr>::apply_third_deriv(
    const tnsr::I<T, VolumeDim, Fr>& x) const noexcept {
  const tnsr::iii<T, VolumeDim, Fr>& third_deriv_a =
      math_function_a_->third_deriv(x);
  const tnsr::iii<T, VolumeDim, Fr>& third_deriv_b =
      math_function_b_->third_deriv(x);
  return TensorExpressions::evaluate<ti_a_t, ti_b_t, ti_c_t>(
      third_deriv_a(ti_a, ti_b, ti_c) + third_deriv_b(ti_a, ti_b, ti_c));
}

template <size_t VolumeDim, typename Fr>
Scalar<double> Sum<VolumeDim, Fr>::operator()(
    const tnsr::I<double, VolumeDim, Fr>& x) const noexcept {
  return apply_call_operator(x);
}
template <size_t VolumeDim, typename Fr>
Scalar<DataVector> Sum<VolumeDim, Fr>::operator()(
    const tnsr::I<DataVector, VolumeDim, Fr>& x) const noexcept {
  return apply_call_operator(x);
}

template <size_t VolumeDim, typename Fr>
tnsr::i<double, VolumeDim, Fr> Sum<VolumeDim, Fr>::first_deriv(
    const tnsr::I<double, VolumeDim, Fr>& x) const noexcept {
  return apply_first_deriv(x);
}
template <size_t VolumeDim, typename Fr>
tnsr::i<DataVector, VolumeDim, Fr> Sum<VolumeDim, Fr>::first_deriv(
    const tnsr::I<DataVector, VolumeDim, Fr>& x) const noexcept {
  return apply_first_deriv(x);
}

template <size_t VolumeDim, typename Fr>
tnsr::ii<double, VolumeDim, Fr> Sum<VolumeDim, Fr>::second_deriv(
    const tnsr::I<double, VolumeDim, Fr>& x) const noexcept {
  return apply_second_deriv(x);
}
template <size_t VolumeDim, typename Fr>
tnsr::ii<DataVector, VolumeDim, Fr> Sum<VolumeDim, Fr>::second_deriv(
    const tnsr::I<DataVector, VolumeDim, Fr>& x) const noexcept {
  return apply_second_deriv(x);
}

template <size_t VolumeDim, typename Fr>
tnsr::iii<double, VolumeDim, Fr> Sum<VolumeDim, Fr>::third_deriv(
    const tnsr::I<double, VolumeDim, Fr>& x) const noexcept {
  return apply_third_deriv(x);
}
template <size_t VolumeDim, typename Fr>
tnsr::iii<DataVector, VolumeDim, Fr> Sum<VolumeDim, Fr>::third_deriv(
    const tnsr::I<DataVector, VolumeDim, Fr>& x) const noexcept {
  return apply_third_deriv(x);
}

template <size_t VolumeDim, typename Fr>
void Sum<VolumeDim, Fr>::pup(PUP::er& p) {
  MathFunction<VolumeDim, Fr>::pup(p);
  p | math_function_a_;
  p | math_function_b_;
}
}  // namespace MathFunctions

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATE(_, data)                                                 \
  template MathFunctions::Sum<DIM(data), FRAME(data)>::Sum(                  \
      std::unique_ptr<MathFunction<DIM(data), FRAME(data)>> math_function_a, \
      std::unique_ptr<MathFunction<DIM(data), FRAME(data)>>                  \
          math_function_b) noexcept;                                         \
  template void MathFunctions::Sum<DIM(data), FRAME(data)>::pup(PUP::er& p);

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3), (Frame::Grid, Frame::Inertial))
#undef DIM
#undef FRAME
#undef INSTANTIATE

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                 \
  template Scalar<DTYPE(data)>                                               \
  MathFunctions::Sum<DIM(data), FRAME(data)>::operator()(                    \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& x) const noexcept; \
  template tnsr::i<DTYPE(data), DIM(data), FRAME(data)>                      \
  MathFunctions::Sum<DIM(data), FRAME(data)>::first_deriv(                   \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& x) const noexcept; \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>                     \
  MathFunctions::Sum<DIM(data), FRAME(data)>::second_deriv(                  \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& x) const noexcept; \
  template tnsr::iii<DTYPE(data), DIM(data), FRAME(data)>                    \
  MathFunctions::Sum<DIM(data), FRAME(data)>::third_deriv(                   \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& x) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3), (Frame::Grid, Frame::Inertial),
                        (double, DataVector))
#undef FRAME
#undef DIM
#undef DTYPE
#undef INSTANTIATE

template MathFunctions::Sum<1, Frame::Grid>::Sum(
    std::unique_ptr<MathFunction<1, Frame::Grid>> math_function_a,
    std::unique_ptr<MathFunction<1, Frame::Grid>> math_function_b) noexcept;
template MathFunctions::Sum<1, Frame::Inertial>::Sum(
    std::unique_ptr<MathFunction<1, Frame::Inertial>> math_function_a,
    std::unique_ptr<MathFunction<1, Frame::Inertial>> math_function_b) noexcept;
template void MathFunctions::Sum<1, Frame::Grid>::pup(PUP::er& p);
template void MathFunctions::Sum<1, Frame::Inertial>::pup(PUP::er& p);

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                   \
  template DTYPE(data)                                                         \
      MathFunctions::Sum<1, FRAME(data)>::operator()(const DTYPE(data) & x)    \
          const noexcept;                                                      \
  template DTYPE(data)                                                         \
      MathFunctions::Sum<1, FRAME(data)>::first_deriv(const DTYPE(data) & x)   \
          const noexcept;                                                      \
  template DTYPE(data)                                                         \
      MathFunctions::Sum<1, FRAME(data)>::second_deriv(const DTYPE(data) & x)  \
          const noexcept;                                                      \
  template DTYPE(data)                                                         \
      MathFunctions::Sum<1, FRAME(data)>::third_deriv(const DTYPE(data) & x)   \
          const noexcept;                                                      \
  template DTYPE(data)                                                         \
      MathFunctions::Sum<1, FRAME(data)>::apply_call_operator(                 \
          const DTYPE(data) & x) const noexcept;                               \
  template DTYPE(data) MathFunctions::Sum<1, FRAME(data)>::apply_first_deriv(  \
      const DTYPE(data) & x) const noexcept;                                   \
  template DTYPE(data) MathFunctions::Sum<1, FRAME(data)>::apply_second_deriv( \
      const DTYPE(data) & x) const noexcept;                                   \
  template DTYPE(data) MathFunctions::Sum<1, FRAME(data)>::apply_third_deriv(  \
      const DTYPE(data) & x) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (Frame::Grid, Frame::Inertial),
                        (double, DataVector))
#undef DTYPE
#undef FRAME
#undef INSTANTIATE

/// \endcond
