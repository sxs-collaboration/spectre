// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/GaussianPlusConstant.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <pup_stl.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace GeneralizedHarmonic::ConstraintDamping {

template <size_t VolumeDim, typename Fr>
GaussianPlusConstant<VolumeDim, Fr>::GaussianPlusConstant(
    const double constant, const double amplitude, const double width,
    const std::array<double, VolumeDim>& center) noexcept
    : constant_(constant),
      amplitude_(amplitude),
      inverse_width_(1.0 / width),
      center_(center) {}

template <size_t VolumeDim, typename Fr>
template <typename T>
tnsr::I<T, VolumeDim, Fr>
GaussianPlusConstant<VolumeDim, Fr>::centered_coordinates(
    const tnsr::I<T, VolumeDim, Fr>& x) const noexcept {
  tnsr::I<T, VolumeDim, Fr> centered_coords = x;
  for (size_t i = 0; i < VolumeDim; ++i) {
    centered_coords.get(i) -= gsl::at(center_, i);
  }
  return centered_coords;
}

template <size_t VolumeDim, typename Fr>
template <typename T>
Scalar<T> GaussianPlusConstant<VolumeDim, Fr>::apply_call_operator(
    const tnsr::I<T, VolumeDim, Fr>& centered_coords) const noexcept {
  Scalar<T> result = dot_product(centered_coords, centered_coords);
  get(result) =
      constant_ + amplitude_ * exp(-get(result) * square(inverse_width_));
  return result;
}

template <size_t VolumeDim, typename Fr>
Scalar<double> GaussianPlusConstant<VolumeDim, Fr>::operator()(
    const tnsr::I<double, VolumeDim, Fr>& x) const noexcept {
  return apply_call_operator(centered_coordinates(x));
}
template <size_t VolumeDim, typename Fr>
Scalar<DataVector> GaussianPlusConstant<VolumeDim, Fr>::operator()(
    const tnsr::I<DataVector, VolumeDim, Fr>& x) const noexcept {
  return apply_call_operator(centered_coordinates(x));
}

template <size_t VolumeDim, typename Fr>
void GaussianPlusConstant<VolumeDim, Fr>::pup(PUP::er& p) {
  DampingFunction<VolumeDim, Fr>::pup(p);
  p | constant_;
  p | amplitude_;
  p | inverse_width_;
  p | center_;
}

template <size_t VolumeDim, typename Fr>
auto GaussianPlusConstant<VolumeDim, Fr>::get_clone() const noexcept
    -> std::unique_ptr<DampingFunction<VolumeDim, Fr>> {
  return std::make_unique<GaussianPlusConstant<VolumeDim, Fr>>(*this);
}
}  // namespace GeneralizedHarmonic::ConstraintDamping

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATE(_, data)                                                  \
  template GeneralizedHarmonic::ConstraintDamping::                           \
      GaussianPlusConstant<DIM(data), FRAME(data)>::GaussianPlusConstant(     \
          const double constant, const double amplitude, const double width,  \
          const std::array<double, DIM(data)>& center) noexcept;              \
  template void GeneralizedHarmonic::ConstraintDamping::GaussianPlusConstant< \
      DIM(data), FRAME(data)>::pup(PUP::er& p);                               \
  template auto GeneralizedHarmonic::ConstraintDamping::GaussianPlusConstant< \
      DIM(data), FRAME(data)>::get_clone() const noexcept                     \
      ->std::unique_ptr<DampingFunction<DIM(data), FRAME(data)>>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial))
#undef DIM
#undef FRAME
#undef INSTANTIATE

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                            \
  template Scalar<DTYPE(data)> GeneralizedHarmonic::ConstraintDamping:: \
      GaussianPlusConstant<DIM(data), FRAME(data)>::operator()(         \
          const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& x)        \
          const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial),
                        (double, DataVector))
#undef FRAME
#undef DIM
#undef DTYPE
#undef INSTANTIATE

/// \endcond
