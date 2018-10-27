// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/LorentzFactor.hpp"

#include <cmath>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"

/// \cond
namespace hydro {
template <typename DataType>
Scalar<DataType> lorentz_factor(
    const Scalar<DataType>& spatial_velocity_squared) noexcept {
  return Scalar<DataType>{1.0 / sqrt(1.0 - get(spatial_velocity_squared))};
}

template <typename DataType, size_t Dim, typename Frame>
Scalar<DataType> lorentz_factor(
    const tnsr::I<DataType, Dim, Frame>& spatial_velocity,
    const tnsr::i<DataType, Dim, Frame>& spatial_velocity_form) noexcept {
  return lorentz_factor(dot_product(spatial_velocity, spatial_velocity_form));
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                \
  template Scalar<DTYPE(data)> lorentz_factor(                              \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& spatial_velocity, \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>&                   \
          spatial_velocity_form) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

template Scalar<DataVector> lorentz_factor(
    const Scalar<DataVector>& spatial_velocity_squared) noexcept;
template Scalar<double> lorentz_factor(
    const Scalar<double>& spatial_velocity_squared) noexcept;

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
}  // namespace hydro
/// \endcond
