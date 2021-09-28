// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/LorentzFactor.hpp"

#include <cmath>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace hydro {
template <typename DataType>
void lorentz_factor(const gsl::not_null<Scalar<DataType>*> result,
                    const Scalar<DataType>& spatial_velocity_squared) {
  destructive_resize_components(result,
                                get_size(get(spatial_velocity_squared)));
  get(*result) = 1.0 / sqrt(1.0 - get(spatial_velocity_squared));
}

template <typename DataType>
Scalar<DataType> lorentz_factor(
    const Scalar<DataType>& spatial_velocity_squared) {
  Scalar<DataType> result{};
  lorentz_factor(make_not_null(&result), spatial_velocity_squared);
  return result;
}

template <typename DataType, size_t Dim, typename Frame>
void lorentz_factor(
    const gsl::not_null<Scalar<DataType>*> result,
    const tnsr::I<DataType, Dim, Frame>& spatial_velocity,
    const tnsr::i<DataType, Dim, Frame>& spatial_velocity_form) {
  destructive_resize_components(result, get_size(get<0>(spatial_velocity)));
  lorentz_factor(result, dot_product(spatial_velocity, spatial_velocity_form));
}

template <typename DataType, size_t Dim, typename Frame>
Scalar<DataType> lorentz_factor(
    const tnsr::I<DataType, Dim, Frame>& spatial_velocity,
    const tnsr::i<DataType, Dim, Frame>& spatial_velocity_form) {
  Scalar<DataType> result{};
  lorentz_factor(make_not_null(&result), spatial_velocity,
                 spatial_velocity_form);
  return result;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                \
  template void lorentz_factor(                                             \
      const gsl::not_null<Scalar<DTYPE(data)>*> result,                     \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& spatial_velocity, \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>&                   \
          spatial_velocity_form);                                           \
  template Scalar<DTYPE(data)> lorentz_factor(                              \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& spatial_velocity, \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>&                   \
          spatial_velocity_form);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

template void lorentz_factor(
    const gsl::not_null<Scalar<DataVector>*> result,
    const Scalar<DataVector>& spatial_velocity_squared);
template Scalar<DataVector> lorentz_factor(
    const Scalar<DataVector>& spatial_velocity_squared);
template void lorentz_factor(const gsl::not_null<Scalar<double>*> result,
                             const Scalar<double>& spatial_velocity_squared);
template Scalar<double> lorentz_factor(
    const Scalar<double>& spatial_velocity_squared);

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
}  // namespace hydro
