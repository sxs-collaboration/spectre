// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "MassFlux.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace hydro {

template <typename DataType, size_t Dim, typename Frame>
void mass_flux(const gsl::not_null<tnsr::I<DataType, Dim, Frame>*> result,
               const Scalar<DataType>& rest_mass_density,
               const tnsr::I<DataType, Dim, Frame>& spatial_velocity,
               const Scalar<DataType>& lorentz_factor,
               const Scalar<DataType>& lapse,
               const tnsr::I<DataType, Dim, Frame>& shift,
               const Scalar<DataType>& sqrt_det_spatial_metric) noexcept {
  destructive_resize_components(result, get_size(get(rest_mass_density)));
  for (size_t i = 0; i < Dim; ++i) {
    result->get(i) = get(rest_mass_density) * get(lorentz_factor) *
                     get(sqrt_det_spatial_metric) *
                     (get(lapse) * spatial_velocity.get(i) - shift.get(i));
  }
}

template <typename DataType, size_t Dim, typename Frame>
tnsr::I<DataType, Dim, Frame> mass_flux(
    const Scalar<DataType>& rest_mass_density,
    const tnsr::I<DataType, Dim, Frame>& spatial_velocity,
    const Scalar<DataType>& lorentz_factor, const Scalar<DataType>& lapse,
    const tnsr::I<DataType, Dim, Frame>& shift,
    const Scalar<DataType>& sqrt_det_spatial_metric) noexcept {
  tnsr::I<DataType, Dim, Frame> result{};
  mass_flux(make_not_null(&result), rest_mass_density, spatial_velocity,
            lorentz_factor, lapse, shift, sqrt_det_spatial_metric);
  return result;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define INSTANTIATE(_, data)                                                \
  template void mass_flux(                                                  \
      const gsl::not_null<tnsr::I<DTYPE(data), DIM(data), FRAME(data)>*>    \
          result,                                                           \
      const Scalar<DTYPE(data)>& rest_mass_density,                         \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& spatial_velocity, \
      const Scalar<DTYPE(data)>& lorentz_factor,                            \
      const Scalar<DTYPE(data)>& lapse,                                     \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,            \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric) noexcept;         \
  template tnsr::I<DTYPE(data), DIM(data), FRAME(data)> mass_flux(          \
      const Scalar<DTYPE(data)>& rest_mass_density,                         \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& spatial_velocity, \
      const Scalar<DTYPE(data)>& lorentz_factor,                            \
      const Scalar<DTYPE(data)>& lapse,                                     \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,            \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
}  // namespace hydro
