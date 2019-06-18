// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "MassFlux.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace hydro {

template <typename DataType, size_t Dim, typename Frame>
tnsr::I<DataType, Dim, Frame> mass_flux(
    const Scalar<DataType>& rest_mass_density,
    const tnsr::I<DataType, Dim, Frame>& spatial_velocity,
    const Scalar<DataType>& lorentz_factor, const Scalar<DataType>& lapse,
    const tnsr::I<DataType, Dim, Frame>& shift,
    const Scalar<DataType>& sqrt_det_spatial_metric) noexcept {
  auto result =
      make_with_value<tnsr::I<DataType, Dim, Frame>>(rest_mass_density, 0.0);
  for (size_t i = 0; i < Dim; ++i) {
    result.get(i) = get(rest_mass_density) * get(lorentz_factor) *
                    get(sqrt_det_spatial_metric) *
                    (get(lapse) * spatial_velocity.get(i) - shift.get(i));
  }
  return result;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define INSTANTIATE(_, data)                                                \
  template tnsr::I<DTYPE(data), DIM(data), FRAME(data)> mass_flux(   \
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
/// \endcond
