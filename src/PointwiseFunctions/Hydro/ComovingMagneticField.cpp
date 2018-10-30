// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/ComovingMagneticField.hpp"

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace hydro {
template <typename DataType, size_t Dim, typename Fr>
tnsr::A<DataType, Dim, Fr> comoving_magnetic_field(
    const tnsr::I<DataType, Dim, Fr>& eulerian_b_field,
    const tnsr::I<DataType, Dim, Fr>& transport_velocity,
    const tnsr::i<DataType, Dim, Fr>& spatial_velocity_oneform,
    const Scalar<DataType>& lorentz_factor,
    const Scalar<DataType>& lapse) noexcept {
  auto result = make_with_value<tnsr::A<DataType, Dim, Fr>>(lapse, 0.0);

  get<0>(result) =
      get(lorentz_factor) *
      get(dot_product(eulerian_b_field, spatial_velocity_oneform)) / get(lapse);
  for (size_t i = 0; i < Dim; ++i) {
    result.get(i + 1) = eulerian_b_field.get(i) / get(lorentz_factor) +
                        get<0>(result) * transport_velocity.get(i);
  }
  return result;
}

template <typename DataType, size_t Dim, typename Fr>
Scalar<DataType> comoving_magnetic_field_squared(
    const tnsr::I<DataType, Dim, Fr>& eulerian_b_field,
    const Scalar<DataType>& eulerian_b_field_squared,
    const tnsr::i<DataType, Dim, Fr>& spatial_velocity_oneform,
    const Scalar<DataType>& lorentz_factor) noexcept {
  return Scalar<DataType>{
      get(eulerian_b_field_squared) / square(get(lorentz_factor)) +
      square(get(dot_product(eulerian_b_field, spatial_velocity_oneform)))};
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                \
  template tnsr::A<DTYPE(data), DIM(data), FRAME(data)>                     \
  comoving_magnetic_field(                                                  \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& eulerian_b_field, \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& spatial_velocity, \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>&                   \
          spatial_velocity_oneform,                                         \
      const Scalar<DTYPE(data)>& lorentz_factor,                            \
      const Scalar<DTYPE(data)>& lapse) noexcept;                           \
  template Scalar<DTYPE(data)> comoving_magnetic_field_squared(             \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& eulerian_b_field, \
      const Scalar<DTYPE(data)>& eulerian_b_field_squared,                  \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>&                   \
          spatial_velocity_oneform,                                         \
      const Scalar<DTYPE(data)>& lorentz_factor) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
}  // namespace hydro
/// \endcond
