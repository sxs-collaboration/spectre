// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/TransportVelocity.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace hydro {
template <typename DataType, size_t Dim, typename Fr>
tnsr::I<DataType, Dim, Fr> transport_velocity(
    const tnsr::I<DataType, Dim, Fr>& spatial_velocity,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, Dim, Fr>& shift) noexcept {
  auto result = make_with_value<tnsr::I<DataType, Dim, Fr>>(lapse, 0.0);
  for (size_t i = 0; i < Dim; ++i) {
    result.get(i) = get(lapse) * spatial_velocity.get(i) - shift.get(i);
  }
  return result;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                \
  template tnsr::I<DTYPE(data), DIM(data), FRAME(data)> transport_velocity( \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& spatial_velocity, \
      const Scalar<DTYPE(data)>& lapse,                                     \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
}  // namespace hydro
/// \endcond
