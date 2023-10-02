// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "TransportVelocity.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace hydro {

template <typename DataType, size_t Dim, typename Fr>
void transport_velocity(const gsl::not_null<tnsr::I<DataType, Dim, Fr>*> result,
                        const tnsr::I<DataType, Dim, Fr>& spatial_velocity,
                        const Scalar<DataType>& lapse,
                        const tnsr::I<DataType, Dim, Fr>& shift) {
  set_number_of_grid_points(result, lapse);
  for (size_t i = 0; i < Dim; i++) {
    result->get(i) = spatial_velocity.get(i) * get(lapse) - shift.get(i);
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                \
  template void transport_velocity<DTYPE(data), DIM(data), FRAME(data)>(    \
      const gsl::not_null<tnsr::I<DTYPE(data), DIM(data), FRAME(data)>*>    \
          result,                                                           \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& spatial_velocity, \
      const Scalar<DTYPE(data)>& lapse,                                     \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE

}  // namespace hydro
