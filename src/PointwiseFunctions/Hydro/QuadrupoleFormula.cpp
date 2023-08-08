// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "QuadrupoleFormula.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace hydro {

template <typename DataType, size_t Dim, typename Fr>
void quadrupole_moment(
    const gsl::not_null<tnsr::ii<DataType, Dim, Fr>*> result,
    const Scalar<DataType>& tilde_d,
    const tnsr::I<DataType, Dim, Fr>& coordinates) {
  set_number_of_grid_points(result, tilde_d);
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = i; j < Dim; j++) {
      result->get(i, j) = get(tilde_d) * coordinates.get(i) *
                          coordinates.get(j);
    }
  }
}

template <typename DataType, size_t Dim, typename Fr>
void quadrupole_moment_derivative(
    const gsl::not_null<tnsr::ii<DataType, Dim, Fr>*> result,
    const Scalar<DataType>& tilde_d,
    const tnsr::I<DataType, Dim, Fr>& coordinates,
    const tnsr::I<DataType, Dim, Fr>& spatial_velocity) {
  set_number_of_grid_points(result, tilde_d);
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = i; j < Dim; j++) {
      result->get(i, j) = get(tilde_d) *
                          (spatial_velocity.get(i) * coordinates.get(j) +
                          coordinates.get(i) * spatial_velocity.get(j));
    }
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                 \
  template void quadrupole_moment<DTYPE(data), DIM(data), FRAME(data)>(      \
      const gsl::not_null<tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>*>    \
          result,                                                            \
      const Scalar<DTYPE(data)>& tilde_d,                                    \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& spatial_position); \
  template void quadrupole_moment_derivative<DTYPE(data), DIM(data),         \
          FRAME(data)>(                                                      \
      const gsl::not_null<tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>*>    \
          result,                                                            \
      const Scalar<DTYPE(data)>& tilde_d,                                    \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& spatial_position,  \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& spatial_velocity);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (DataVector),
                        (Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE

}  // namespace hydro
