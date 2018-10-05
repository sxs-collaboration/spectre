// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace GeneralizedHarmonic {
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::iaa<DataType, SpatialDim, Frame> three_index_constraint(
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  auto constraint =
      make_with_value<tnsr::iaa<DataType, SpatialDim, Frame>>(phi, 0.0);
  three_index_constraint<SpatialDim, Frame, DataType>(&constraint,
                                                      d_spacetime_metric, phi);
  return constraint;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void three_index_constraint(
    gsl::not_null<tnsr::iaa<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  // Declare iterators for d_spacetime_metric and phi outside the for loop,
  // because they are const but constraint is not
  auto d_spacetime_metric_it = d_spacetime_metric.begin(), phi_it = phi.begin();

  for (auto constraint_it = (*constraint).begin();
       constraint_it != (*constraint).end();
       ++constraint_it, (void)++d_spacetime_metric_it, (void)++phi_it) {
    *constraint_it = *d_spacetime_metric_it - *phi_it;
  }
}
}  // namespace GeneralizedHarmonic

// Explicit Instantiations
/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                               \
  template tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>                  \
  GeneralizedHarmonic::three_index_constraint(                             \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                \
          d_spacetime_metric,                                              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept; \
  template void GeneralizedHarmonic::three_index_constraint(               \
      gsl::not_null<tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>*>       \
          constraint,                                                      \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                \
          d_spacetime_metric,                                              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
/// \endcond
