// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/HamiltonianConstraint.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace gr {
template <typename DataType, size_t SpatialDim, typename Frame>
void hamiltonian_constraint_in_vacuum(
    const gsl::not_null<Scalar<DataType>*> hamiltonian_constraint,
    const Scalar<DataType>& ricci_scalar,
    const Scalar<DataType>& trace_extrinsic_curvature,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature) {
  set_number_of_grid_points(hamiltonian_constraint, ricci_scalar);
  tenex::evaluate(hamiltonian_constraint,
                  ricci_scalar() + square(trace_extrinsic_curvature()) -
                      extrinsic_curvature(ti::i, ti::j) *
                          inverse_spatial_metric(ti::I, ti::K) *
                          inverse_spatial_metric(ti::J, ti::L) *
                          extrinsic_curvature(ti::k, ti::l));
}

template <typename DataType, size_t SpatialDim, typename Frame>
Scalar<DataType> hamiltonian_constraint_in_vacuum(
    const Scalar<DataType>& ricci_scalar,
    const Scalar<DataType>& trace_extrinsic_curvature,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature) {
  Scalar<DataType> result{};
  hamiltonian_constraint_in_vacuum(make_not_null(&result), ricci_scalar,
                                   trace_extrinsic_curvature,
                                   inverse_spatial_metric, extrinsic_curvature);
  return result;
}
}  // namespace gr

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                         \
  template void gr::hamiltonian_constraint_in_vacuum(                \
      gsl::not_null<Scalar<DTYPE(data)>*> hamiltonian_constraint,    \
      const Scalar<DTYPE(data)>& ricci_scalar,                       \
      const Scalar<DTYPE(data)>& trace_extrinsic_curvature,          \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&           \
          inverse_spatial_metric,                                    \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&           \
          extrinsic_curvature);                                      \
  template Scalar<DTYPE(data)> gr::hamiltonian_constraint_in_vacuum( \
      const Scalar<DTYPE(data)>& ricci_scalar,                       \
      const Scalar<DTYPE(data)>& trace_extrinsic_curvature,          \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&           \
          inverse_spatial_metric,                                    \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&           \
          extrinsic_curvature);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector), (1, 2, 3),
                        (Frame::Grid, Frame::Inertial))

#undef DTYPE
#undef DIM
#undef FRAME
#undef INSTANTIATE
