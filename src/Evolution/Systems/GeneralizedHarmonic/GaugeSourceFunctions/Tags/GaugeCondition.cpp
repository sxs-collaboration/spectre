// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Tags/GaugeCondition.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Dispatch.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Gauges.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Options.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace GeneralizedHarmonic::gauges::Tags {
template <size_t Dim>
void GaugeAndDerivativeCompute<Dim>::function(
    const gsl::not_null<return_type*> gauge_and_deriv,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const tnsr::a<DataVector, Dim, Frame::Inertial>&
        spacetime_unit_normal_one_form,
    const tnsr::A<DataVector, Dim, Frame::Inertial>& spacetime_unit_normal,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::II<DataVector, Dim, Frame::Inertial>& inverse_spatial_metric,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi,
    const Mesh<Dim>& mesh, const double time,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian,
    const gauges::GaugeCondition& gauge_condition) {
  if (const size_t num_points = mesh.number_of_grid_points();
      UNLIKELY(gauge_and_deriv->number_of_grid_points() != num_points)) {
    gauge_and_deriv->initialize(num_points);
  }
  dispatch(make_not_null(
               &get<::GeneralizedHarmonic::Tags::GaugeH<Dim, Frame::Inertial>>(
                   *gauge_and_deriv)),
           make_not_null(&get<::GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<
                             Dim, Frame::Inertial>>(*gauge_and_deriv)),
           lapse, shift, spacetime_unit_normal_one_form, spacetime_unit_normal,
           sqrt_det_spatial_metric, inverse_spatial_metric, spacetime_metric,
           pi, phi, mesh, time, inertial_coords, inverse_jacobian,
           gauge_condition);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) \
  template class GaugeAndDerivativeCompute<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace GeneralizedHarmonic::gauges::Tags
