// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Tags/GaugeCondition.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Dispatch.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Gauges.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpacetimeDerivativeOfSpacetimeMetric.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace gh::gauges::Tags {
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
  // Just compute d4_spacetime_metric internally for now. Since compute tags
  // are only used for observing, this should be fine.
  tnsr::abb<DataVector, Dim, Frame::Inertial> d4_spacetime_metric{
      get(lapse).size()};
  gh::spacetime_derivative_of_spacetime_metric(
      make_not_null(&d4_spacetime_metric), lapse, shift, pi, phi);
  Scalar<DataVector> half_pi_two_normals{get(lapse).size(), 0.0};
  tnsr::i<DataVector, Dim, Frame::Inertial> half_phi_two_normals{
      get(lapse).size(), 0.0};
  for (size_t a = 0; a < Dim + 1; ++a) {
    get(half_pi_two_normals) += spacetime_unit_normal.get(a) *
                                spacetime_unit_normal.get(a) * pi.get(a, a);
    for (size_t i = 0; i < Dim; ++i) {
      half_phi_two_normals.get(i) += 0.5 * spacetime_unit_normal.get(a) *
                                     spacetime_unit_normal.get(a) *
                                     phi.get(i, a, a);
    }
    for (size_t b = a + 1; b < Dim + 1; ++b) {
      get(half_pi_two_normals) += 2.0 * spacetime_unit_normal.get(a) *
                                  spacetime_unit_normal.get(b) * pi.get(a, b);
      for (size_t i = 0; i < Dim; ++i) {
        half_phi_two_normals.get(i) += spacetime_unit_normal.get(a) *
                                       spacetime_unit_normal.get(b) *
                                       phi.get(i, a, b);
      }
    }
  }
  get(half_pi_two_normals) *= 0.5;

  dispatch(
      make_not_null(
          &get<::gh::Tags::GaugeH<DataVector, Dim>>(*gauge_and_deriv)),
      make_not_null(&get<::gh::Tags::SpacetimeDerivGaugeH<DataVector, Dim>>(
          *gauge_and_deriv)),
      lapse, shift, spacetime_unit_normal_one_form, spacetime_unit_normal,
      sqrt_det_spatial_metric, inverse_spatial_metric, d4_spacetime_metric,
      half_pi_two_normals, half_phi_two_normals, spacetime_metric, pi, phi,
      mesh, time, inertial_coords, inverse_jacobian, gauge_condition);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) \
  template class GaugeAndDerivativeCompute<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace gh::gauges::Tags
