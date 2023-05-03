// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/Sources.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ForceFree/ElectricCurrentDensity.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace ForceFree {

namespace detail {
void sources_impl(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        source_tilde_e,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        source_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> source_tilde_psi,
    const gsl::not_null<Scalar<DataVector>*> source_tilde_phi,

    // temp variables
    const tnsr::I<DataVector, 3, Frame::Inertial>&
        trace_spatial_christoffel_second,

    // EM args
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_psi, const Scalar<DataVector>& tilde_phi,
    const Scalar<DataVector>& tilde_q,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_j_drift,
    const double kappa_psi, const double kappa_phi,
    // GR args
    const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, 3, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& d_shift,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& extrinsic_curvature) {
  // S(\tilde{E}^i)
  raise_or_lower_index(source_tilde_e, d_lapse, inv_spatial_metric);
  for (size_t i = 0; i < 3; ++i) {
    source_tilde_e->get(i) -=
        get(lapse) * trace_spatial_christoffel_second.get(i);
    source_tilde_e->get(i) *= get(tilde_psi);
    source_tilde_e->get(i) -= tilde_j_drift.get(i);
    for (size_t m = 0; m < 3; ++m) {
      source_tilde_e->get(i) -= tilde_e.get(m) * d_shift.get(m, i);
    }
  }

  // S(\tilde{B}^i)
  raise_or_lower_index(source_tilde_b, d_lapse, inv_spatial_metric);
  for (size_t i = 0; i < 3; ++i) {
    source_tilde_b->get(i) -=
        get(lapse) * trace_spatial_christoffel_second.get(i);
    source_tilde_b->get(i) *= get(tilde_phi);
    for (size_t m = 0; m < 3; ++m) {
      source_tilde_b->get(i) -= tilde_b.get(m) * d_shift.get(m, i);
    }
  }

  // S(\tilde{\psi})
  trace(source_tilde_psi, extrinsic_curvature, inv_spatial_metric);
  get(*source_tilde_psi) += kappa_psi;
  get(*source_tilde_psi) *= -1.0 * get(tilde_psi);
  get(*source_tilde_psi) += get(tilde_q);
  get(*source_tilde_psi) *= get(lapse);
  for (size_t m = 0; m < 3; ++m) {
    get(*source_tilde_psi) += tilde_e.get(m) * d_lapse.get(m);
  }

  // S(\tilde{\phi})
  trace(source_tilde_phi, extrinsic_curvature, inv_spatial_metric);
  get(*source_tilde_phi) += kappa_phi;
  get(*source_tilde_phi) *= -1.0 * get(lapse) * get(tilde_phi);
  for (size_t m = 0; m < 3; ++m) {
    get(*source_tilde_phi) += tilde_b.get(m) * d_lapse.get(m);
  }
}
}  // namespace detail

void Sources::apply(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        source_tilde_e,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        source_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> source_tilde_psi,
    const gsl::not_null<Scalar<DataVector>*> source_tilde_phi,
    // EM variables
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_psi, const Scalar<DataVector>& tilde_phi,
    const Scalar<DataVector>& tilde_q, const double kappa_psi,
    const double kappa_phi, const double parallel_conductivity,
    // GR variables
    const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, 3, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& d_shift,
    const tnsr::ijj<DataVector, 3, Frame::Inertial>& d_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& extrinsic_curvature) {
  // temp variable to store metric derivative quantities
  Variables<tmpl::list<
      ::Tags::TempI<0, 3>, gr::Tags::SpatialChristoffelFirstKind<DataVector, 3>,
      gr::Tags::SpatialChristoffelSecondKind<DataVector, 3>,
      gr::Tags::TraceSpatialChristoffelSecondKind<DataVector, 3>>>
      temp_tensors(get<0>(tilde_e).size());

  // compute the drift component of tilde_J
  auto& tilde_j_drift = get<::Tags::TempI<0, 3>>(temp_tensors);
  ComputeDriftTildeJ::apply(make_not_null(&tilde_j_drift), tilde_q, tilde_e,
                            tilde_b, parallel_conductivity, lapse,
                            sqrt_det_spatial_metric, spatial_metric);

  // compute the product \gamma^jk \Gamma^i_{jk}
  auto& spatial_christoffel_first_kind =
      get<gr::Tags::SpatialChristoffelFirstKind<DataVector, 3>>(temp_tensors);
  gr::christoffel_first_kind(make_not_null(&spatial_christoffel_first_kind),
                             d_spatial_metric);
  auto& spatial_christoffel_second_kind =
      get<gr::Tags::SpatialChristoffelSecondKind<DataVector, 3>>(temp_tensors);
  raise_or_lower_first_index(make_not_null(&spatial_christoffel_second_kind),
                             spatial_christoffel_first_kind,
                             inv_spatial_metric);
  auto& trace_spatial_christoffel_second =
      get<gr::Tags::TraceSpatialChristoffelSecondKind<DataVector, 3>>(
          temp_tensors);
  trace_last_indices(make_not_null(&trace_spatial_christoffel_second),
                     spatial_christoffel_second_kind, inv_spatial_metric);

  detail::sources_impl(source_tilde_e, source_tilde_b, source_tilde_psi,
                       source_tilde_phi, trace_spatial_christoffel_second,
                       tilde_e, tilde_b, tilde_psi, tilde_phi, tilde_q,
                       tilde_j_drift, kappa_psi, kappa_phi, lapse, d_lapse,
                       d_shift, inv_spatial_metric, extrinsic_curvature);
}

}  // namespace ForceFree
