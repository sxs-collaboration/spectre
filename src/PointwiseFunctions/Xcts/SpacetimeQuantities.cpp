// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Xcts/SpacetimeQuantities.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Elasticity/Strain.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Xcts/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/Xcts/LongitudinalOperator.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts {

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::ii<DataVector, 3>*> spatial_metric,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::SpatialMetric<DataVector, 3> /*meta*/) const {
  *spatial_metric = conformal_metric;
  for (auto& spatial_metric_component : *spatial_metric) {
    spatial_metric_component *= pow<4>(get(conformal_factor));
  }
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::II<DataVector, 3>*> inv_spatial_metric,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::InverseSpatialMetric<DataVector, 3> /*meta*/) const {
  *inv_spatial_metric = inv_conformal_metric;
  for (auto& inv_spatial_metric_component : *inv_spatial_metric) {
    inv_spatial_metric_component /= pow<4>(get(conformal_factor));
  }
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::i<DataVector, 3>*> deriv_conformal_factor,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Tags::ConformalFactor<DataVector>, tmpl::size_t<3>,
                  Frame::Inertial> /*meta*/) const {
  partial_derivative(deriv_conformal_factor, conformal_factor, mesh,
                     inv_jacobian);
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<Scalar<DataVector>*>
        conformal_laplacian_of_conformal_factor,
    const gsl::not_null<Cache*> cache,
    detail::ConformalLaplacianOfConformalFactor<DataVector> /*meta*/) const {
  const auto& deriv_conformal_factor =
      cache->get_var(*this, ::Tags::deriv<Tags::ConformalFactor<DataVector>,
                                          tmpl::size_t<3>, Frame::Inertial>{});
  const auto conformal_factor_flux = tenex::evaluate<ti::I>(
      inv_conformal_metric(ti::I, ti::J) * deriv_conformal_factor(ti::j));
  const auto deriv_conformal_factor_flux =
      partial_derivative(conformal_factor_flux, mesh, inv_jacobian);
  tenex::evaluate(conformal_laplacian_of_conformal_factor,
                  deriv_conformal_factor_flux(ti::i, ti::I) +
                      conformal_christoffel_contracted(ti::i) *
                          conformal_factor_flux(ti::I));
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::i<DataVector, 3>*>
        deriv_lapse_times_conformal_factor,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Tags::LapseTimesConformalFactor<DataVector>, tmpl::size_t<3>,
                  Frame::Inertial> /*meta*/) const {
  partial_derivative(deriv_lapse_times_conformal_factor,
                     lapse_times_conformal_factor, mesh, inv_jacobian);
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Lapse<DataVector> /*meta*/) const {
  get(*lapse) = get(lapse_times_conformal_factor) / get(conformal_factor);
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Shift<DataVector, 3> /*meta*/) const {
  for (size_t i = 0; i < 3; ++i) {
    shift->get(i) = shift_excess.get(i) + shift_background.get(i);
  }
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::iJ<DataVector, 3>*> deriv_shift_excess,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Tags::ShiftExcess<DataVector, 3, Frame::Inertial>,
                  tmpl::size_t<3>, Frame::Inertial> /*meta*/) const {
  // It's important to avoid computing a numeric derivative of the full shift
  // (background + excess), since the background shift may increase linearly
  // with distance (it can have `Omega x r` and `a_dot * r` terms). Instead, the
  // derivative of the background shift is known analytically and included in
  // `longitudinal_shift_background_minus_dt_conformal_metric`.
  partial_derivative(deriv_shift_excess, shift_excess, mesh, inv_jacobian);
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::ii<DataVector, 3>*> shift_strain,
    const gsl::not_null<Cache*> cache,
    Tags::ShiftStrain<DataVector, 3, Frame::Inertial> /*meta*/) const {
  const auto& deriv_shift_excess = cache->get_var(
      *this, ::Tags::deriv<Tags::ShiftExcess<DataVector, 3, Frame::Inertial>,
                           tmpl::size_t<3>, Frame::Inertial>{});
  Elasticity::strain(shift_strain, deriv_shift_excess, conformal_metric,
                     deriv_conformal_metric, conformal_christoffel_first_kind,
                     shift_excess);
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::II<DataVector, 3>*> longitudinal_shift_excess,
    const gsl::not_null<Cache*> cache,
    Tags::LongitudinalShiftExcess<DataVector, 3, Frame::Inertial> /*meta*/)
    const {
  const auto& shift_strain = cache->get_var(
      *this, Tags::ShiftStrain<DataVector, 3, Frame::Inertial>{});
  Xcts::longitudinal_operator(longitudinal_shift_excess, shift_strain,
                              inv_conformal_metric);
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::I<DataVector, 3>*> div_longitudinal_shift_excess,
    const gsl::not_null<Cache*> cache,
    ::Tags::div<
        Tags::LongitudinalShiftExcess<DataVector, 3, Frame::Inertial>> /*meta*/)
    const {
  // Copy into a Variables to take the divergence because currently (Mar 2022)
  // the `divergence` function only works with Variables. This won't be used
  // for anything performance-critical, but adding a `divergence` overload
  // that takes a Tensor is an obvious optimization here.
  using tag = Tags::LongitudinalShiftExcess<DataVector, 3, Frame::Inertial>;
  Variables<tmpl::list<tag>> vars{mesh.number_of_grid_points()};
  get<tag>(vars) = cache->get_var(*this, tag{});
  const auto derivs = divergence(vars, mesh, inv_jacobian);
  *div_longitudinal_shift_excess = get<::Tags::div<tag>>(derivs);
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::II<DataVector, 3>*>
        longitudinal_shift_minus_dt_conformal_metric,
    const gsl::not_null<Cache*> cache,
    detail::LongitudinalShiftMinusDtConformalMetric<DataVector> /*meta*/)
    const {
  const auto& longitudinal_shift_excess = cache->get_var(
      *this, Tags::LongitudinalShiftExcess<DataVector, 3, Frame::Inertial>{});
  tenex::evaluate<ti::I, ti::J>(
      longitudinal_shift_minus_dt_conformal_metric,
      longitudinal_shift_excess(ti::I, ti::J) +
          longitudinal_shift_background_minus_dt_conformal_metric(ti::I,
                                                                  ti::J));
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::ii<DataVector, 3>*> extrinsic_curvature,
    const gsl::not_null<Cache*> cache,
    gr::Tags::ExtrinsicCurvature<DataVector, 3> /*meta*/) const {
  const auto& lapse = cache->get_var(*this, gr::Tags::Lapse<DataVector>{});
  const auto& longitudinal_shift_minus_dt_conformal_metric = cache->get_var(
      *this, detail::LongitudinalShiftMinusDtConformalMetric<DataVector>{});
  Xcts::extrinsic_curvature(
      extrinsic_curvature, conformal_factor, lapse, conformal_metric,
      longitudinal_shift_minus_dt_conformal_metric, trace_extrinsic_curvature);
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    const gsl::not_null<Cache*> cache,
    gr::Tags::HamiltonianConstraint<DataVector> /*meta*/) const {
  const auto& conformal_laplacian_of_conformal_factor = cache->get_var(
      *this, detail::ConformalLaplacianOfConformalFactor<DataVector>{});
  const auto& inv_spatial_metric =
      cache->get_var(*this, gr::Tags::InverseSpatialMetric<DataVector, 3>{});
  const auto& extrinsic_curvature =
      cache->get_var(*this, gr::Tags::ExtrinsicCurvature<DataVector, 3>{});
  // Eq. 3.12 in BaumgarteShapiro, divided by 2 for consistency with SpEC
  tenex::evaluate(hamiltonian_constraint,
                  4. * conformal_laplacian_of_conformal_factor() -
                      0.5 * (conformal_factor() * conformal_ricci_scalar() +
                             pow<5>(conformal_factor()) *
                                 (square(trace_extrinsic_curvature()) -
                                  inv_spatial_metric(ti::I, ti::K) *
                                      inv_spatial_metric(ti::J, ti::L) *
                                      extrinsic_curvature(ti::i, ti::j) *
                                      extrinsic_curvature(ti::k, ti::l) -
                                  16. * M_PI * energy_density())));
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::I<DataVector, 3>*> momentum_constraint,
    const gsl::not_null<Cache*> cache,
    gr::Tags::MomentumConstraint<DataVector, 3> /*meta*/) const {
  const auto& deriv_conformal_factor =
      cache->get_var(*this, ::Tags::deriv<Tags::ConformalFactor<DataVector>,
                                          tmpl::size_t<3>, Frame::Inertial>{});
  const auto& deriv_lapse_times_conformal_factor = cache->get_var(
      *this, ::Tags::deriv<Tags::LapseTimesConformalFactor<DataVector>,
                           tmpl::size_t<3>, Frame::Inertial>{});
  const auto& div_longitudinal_shift_excess = cache->get_var(
      *this,
      ::Tags::div<
          Tags::LongitudinalShiftExcess<DataVector, 3, Frame::Inertial>>{});
  const auto& longitudinal_shift_minus_dt_conformal_metric = cache->get_var(
      *this, detail::LongitudinalShiftMinusDtConformalMetric<DataVector>{});
  // Eq. 3.109 in BaumgarteShapiro
  tenex::evaluate<ti::I>(
      momentum_constraint,
      0.5 * (div_longitudinal_shift_excess(ti::I) +
             div_longitudinal_shift_background_minus_dt_conformal_metric(
                 ti::I) +
             conformal_christoffel_second_kind(ti::I, ti::j, ti::k) *
                 longitudinal_shift_minus_dt_conformal_metric(ti::J, ti::K) +
             conformal_christoffel_contracted(ti::j) *
                 longitudinal_shift_minus_dt_conformal_metric(ti::I, ti::J) -
             longitudinal_shift_minus_dt_conformal_metric(ti::I, ti::J) *
                 (deriv_lapse_times_conformal_factor(ti::j) /
                      lapse_times_conformal_factor() -
                  7. * deriv_conformal_factor(ti::j) / conformal_factor()) -
             4. / 3. * lapse_times_conformal_factor() / conformal_factor() *
                 inv_conformal_metric(ti::I, ti::J) *
                 deriv_trace_extrinsic_curvature(ti::j)) -
          8. * M_PI * lapse_times_conformal_factor() *
              cube(conformal_factor()) * momentum_density(ti::I));
}

}  // namespace Xcts
