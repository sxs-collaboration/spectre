// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts {

namespace detail {
template <typename DataType>
struct ConformalLaplacianOfConformalFactor : db::SimpleTag {
  using type = Scalar<DataType>;
};
template <typename DataType>
struct LongitudinalShiftMinusDtConformalMetric : db::SimpleTag {
  using type = tnsr::II<DataType, 3>;
};
}  // namespace detail

/// General-relativistic 3+1 quantities computed from XCTS variables.
using SpacetimeQuantities = CachedTempBuffer<
    // Derivatives of XCTS variables
    ::Tags::deriv<Tags::ConformalFactor<DataVector>, tmpl::size_t<3>,
                  Frame::Inertial>,
    detail::ConformalLaplacianOfConformalFactor<DataVector>,
    ::Tags::deriv<Tags::LapseTimesConformalFactor<DataVector>, tmpl::size_t<3>,
                  Frame::Inertial>,
    ::Tags::deriv<Tags::ShiftExcess<DataVector, 3, Frame::Inertial>,
                  tmpl::size_t<3>, Frame::Inertial>,
    Xcts::Tags::ShiftStrain<DataVector, 3, Frame::Inertial>,
    Xcts::Tags::LongitudinalShiftExcess<DataVector, 3, Frame::Inertial>,
    ::Tags::div<Tags::LongitudinalShiftExcess<DataVector, 3, Frame::Inertial>>,
    detail::LongitudinalShiftMinusDtConformalMetric<DataVector>,
    // ADM quantities
    gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
    gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>,
    gr::Tags::Lapse<DataVector>,
    gr::Tags::Shift<3, Frame::Inertial, DataVector>,
    gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>,
    // Constraints
    gr::Tags::HamiltonianConstraint<DataVector>,
    gr::Tags::MomentumConstraint<3, Frame::Inertial, DataVector>>;

/// `CachedTempBuffer` computer class for 3+1 quantities from XCTS variables.
/// See `Xcts::SpacetimeQuantities`.
struct SpacetimeQuantitiesComputer {
  using Cache = SpacetimeQuantities;

  void operator()(
      gsl::not_null<tnsr::ii<DataVector, 3>*> spatial_metric,
      gsl::not_null<Cache*> cache,
      gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::II<DataVector, 3>*> inv_spatial_metric,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::InverseSpatialMetric<3, Frame::Inertial,
                                                 DataVector> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::i<DataVector, 3>*> deriv_conformal_factor,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Tags::ConformalFactor<DataVector>, tmpl::size_t<3>,
                    Frame::Inertial> /*meta*/) const;
  void operator()(
      gsl::not_null<Scalar<DataVector>*>
          conformal_laplacian_of_conformal_factor,
      gsl::not_null<Cache*> cache,
      detail::ConformalLaplacianOfConformalFactor<DataVector> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::i<DataVector, 3>*> deriv_lapse_times_conformal_factor,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Tags::LapseTimesConformalFactor<DataVector>,
                    tmpl::size_t<3>, Frame::Inertial> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataVector>*> lapse,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::Lapse<DataVector> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::I<DataVector, 3>*> shift, gsl::not_null<Cache*> cache,
      gr::Tags::Shift<3, Frame::Inertial, DataVector> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::iJ<DataVector, 3>*> deriv_shift_excess,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Tags::ShiftExcess<DataVector, 3, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::ii<DataVector, 3>*> shift_strain,
      gsl::not_null<Cache*> cache,
      Tags::ShiftStrain<DataVector, 3, Frame::Inertial> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::II<DataVector, 3>*> longitudinal_shift_excess,
      gsl::not_null<Cache*> cache,
      Tags::LongitudinalShiftExcess<DataVector, 3, Frame::Inertial> /*meta*/)
      const;
  void operator()(
      gsl::not_null<tnsr::I<DataVector, 3>*> div_longitudinal_shift_excess,
      gsl::not_null<Cache*> cache,
      ::Tags::div<Tags::LongitudinalShiftExcess<
          DataVector, 3, Frame::Inertial>> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::II<DataVector, 3>*>
          longitudinal_shift_minus_dt_conformal_metric,
      gsl::not_null<Cache*> cache,
      detail::LongitudinalShiftMinusDtConformalMetric<DataVector> /*meta*/)
      const;
  void operator()(gsl::not_null<tnsr::ii<DataVector, 3>*> extrinsic_curvature,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::ExtrinsicCurvature<3, Frame::Inertial,
                                               DataVector> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::HamiltonianConstraint<DataVector> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::I<DataVector, 3>*> momentum_constraint,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::MomentumConstraint<3, Frame::Inertial,
                                               DataVector> /*meta*/) const;

  // XCTS variables
  const Scalar<DataVector>& conformal_factor;
  const Scalar<DataVector>& lapse_times_conformal_factor;
  const tnsr::I<DataVector, 3>& shift_excess;
  // Background
  const tnsr::ii<DataVector, 3>& conformal_metric;
  const tnsr::II<DataVector, 3>& inv_conformal_metric;
  const tnsr::ijj<DataVector, 3>& deriv_conformal_metric;
  const tnsr::ijj<DataVector, 3>& conformal_christoffel_first_kind;
  const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind;
  const tnsr::i<DataVector, 3>& conformal_christoffel_contracted;
  const Scalar<DataVector>& conformal_ricci_scalar;
  const Scalar<DataVector>& trace_extrinsic_curvature;
  const tnsr::i<DataVector, 3>& deriv_trace_extrinsic_curvature;
  const tnsr::I<DataVector, 3>& shift_background;
  const tnsr::II<DataVector, 3>&
      longitudinal_shift_background_minus_dt_conformal_metric;
  const tnsr::I<DataVector, 3>&
      div_longitudinal_shift_background_minus_dt_conformal_metric;
  const Scalar<DataVector>& energy_density;
  const tnsr::I<DataVector, 3>& momentum_density;
  // Grid
  const Mesh<3>& mesh;
  const InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>&
      inv_jacobian;
};

namespace Tags {
/// Compute tag for the 3+1 quantities `Tags` from XCTS variables. The `Tags`
/// can be any subset of the tags supported by `Xcts::SpacetimeQuantities`.
template <typename Tags>
struct SpacetimeQuantitiesCompute : ::Tags::Variables<Tags>, db::ComputeTag {
  using base = ::Tags::Variables<Tags>;
  using argument_tags = tmpl::list<
      domain::Tags::Mesh<3>, ConformalFactor<DataVector>,
      LapseTimesConformalFactor<DataVector>,
      ShiftExcess<DataVector, 3, Frame::Inertial>,
      ConformalMetric<DataVector, 3, Frame::Inertial>,
      InverseConformalMetric<DataVector, 3, Frame::Inertial>,
      ::Tags::deriv<ConformalMetric<DataVector, 3, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial>,
      ConformalChristoffelFirstKind<DataVector, 3, Frame::Inertial>,
      ConformalChristoffelSecondKind<DataVector, 3, Frame::Inertial>,
      ConformalChristoffelContracted<DataVector, 3, Frame::Inertial>,
      ConformalRicciScalar<DataVector>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>,
      ShiftBackground<DataVector, 3, Frame::Inertial>,
      LongitudinalShiftBackgroundMinusDtConformalMetric<DataVector, 3,
                                                        Frame::Inertial>,
      ::Tags::div<LongitudinalShiftBackgroundMinusDtConformalMetric<
          DataVector, 3, Frame::Inertial>>,
      gr::Tags::Conformal<gr::Tags::EnergyDensity<DataVector>, 0>,
      gr::Tags::Conformal<
          gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>, 0>,
      domain::Tags::Mesh<3>,
      domain::Tags::InverseJacobian<3, Frame::ElementLogical, Frame::Inertial>>;
  template <typename... Args>
  static void function(const gsl::not_null<typename base::type*> result,
                       const Mesh<3>& mesh, const Args&... args) {
    const size_t num_points = mesh.number_of_grid_points();
    if (result->number_of_grid_points() != num_points) {
      result->initialize(num_points);
    }
    SpacetimeQuantities spacetime_quantities{num_points};
    const SpacetimeQuantitiesComputer computer{args...};
    tmpl::for_each<Tags>(
        [&spacetime_quantities, &computer, &result](const auto tag_v) {
          using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
          get<tag>(*result) = spacetime_quantities.get_var(computer, tag{});
        });
  }
};
}  // namespace Tags

}  // namespace Xcts
