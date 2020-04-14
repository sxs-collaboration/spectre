// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>  // IWYU pragma: keep
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Initialization/DiscontinuousGalerkin.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace GeneralizedHarmonic {
namespace Actions {
template <size_t Dim>
struct InitializeConstraints {
  using frame = Frame::Inertial;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using compute_tags = tmpl::flatten<db::AddComputeTags<
        GeneralizedHarmonic::Tags::GaugeConstraintCompute<Dim, frame>,
        // following tags added to observe constraints
        ::Tags::PointwiseL2NormCompute<
            GeneralizedHarmonic::Tags::GaugeConstraint<Dim, frame>>,
        ::Tags::PointwiseL2NormCompute<
            GeneralizedHarmonic::Tags::ThreeIndexConstraint<Dim, frame>>,
        // The 4-index constraint is only implemented in 3d
        tmpl::conditional_t<
            Dim == 3,
            tmpl::list<GeneralizedHarmonic::Tags::FourIndexConstraintCompute<
                           Dim, frame>,
                       ::Tags::PointwiseL2NormCompute<
                           GeneralizedHarmonic::Tags::FourIndexConstraint<
                               Dim, frame>>>,
            tmpl::list<>>>>;

    return std::make_tuple(
        Initialization::merge_into_databox<InitializeConstraints,
                                           db::AddSimpleTags<>, compute_tags>(
            std::move(box)));
  }
};

template <size_t Dim>
struct InitializeGhAnd3Plus1Variables {
  using frame = Frame::Inertial;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using compute_tags = db::AddComputeTags<
        gr::Tags::SpatialMetricCompute<Dim, frame, DataVector>,
        gr::Tags::DetAndInverseSpatialMetricCompute<Dim, frame, DataVector>,
        gr::Tags::ShiftCompute<Dim, frame, DataVector>,
        gr::Tags::LapseCompute<Dim, frame, DataVector>,
        gr::Tags::SqrtDetSpatialMetricCompute<Dim, frame, DataVector>,
        gr::Tags::SpacetimeNormalOneFormCompute<Dim, frame, DataVector>,
        gr::Tags::SpacetimeNormalVectorCompute<Dim, frame, DataVector>,
        gr::Tags::InverseSpacetimeMetricCompute<Dim, frame, DataVector>,
        GeneralizedHarmonic::Tags::DerivSpatialMetricCompute<Dim, frame>,
        GeneralizedHarmonic::Tags::DerivLapseCompute<Dim, frame>,
        GeneralizedHarmonic::Tags::DerivShiftCompute<Dim, frame>,
        GeneralizedHarmonic::Tags::TimeDerivSpatialMetricCompute<Dim, frame>,
        GeneralizedHarmonic::Tags::TimeDerivLapseCompute<Dim, frame>,
        GeneralizedHarmonic::Tags::TimeDerivShiftCompute<Dim, frame>,
        gr::Tags::DerivativesOfSpacetimeMetricCompute<Dim, frame>,
        gr::Tags::DerivSpacetimeMetricCompute<Dim, frame>,
        GeneralizedHarmonic::Tags::ThreeIndexConstraintCompute<Dim, frame>,
        gr::Tags::SpacetimeChristoffelFirstKindCompute<Dim, frame, DataVector>,
        gr::Tags::SpacetimeChristoffelSecondKindCompute<Dim, frame, DataVector>,
        gr::Tags::TraceSpacetimeChristoffelFirstKindCompute<Dim, frame,
                                                            DataVector>,
        gr::Tags::SpatialChristoffelFirstKindCompute<Dim, frame, DataVector>,
        gr::Tags::SpatialChristoffelSecondKindCompute<Dim, frame, DataVector>,
        gr::Tags::TraceSpatialChristoffelFirstKindCompute<Dim, frame,
                                                          DataVector>,
        GeneralizedHarmonic::Tags::ExtrinsicCurvatureCompute<Dim, frame>,
        GeneralizedHarmonic::Tags::TraceExtrinsicCurvatureCompute<Dim, frame>,
        GeneralizedHarmonic::Tags::ConstraintGamma0Compute<Dim, frame>,
        GeneralizedHarmonic::Tags::ConstraintGamma1Compute<Dim, frame>,
        GeneralizedHarmonic::Tags::ConstraintGamma2Compute<Dim, frame>>;

    return std::make_tuple(
        Initialization::merge_into_databox<InitializeGhAnd3Plus1Variables,
                                           db::AddSimpleTags<>, compute_tags>(
            std::move(box)));
  }
};

template <size_t Dim>
struct InitializeDampedHarmonicRollonGauge {
  using frame = Frame::Inertial;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    // compute initial-gauge related quantities
    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const size_t num_grid_points = mesh.number_of_grid_points();

    const auto& spacetime_metric =
        db::get<gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>>(
            box);
    const auto& pi =
        db::get<GeneralizedHarmonic::Tags::Pi<Dim, Frame::Inertial>>(box);
    const auto& phi =
        db::get<GeneralizedHarmonic::Tags::Phi<Dim, Frame::Inertial>>(box);

    const auto spatial_metric = gr::spatial_metric(spacetime_metric);
    const auto inverse_spatial_metric =
        determinant_and_inverse(spatial_metric).second;
    const auto shift = gr::shift(spacetime_metric, inverse_spatial_metric);
    const auto lapse = gr::lapse(shift, spacetime_metric);
    const auto inverse_spacetime_metric =
        gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);
    tnsr::abb<DataVector, Dim, Frame::Inertial> da_spacetime_metric{};
    GeneralizedHarmonic::spacetime_derivative_of_spacetime_metric(
        make_not_null(&da_spacetime_metric), lapse, shift, pi, phi);
    // H_a=-Gamma_a
    auto initial_gauge_h =
        trace_last_indices(gr::christoffel_first_kind(da_spacetime_metric),
                           inverse_spacetime_metric);
    for (size_t i = 0; i < initial_gauge_h.size(); ++i) {
      initial_gauge_h[i] *= -1.0;
    }

    // set time derivatives of InitialGaugeH = 0
    // NOTE: this will need to be generalized to handle numerical initial data
    // and analytic initial data whose gauge is not initially stationary.
    auto dt_initial_gauge_source =
        make_with_value<tnsr::a<DataVector, Dim, frame>>(initial_gauge_h, 0.);

    // compute spatial derivatives of InitialGaugeH
    // The `partial_derivatives` function does not support single Tensor input,
    // and so we must store the tensor in a Variables first.
    using InitialGaugeHVars = ::Variables<
        tmpl::list<GeneralizedHarmonic::Tags::InitialGaugeH<Dim, frame>>>;
    InitialGaugeHVars initial_gauge_h_vars{num_grid_points};

    get<GeneralizedHarmonic::Tags::InitialGaugeH<Dim, frame>>(
        initial_gauge_h_vars) = initial_gauge_h;
    const auto& inverse_jacobian =
        db::get<domain::Tags::InverseJacobian<Dim, Frame::Logical, frame>>(box);
    auto d_initial_gauge_source =
        get<::Tags::deriv<GeneralizedHarmonic::Tags::InitialGaugeH<Dim, frame>,
                          tmpl::size_t<Dim>, frame>>(
            partial_derivatives<typename InitialGaugeHVars::tags_list>(
                initial_gauge_h_vars, mesh, inverse_jacobian));

    // compute spacetime derivatives of InitialGaugeH
    tnsr::ab<DataVector, Dim, Frame::Inertial> initial_d4_gauge_h{};
    GeneralizedHarmonic::Tags::SpacetimeDerivGaugeHCompute<
        Dim, frame>::function(make_not_null(&initial_d4_gauge_h),
                              std::move(dt_initial_gauge_source),
                              std::move(d_initial_gauge_source));
    // Add gauge tags
    using compute_tags = db::AddComputeTags<
        GeneralizedHarmonic::gauges::DampedHarmonicHCompute<Dim, frame>,
        GeneralizedHarmonic::gauges::SpacetimeDerivDampedHarmonicHCompute<
            Dim, frame>>;

    // Finally, insert gauge related quantities to the box
    return std::make_tuple(
        Initialization::merge_into_databox<
            InitializeDampedHarmonicRollonGauge,
            db::AddSimpleTags<
                GeneralizedHarmonic::Tags::InitialGaugeH<Dim, frame>,
                GeneralizedHarmonic::Tags::SpacetimeDerivInitialGaugeH<Dim,
                                                                       frame>>,
            compute_tags>(std::move(box), std::move(initial_gauge_h),
                          std::move(initial_d4_gauge_h)));
  }
};

}  // namespace Actions
}  // namespace GeneralizedHarmonic
