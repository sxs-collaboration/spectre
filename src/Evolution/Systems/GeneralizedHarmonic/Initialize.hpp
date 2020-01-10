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
    using compute_tags = db::AddComputeTags<
        GeneralizedHarmonic::Tags::GaugeConstraintCompute<Dim, frame>,
        GeneralizedHarmonic::Tags::FourIndexConstraintCompute<Dim, frame>,
        // following tags added to observe constraints
        ::Tags::PointwiseL2NormCompute<
            GeneralizedHarmonic::Tags::GaugeConstraint<Dim, frame>>,
        ::Tags::PointwiseL2NormCompute<
            GeneralizedHarmonic::Tags::ThreeIndexConstraint<Dim, frame>>,
        ::Tags::PointwiseL2NormCompute<
            GeneralizedHarmonic::Tags::FourIndexConstraint<Dim, frame>>>;

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
struct InitializeGauge {
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
    const auto& mesh = db::get<::Tags::Mesh<Dim>>(box);
    const size_t num_grid_points = mesh.number_of_grid_points();
    const auto& lapse = get<gr::Tags::Lapse<DataVector>>(box);
    const auto& dt_lapse = get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(box);
    const auto& deriv_lapse = get<
        ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<Dim>, frame>>(
        box);
    const auto& shift = get<gr::Tags::Shift<Dim, frame, DataVector>>(box);
    const auto& dt_shift =
        get<::Tags::dt<gr::Tags::Shift<Dim, frame, DataVector>>>(box);
    const auto& deriv_shift =
        get<::Tags::deriv<gr::Tags::Shift<Dim, frame, DataVector>,
                          tmpl::size_t<Dim>, frame>>(box);
    const auto& spatial_metric =
        get<gr::Tags::SpatialMetric<Dim, frame, DataVector>>(box);
    const auto& trace_extrinsic_curvature =
        get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(box);
    const auto& trace_christoffel_last_indices =
        get<gr::Tags::TraceSpatialChristoffelFirstKind<Dim, frame, DataVector>>(
            box);

    // compute the initial gauge source function
    auto initial_gauge_h = GeneralizedHarmonic::gauge_source<Dim, frame>(
        lapse, dt_lapse, deriv_lapse, shift, dt_shift, deriv_shift,
        spatial_metric, trace_extrinsic_curvature,
        trace_christoffel_last_indices);
    // set time derivatives of InitialGaugeH = 0
    // NOTE: this will need to be generalized to handle numerical initial data
    // and analytic initial data whose gauge is not initially stationary.
    auto dt_initial_gauge_source =
        make_with_value<tnsr::a<DataVector, Dim, frame>>(lapse, 0.);

    // compute spatial derivatives of InitialGaugeH
    // The `partial_derivatives` function does not support single Tensor input,
    // and so we must store the tensor in a Variables first.
    using InitialGaugeHVars = ::Variables<
        tmpl::list<GeneralizedHarmonic::Tags::InitialGaugeH<Dim, frame>>>;
    InitialGaugeHVars initial_gauge_h_vars{num_grid_points};

    get<GeneralizedHarmonic::Tags::InitialGaugeH<Dim, frame>>(
        initial_gauge_h_vars) = initial_gauge_h;
    const auto& inverse_jacobian =
        db::get<::Tags::InverseJacobian<Dim, Frame::Logical, frame>>(box);
    auto d_initial_gauge_source =
        get<::Tags::deriv<GeneralizedHarmonic::Tags::InitialGaugeH<Dim, frame>,
                          tmpl::size_t<Dim>, frame>>(
            partial_derivatives<typename InitialGaugeHVars::tags_list>(
                initial_gauge_h_vars, mesh, inverse_jacobian));

    // compute spacetime derivatives of InitialGaugeH
    auto initial_d4_gauge_h =
        GeneralizedHarmonic::Tags::SpacetimeDerivGaugeHCompute<
            Dim, frame>::function(std::move(dt_initial_gauge_source),
                                  std::move(d_initial_gauge_source));
    // Add gauge tags
    using compute_tags = db::AddComputeTags<
        GeneralizedHarmonic::DampedHarmonicHCompute<Dim, frame>,
        GeneralizedHarmonic::SpacetimeDerivDampedHarmonicHCompute<Dim, frame>>;

    // Finally, insert gauge related quantities to the box
    return std::make_tuple(
        Initialization::merge_into_databox<
            InitializeGauge,
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
