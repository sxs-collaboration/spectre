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
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Initialization/DiscontinuousGalerkin.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Systems/CurvedScalarWave/Characteristics.hpp"
#include "Evolution/Systems/CurvedScalarWave/Constraints.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace CurvedScalarWave {
namespace Actions {
template <size_t Dim>
struct InitializeGrVars {
  using frame = Frame::Inertial;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    static constexpr size_t dim = system::volume_dim;
    using gr_tag = typename system::spacetime_variables_tag;
    using simple_tags = db::AddSimpleTags<gr_tag>;

    const double initial_time = db::get<Initialization::Tags::InitialTime>(box);
    using GrVars = typename gr_tag::type;

    const size_t num_grid_points =
        db::get<domain::Tags::Mesh<dim>>(box).number_of_grid_points();
    const auto inertial_coords =
        db::get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    // Set initial data from analytic solution
    GrVars gr_vars{num_grid_points};
    gr_vars.assign_subset(evolution::initial_data(
        Parallel::get<::Tags::AnalyticSolutionOrData>(cache), inertial_coords,
        initial_time, typename GrVars::tags_list{}));

    using compute_tags = db::AddComputeTags<
        Tags::ConstraintGamma1Compute, Tags::ConstraintGamma2Compute,
        gr::Tags::SpatialChristoffelFirstKindCompute<Dim, Frame::Inertial,
                                                     DataVector>,
        gr::Tags::SpatialChristoffelSecondKindCompute<Dim, Frame::Inertial,
                                                      DataVector>,
        gr::Tags::TraceSpatialChristoffelSecondKindCompute<Dim, Frame::Inertial,
                                                           DataVector>,
        GeneralizedHarmonic::Tags::TraceExtrinsicCurvatureCompute<
            Dim, Frame::Inertial>>;

    return std::make_tuple(
        Initialization::merge_into_databox<InitializeGrVars, simple_tags,
                                           compute_tags>(std::move(box),
                                                         std::move(gr_vars)));
  }
};

template <size_t Dim>
struct InitializeConstraints {
  using frame = Frame::Inertial;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using compute_tags = db::AddComputeTags<
        Tags::OneIndexConstraintCompute<Dim>,
        Tags::TwoIndexConstraintCompute<Dim>,
        // following tags added to observe constraints
        ::Tags::PointwiseL2NormCompute<Tags::OneIndexConstraint<Dim>>,
        ::Tags::PointwiseL2NormCompute<Tags::TwoIndexConstraint<Dim>>>;

    return std::make_tuple(
        Initialization::merge_into_databox<InitializeConstraints,
                                           db::AddSimpleTags<>, compute_tags>(
            std::move(box)));
  }
};

}  // namespace Actions
}  // namespace CurvedScalarWave
