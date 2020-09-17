// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic {
namespace Actions {

struct InitializeFixedSources {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& cache,
                    const ElementId<Dim>& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    using fields_tag = typename system::fields_tag;
    using fixed_sources_tag =
        db::add_tag_prefix<::Tags::FixedSource, fields_tag>;

    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const size_t num_grid_points = mesh.number_of_grid_points();
    const auto& inertial_coords =
        get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box);

    // Retrieve the sources of the elliptic system from the analytic solution,
    // which defines the problem we want to solve.
    // We need only retrieve sources for the primal fields, since the auxiliary
    // fields will never be sourced.
    typename fixed_sources_tag::type fixed_sources{num_grid_points, 0.};
    fixed_sources.assign_subset(
        Parallel::get<typename Metavariables::analytic_solution_tag>(cache)
            .variables(inertial_coords,
                       db::wrap_tags_in<::Tags::FixedSource,
                                        typename system::primal_fields>{}));

    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeFixedSources, db::AddSimpleTags<fixed_sources_tag>>(
            std::move(box), std::move(fixed_sources)));
  }
};

}  // namespace Actions
}  // namespace elliptic
