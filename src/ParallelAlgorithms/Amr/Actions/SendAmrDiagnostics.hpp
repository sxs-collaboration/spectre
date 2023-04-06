// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/rational.hpp>
#include <cstddef>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Amr/Helpers.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Amr/Actions/Component.hpp"
#include "ParallelAlgorithms/Amr/Actions/RunAmrDiagnostics.hpp"
#include "Utilities/Serialization/PupBoost.hpp"

namespace amr::Actions {

/// \brief Send AMR diagnostics about Element%s to amr::Component
///
/// Sends the following:
/// - The fraction of a Block volume (in the logical coordinate frame) covered
///   by the Element
/// - One, in order to count the number of Element%s
/// - The number of grid points
/// - The refinement level in each logical dimension
/// - The number of grid points in each logical dimension
///
/// The information is sent to amr::Component which runs the action
/// amr::Actions::RunAmrDiagnostics after all Element%s have contributed to the
/// reduction
struct SendAmrDiagnostics {
  using ReductionData = Parallel::ReductionData<
      // fraction of Block volume
      Parallel::ReductionDatum<boost::rational<size_t>, funcl::Plus<>>,
      // number of elements
      Parallel::ReductionDatum<size_t, funcl::Plus<>>,
      // number of grid points
      Parallel::ReductionDatum<size_t, funcl::Plus<>>,
      // average refinement level by dimension
      Parallel::ReductionDatum<std::vector<size_t>,
                               funcl::ElementWise<funcl::Plus<>>>,
      // average number of grid points by dimension
      Parallel::ReductionDatum<std::vector<size_t>,
                               funcl::ElementWise<funcl::Plus<>>>>;

  template <typename DbTagList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* /*meta*/) {
    constexpr size_t volume_dim = Metavariables::volume_dim;
    const ElementId<volume_dim> element_id{array_index};
    const auto& mesh = db::get<::domain::Tags::Mesh<volume_dim>>(box);
    const auto& my_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index];
    const auto& target_proxy =
        Parallel::get_parallel_component<amr::Component<Metavariables>>(cache);
    std::vector<size_t> refinement_levels_by_dim(volume_dim);
    std::vector<size_t> extents_by_dim(volume_dim);
    const auto refinement_levels = element_id.refinement_levels();
    size_t number_of_grid_points = 1;
    for (size_t d = 0; d < volume_dim; ++d) {
      refinement_levels_by_dim[d] = gsl::at(refinement_levels, d);
      extents_by_dim[d] = mesh.extents(d);
      number_of_grid_points *= mesh.extents(d);
    }
    Parallel::contribute_to_reduction<amr::Actions::RunAmrDiagnostics>(
        ReductionData{amr::fraction_of_block_volume(element_id), 1,
                      number_of_grid_points, refinement_levels_by_dim,
                      extents_by_dim},
        my_proxy, target_proxy);
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace amr::Actions
