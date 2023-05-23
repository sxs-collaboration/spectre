// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/rational.hpp>
#include <cstddef>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Amr/Helpers.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Amr/Actions/Component.hpp"
#include "ParallelAlgorithms/Amr/Actions/RunAmrDiagnostics.hpp"
#include "ParallelAlgorithms/Amr/Tags.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Serialization/PupBoost.hpp"
#include "Utilities/StdHelpers.hpp"

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
  using const_global_cache_tags =
      tmpl::list<logging::Tags::Verbosity<amr::OptionTags::AmrGroup>>;

  using ReductionData = Parallel::ReductionData<
      // fraction of Block volume
      Parallel::ReductionDatum<boost::rational<size_t>, funcl::Plus<>>,
      // number of elements
      Parallel::ReductionDatum<size_t, funcl::Plus<>>,
      // number of grid points
      Parallel::ReductionDatum<size_t, funcl::Plus<>>,
      // average refinement level by dimension
      Parallel::ReductionDatum<
          std::vector<double>, funcl::ElementWise<funcl::Plus<>>,
          funcl::ElementWise<funcl::Divides<>>, std::index_sequence<1>>,
      // average number of grid points by dimension
      Parallel::ReductionDatum<
          std::vector<double>, funcl::ElementWise<funcl::Plus<>>,
          funcl::ElementWise<funcl::Divides<>>, std::index_sequence<1>>>;

  template <typename DbTagList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* /*meta*/) {
    const auto& mesh = db::get<::domain::Tags::Mesh<Dim>>(box);
    const auto& my_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache)[element_id];
    const auto& target_proxy =
        Parallel::get_parallel_component<amr::Component<Metavariables>>(cache);
    std::vector<double> refinement_levels_by_dim(Dim);
    std::vector<double> extents_by_dim(Dim);
    const auto refinement_levels = element_id.refinement_levels();
    for (size_t d = 0; d < Dim; ++d) {
      refinement_levels_by_dim[d] = gsl::at(refinement_levels, d);
      extents_by_dim[d] = mesh.extents(d);
    }
    if (db::get<logging::Tags::Verbosity<amr::OptionTags::AmrGroup>>(box) >=
        Verbosity::Debug) {
      Parallel::printf("%s h-refinement %s, p-refinement %s\n",
                       get_output(element_id), get_output(refinement_levels),
                       get_output(mesh.extents()));
    }
    Parallel::contribute_to_reduction<amr::Actions::RunAmrDiagnostics>(
        ReductionData{amr::fraction_of_block_volume(element_id), 1,
                      mesh.number_of_grid_points(), refinement_levels_by_dim,
                      extents_by_dim},
        my_proxy, target_proxy);
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace amr::Actions
