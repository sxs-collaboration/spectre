// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "AlgorithmArray.hpp"
#include "Domain/Block.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/InitialElementIds.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/NumericalInitialData.hpp"
#include "Evolution/Tags.hpp"
#include "IO/Importers/VolumeDataReader.hpp"
#include "IO/Importers/VolumeDataReaderActions.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "ParallelAlgorithms/Actions/SetData.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace DgElementArray_detail {

template <typename Metavariables, typename DgElementArray>
using read_element_data_action = importers::ThreadedActions::ReadVolumeData<
    ::OptionTags::NumericalInitialData,
    typename Metavariables::initial_data::import_fields,
    ::Actions::SetData<typename Metavariables::initial_data::import_fields>,
    DgElementArray>;

template <typename Metavariables, typename DgElementArray,
          bool Enable =
              is_numerical_initial_data_v<typename Metavariables::initial_data>>
struct import_numeric_data_cache_tags {
  using type = tmpl::list<>;
};

template <typename Metavariables, typename DgElementArray>
struct import_numeric_data_cache_tags<Metavariables, DgElementArray, true> {
  using type = typename read_element_data_action<
      Metavariables, DgElementArray>::const_global_cache_tags;
};

template <typename Metavariables, typename DgElementArray,
          bool Enable =
              is_numerical_initial_data_v<typename Metavariables::initial_data>>
struct try_import_data {
  static void apply(
      const typename Metavariables::Phase /*next_phase*/,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& /*global_cache*/) {}
};

template <typename Metavariables, typename DgElementArray>
struct try_import_data<Metavariables, DgElementArray, true> {
  static void apply(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    if (next_phase == Metavariables::Phase::ImportData) {
      auto& local_cache = *(global_cache.ckLocalBranch());
      Parallel::threaded_action<
          read_element_data_action<Metavariables, DgElementArray>>(
          Parallel::get_parallel_component<
              importers::VolumeDataReader<Metavariables>>(local_cache));
    }
  }
};

}  // namespace DgElementArray_detail

/*!
 * \brief The parallel component responsible for managing the DG elements that
 * compose the computational domain
 *
 * This parallel component will perform the actions specified by the
 * `PhaseDepActionList`.
 *
 */
template <class Metavariables, class PhaseDepActionList>
struct DgElementArray {
  static constexpr size_t volume_dim = Metavariables::volume_dim;

  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using phase_dependent_action_list = PhaseDepActionList;
  using array_index = ElementIndex<volume_dim>;

  using const_global_cache_tags = tmpl::list<
      tmpl::type_from<DgElementArray_detail::import_numeric_data_cache_tags<
          Metavariables, DgElementArray>>,
      tmpl::list<::Tags::Domain<volume_dim>>>;

  using array_allocation_tags =
      tmpl::list<::Tags::InitialRefinementLevels<volume_dim>>;

  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>,
      array_allocation_tags>;

  static void allocate_array(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<initialization_tags>&
          initialization_items) noexcept;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<DgElementArray>(local_cache)
        .start_phase(next_phase);

    DgElementArray_detail::try_import_data<Metavariables,
                                           DgElementArray>::apply(next_phase,
                                                                  global_cache);
  }
};

template <class Metavariables, class PhaseDepActionList>
void DgElementArray<Metavariables, PhaseDepActionList>::allocate_array(
    Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
    const tuples::tagged_tuple_from_typelist<initialization_tags>&
        initialization_items) noexcept {
  auto& local_cache = *(global_cache.ckLocalBranch());
  auto& dg_element_array =
      Parallel::get_parallel_component<DgElementArray>(local_cache);
  const auto& domain = Parallel::get<::Tags::Domain<volume_dim>>(local_cache);
  const auto& initial_refinement_levels =
      get<::Tags::InitialRefinementLevels<volume_dim>>(initialization_items);
  for (const auto& block : domain.blocks()) {
    const auto initial_ref_levs = initial_refinement_levels[block.id()];
    const std::vector<ElementId<volume_dim>> element_ids =
        initial_element_ids(block.id(), initial_ref_levs);
    int which_proc = 0;
    const int number_of_procs = Parallel::number_of_procs();
    for (size_t i = 0; i < element_ids.size(); ++i) {
      dg_element_array(ElementIndex<volume_dim>(element_ids[i]))
          .insert(global_cache, initialization_items, which_proc);
      which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
    }
  }
  dg_element_array.doneInserting();
}
