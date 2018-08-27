// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>
#include <utility>

#include "ApparentHorizons/HorizonManagerComponent.hpp"
#include "ApparentHorizons/HorizonManagerComponentActions.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainCreators/DomainCreator.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/InitialElementIds.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/Algorithm.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "PointwiseFunctions/AnalyticSolutions/EinsteinSolutions/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/GrTags.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ApparentHorizons/TestHorizonManagerHelper_ElementActions.hpp"

namespace test_ah {
template <class Metavariables>
struct DgElementArray {
  using chare_type = Parallel::Algorithms::Array;
  using const_global_cache_tag_list = tmpl::list<>;
  using metavariables = Metavariables;
  using action_list = tmpl::list<>;
  using array_index = ElementIndex<3>;
  using initial_databox = db::compute_databox_type<
      typename Actions::DgElementArray::InitializeElement::return_tag_list>;
  using options = tmpl::list<typename Metavariables::domain_creator_tag>;
  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
      std::unique_ptr<DomainCreator<3, Frame::Inertial>>
          domain_creator) noexcept;
  static void execute_next_phase(
      typename Metavariables::Phase next_phase,
      const Parallel::CProxy_ConstGlobalCache<Metavariables>&
          global_cache) noexcept;
};

template <class Metavariables>
void DgElementArray<Metavariables>::execute_next_phase(
    const typename Metavariables::Phase next_phase,
    const Parallel::CProxy_ConstGlobalCache<Metavariables>&
        global_cache) noexcept {
  if (next_phase == Metavariables::Phase::InitialCommunication) {
    auto& dg_element_array = Parallel::get_parallel_component<DgElementArray>(
        *(global_cache.ckLocalBranch()));

    Parallel::simple_action<Actions::DgElementArray::SendNumElements<
        ah::DataInterpolator<Metavariables>>>(dg_element_array);
  }
  if (next_phase == Metavariables::Phase::CheckAnswer) {
    auto& my_proxy =
        Parallel::get_parallel_component<ah::DataInterpolator<Metavariables>>(
            *(global_cache.ckLocalBranch()));
    Parallel::simple_action<ah::Actions::DataInterpolator::PrintNumElements>(
        my_proxy);
  }
  if (next_phase == Metavariables::Phase::BeginHorizonSearch) {
    // Time steps on which to find the horizon
    Slab slab(0.0, 1.0);
    const std::vector<Time> timesteps = {Time(slab, 0), Time(slab, 1)};

    auto& dg_element_array = Parallel::get_parallel_component<DgElementArray>(
        *(global_cache.ckLocalBranch()));

    Parallel::simple_action<Actions::DgElementArray::BeginHorizonSearch<
        ah::DataInterpolator<Metavariables>>>(dg_element_array, timesteps);

    tmpl::for_each<typename Metavariables::horizon_tags>(
        [&global_cache, &timesteps](auto x) {
          using tag = typename decltype(x)::type;
          auto& my_proxy =
              Parallel::get_parallel_component<ah::Finder<Metavariables, tag>>(
                  *(global_cache.ckLocalBranch()));
          Parallel::simple_action<ah::Actions::Finder::AddTimeSteps<tag>>(
              my_proxy, timesteps);
        });
  }
}

template <class Metavariables>
void DgElementArray<Metavariables>::initialize(
    Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
    std::unique_ptr<DomainCreator<3, Frame::Inertial>>
        domain_creator) noexcept {
  auto& dg_element_array = Parallel::get_parallel_component<DgElementArray>(
      *(global_cache.ckLocalBranch()));

  auto domain = domain_creator->create_domain();
  const int number_of_procs = Parallel::number_of_procs();
  for (const auto& block : domain.blocks()) {
    const auto initial_ref_levs =
        domain_creator->initial_refinement_levels()[block.id()];
    const std::vector<ElementId<3>> element_ids =
        initial_element_ids(block.id(), initial_ref_levs);
    int which_proc = 0;
    for (const auto& element_id : element_ids) {
      dg_element_array(ElementIndex<3>(element_id))
          .insert(global_cache, which_proc);
      ++which_proc %= number_of_procs;
    }
  }
  dg_element_array.doneInserting();

  dg_element_array
      .template simple_action<Actions::DgElementArray::InitializeElement>(
          std::make_tuple(domain_creator->initial_extents(),
                          std::move(domain)));
}
}  // namespace test_ah
