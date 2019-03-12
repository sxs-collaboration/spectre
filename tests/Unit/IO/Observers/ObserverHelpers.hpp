// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <functional>

#include "Domain/ElementIndex.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "IO/Observer/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace TestObservers_detail {
using ElementIndexType = ElementIndex<2>;

template <typename Metavariables>
struct element_component {
  using component_being_mocked = void;  // Not needed
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndexType;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<>;
  using initial_databox = db::DataBox<tmpl::list<>>;
};

template <typename Metavariables>
struct observer_component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<>;
  using component_being_mocked = observers::Observer<Metavariables>;
  using simple_tags =
      typename observers::Actions::Initialize<Metavariables>::simple_tags;
  using compute_tags =
      typename observers::Actions::Initialize<Metavariables>::compute_tags;
  using initial_databox =
      db::compute_databox_type<tmpl::append<simple_tags, compute_tags>>;
};

template <typename Metavariables>
struct observer_writer_component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list =
      tmpl::list<observers::OptionTags::ReductionFileName,
                 observers::OptionTags::VolumeFileName>;
  using action_list = tmpl::list<>;
  using component_being_mocked = observers::ObserverWriter<Metavariables>;
  using simple_tags =
      typename observers::Actions::InitializeWriter<Metavariables>::simple_tags;
  using compute_tags = typename observers::Actions::InitializeWriter<
      Metavariables>::compute_tags;
  using initial_databox =
      db::compute_databox_type<tmpl::append<simple_tags, compute_tags>>;
};

using l2_error_datum = Parallel::ReductionDatum<double, funcl::Plus<>,
                                                funcl::Sqrt<funcl::Divides<>>,
                                                std::index_sequence<1>>;
using reduction_data = Parallel::ReductionData<
    Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
    Parallel::ReductionDatum<size_t, funcl::Plus<>>, l2_error_datum,
    l2_error_datum>;

struct Metavariables {
  using component_list = tmpl::list<element_component<Metavariables>,
                                    observer_component<Metavariables>,
                                    observer_writer_component<Metavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;

  struct ObservationType {};
  using element_observation_type = ObservationType;

  /// [make_reduction_data_tags]
  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<reduction_data>>;
  /// [make_reduction_data_tags]

  enum class Phase { Initialize, Exit };
};
}  // namespace TestObservers_detail
