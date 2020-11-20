// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <functional>

#include "Domain/Structure/ElementId.hpp"
#include "Framework/ActionTesting.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/Initialize.hpp"         // IWYU pragma: keep
#include "IO/Observer/ObservationId.hpp"      // IWYU pragma: keep
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
/// \endcond

namespace TestObservers_detail {
using ElementIdType = ElementId<2>;

template <observers::TypeOfObservation TypeOfObservation>
struct RegisterObservers {
  template <typename ParallelComponent, typename DbTagsList,
            typename ArrayIndex>
  static std::pair<observers::TypeOfObservation, observers::ObservationKey>
  register_info(const db::DataBox<DbTagsList>& /*box*/,
                const ArrayIndex& /*array_index*/) noexcept {
    return {TypeOfObservation,
            observers::ObservationKey{"ElementObservationType"}};
  }
};

template <typename Metavariables, typename RegistrationActionsList>
struct element_component {
  using component_being_mocked = void;  // Not needed
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIdType;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase,
      Metavariables::Phase::RegisterWithObservers, RegistrationActionsList>>;
};

template <typename Metavariables>
struct observer_component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;

  using component_being_mocked = observers::Observer<Metavariables>;
  using simple_tags =
      typename observers::Actions::Initialize<Metavariables>::simple_tags;
  using compute_tags =
      typename observers::Actions::Initialize<Metavariables>::compute_tags;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<Actions::SetupDataBox,
                 observers::Actions::Initialize<Metavariables>>>>;
};

template <typename Metavariables>
struct observer_writer_component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<observers::Tags::ReductionFileName,
                                             observers::Tags::VolumeFileName>;

  using component_being_mocked = observers::ObserverWriter<Metavariables>;
  using simple_tags =
      typename observers::Actions::InitializeWriter<Metavariables>::simple_tags;
  using compute_tags = typename observers::Actions::InitializeWriter<
      Metavariables>::compute_tags;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<Actions::SetupDataBox,
                 observers::Actions::InitializeWriter<Metavariables>>>>;
};

using l2_error_datum = Parallel::ReductionDatum<double, funcl::Plus<>,
                                                funcl::Sqrt<funcl::Divides<>>,
                                                std::index_sequence<1>>;
using reduction_data_from_doubles = Parallel::ReductionData<
    Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
    Parallel::ReductionDatum<size_t, funcl::Plus<>>, l2_error_datum,
    l2_error_datum>;

using reduction_data_from_vector = Parallel::ReductionData<
    Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
    Parallel::ReductionDatum<size_t, funcl::Plus<>>,
    Parallel::ReductionDatum<std::vector<double>, funcl::VectorPlus>>;

// Nothing special about the order. We just want doubles and std::vector's.
using reduction_data_from_ds_and_vs = Parallel::ReductionData<
    Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
    Parallel::ReductionDatum<size_t, funcl::Plus<>>, l2_error_datum,
    Parallel::ReductionDatum<std::vector<double>, funcl::VectorPlus>,
    Parallel::ReductionDatum<std::vector<double>, funcl::VectorPlus>,
    l2_error_datum>;

template <typename RegistrationActionsList>
struct Metavariables {
  using component_list =
      tmpl::list<element_component<Metavariables, RegistrationActionsList>,
                 observer_component<Metavariables>,
                 observer_writer_component<Metavariables>>;

  /// [make_reduction_data_tags]
  using observed_reduction_data_tags = observers::make_reduction_data_tags<
      tmpl::list<reduction_data_from_doubles, reduction_data_from_vector,
                 reduction_data_from_ds_and_vs>>;
  /// [make_reduction_data_tags]

  enum class Phase { Initialization, RegisterWithObservers, Testing, Exit };
};
}  // namespace TestObservers_detail
