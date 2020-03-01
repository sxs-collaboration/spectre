// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <functional>

#include "Domain/ElementIndex.hpp"
#include "IO/Observer/Actions.hpp"  // IWYU pragma: keep
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"      // IWYU pragma: keep
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

/// \cond
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
/// \endcond

namespace TestObservers_detail {
using ElementIndexType = ElementIndex<2>;

template <observers::TypeOfObservation TypeOfObservation>
struct RegisterThisObsType {
  struct ElementObservationType {};
  template <typename ParallelComponent, typename DbTagsList,
            typename ArrayIndex>
  static std::pair<observers::TypeOfObservation, observers::ObservationId>
  register_info(const db::DataBox<DbTagsList>& /*box*/,
                const ArrayIndex& /*array_index*/) noexcept {
    return {TypeOfObservation,
            observers::ObservationId{3.0, ElementObservationType{}}};
  }
};

template <typename Metavariables,
          observers::TypeOfObservation TypeOfObservation>
struct element_component {
  using component_being_mocked = void;  // Not needed
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndexType;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase,
      Metavariables::Phase::RegisterWithObservers,
      tmpl::list<observers::Actions::RegisterWithObservers<
                     RegisterThisObsType<TypeOfObservation>>,
                 Parallel::Actions::TerminatePhase>>>;
};

template <typename Metavariables>
struct observer_component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using component_being_mocked = observers::Observer<Metavariables>;
  using simple_tags =
      typename observers::Actions::Initialize<Metavariables>::simple_tags;
  using compute_tags =
      typename observers::Actions::Initialize<Metavariables>::compute_tags;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<observers::Actions::Initialize<Metavariables>>>>;
};

template <typename Metavariables>
struct observer_writer_component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tags = tmpl::list<observers::Tags::ReductionFileName,
                                             observers::Tags::VolumeFileName>;

  using component_being_mocked = observers::ObserverWriter<Metavariables>;
  using simple_tags =
      typename observers::Actions::InitializeWriter<Metavariables>::simple_tags;
  using compute_tags = typename observers::Actions::InitializeWriter<
      Metavariables>::compute_tags;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<observers::Actions::InitializeWriter<Metavariables>>>>;
};

using l2_error_datum = Parallel::ReductionDatum<double, funcl::Plus<>,
                                                funcl::Sqrt<funcl::Divides<>>,
                                                std::index_sequence<1>>;
using reduction_data_from_doubles = Parallel::ReductionData<
    Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
    Parallel::ReductionDatum<size_t, funcl::Plus<>>, l2_error_datum,
    l2_error_datum>;

struct VectorPlus {
  std::vector<double> operator()(const std::vector<double>& lhs,
                                 const std::vector<double>& rhs) const
      noexcept {
    ASSERT(lhs.size() == rhs.size(),
           "Vector sizes in `VectorPlus` operator do not match. First argument "
           "size: "
               << lhs.size() << ". Second argument size: " << rhs.size()
               << ".");
    std::vector<double> result(lhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
      result[i] = lhs[i] + rhs[i];
    }
    return result;
  }
};

using reduction_data_from_vector = Parallel::ReductionData<
    Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
    Parallel::ReductionDatum<size_t, funcl::Plus<>>,
    Parallel::ReductionDatum<std::vector<double>, VectorPlus>>;

// Nothing special about the order. We just want doubles and std::vector's.
using reduction_data_from_ds_and_vs = Parallel::ReductionData<
    Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
    Parallel::ReductionDatum<size_t, funcl::Plus<>>, l2_error_datum,
    Parallel::ReductionDatum<std::vector<double>, VectorPlus>,
    Parallel::ReductionDatum<std::vector<double>, VectorPlus>, l2_error_datum>;

template <observers::TypeOfObservation TypeOfObservation>
struct Metavariables {
  using component_list =
      tmpl::list<element_component<Metavariables, TypeOfObservation>,
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
