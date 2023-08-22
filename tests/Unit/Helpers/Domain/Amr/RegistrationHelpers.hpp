// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unordered_set>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Protocols/ElementRegistrar.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::amr {
template <size_t Dim>
struct RegisteredElements : db::SimpleTag {
  using type = std::unordered_set<ElementId<Dim>>;
};

template <typename Metavariables>
struct Registrar {
  using metavariables = Metavariables;
  static constexpr size_t volume_dim = Metavariables::volume_dim;
  using chare_type = ActionTesting::MockGroupChare;
  using array_index = int;
  using simple_tags = tmpl::list<RegisteredElements<volume_dim>>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
};

struct DeregisterWithRegistrant {
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagList>& box,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ElementId<Metavariables::volume_dim>& element_id) {
    static constexpr size_t volume_dim = Metavariables::volume_dim;
    db::mutate<RegisteredElements<volume_dim>>(
        [&element_id](
            const gsl::not_null<std::unordered_set<ElementId<volume_dim>>*>
                registered_elements) {
          registered_elements->erase(element_id);
        },
        make_not_null(&box));
  }
};

struct RegisterWithRegistrant {
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagList>& box,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ElementId<Metavariables::volume_dim>& element_id) {
    static constexpr size_t volume_dim = Metavariables::volume_dim;
    db::mutate<RegisteredElements<volume_dim>>(
        [&element_id](
            const gsl::not_null<std::unordered_set<ElementId<volume_dim>>*>
                registered_elements) {
          registered_elements->insert(element_id);
        },
        make_not_null(&box));
  }
};

// [element_registrar_example]
struct RegisterElement : tt::ConformsTo<Parallel::protocols::ElementRegistrar> {
 private:
  template <typename ParallelComponent, typename RegisterOrDeregisterAction,
            typename DbTagList, typename Metavariables, typename ArrayIndex>
  static void register_or_deregister_impl(
      db::DataBox<DbTagList>& /*box*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index) {
    auto& registrar = *Parallel::local_branch(
        Parallel::get_parallel_component<Registrar<Metavariables>>(cache));
    Parallel::simple_action<RegisterOrDeregisterAction>(registrar, array_index);
  }

 public:
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex>
  static void perform_registration(db::DataBox<DbTagList>& box,
                                   Parallel::GlobalCache<Metavariables>& cache,
                                   const ArrayIndex& array_index) {
    register_or_deregister_impl<ParallelComponent, RegisterWithRegistrant>(
        box, cache, array_index);
  }

  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex>
  static void perform_deregistration(
      db::DataBox<DbTagList>& box, Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index) {
    register_or_deregister_impl<ParallelComponent, DeregisterWithRegistrant>(
        box, cache, array_index);
  }
};
// [element_registrar_example]

}  // namespace TestHelpers::amr
