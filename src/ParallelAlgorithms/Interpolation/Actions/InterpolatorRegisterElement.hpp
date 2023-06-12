// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
namespace intrp {
template <typename Metavariables>
struct Interpolator;
}  // namespace intrp
/// \endcond

namespace intrp {
namespace Actions {

/// \ingroup ActionsGroup
/// \brief Invoked on the `Interpolator` ParallelComponent to register an
/// element with the `Interpolator`.
///
/// This is called by `RegisterElementWithInterpolator` below.
///
/// Uses: nothing
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::NumberOfElements`
///
/// For requirements on Metavariables, see `InterpolationTarget`.
struct RegisterElement {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/) {
    db::mutate<Tags::NumberOfElements>(
        [](const gsl::not_null<size_t*> num_elements) { ++(*num_elements); },
        make_not_null(&box));
  }
};

/// \ingroup ActionsGroup
/// \brief Invoked on the `Interpolator` ParallelComponent to deregister an
/// element with the `Interpolator`.
///
/// This is called by `RegisterElementWithInterpolator` below.
///
/// Uses: nothing
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::NumberOfElements`
///
/// For requirements on Metavariables, see `InterpolationTarget`.
struct DeregisterElement {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/) {
    db::mutate<Tags::NumberOfElements>(
        [](const gsl::not_null<size_t*> num_elements) { --(*num_elements); },
        make_not_null(&box));
  }
};

/// \ingroup ActionsGroup
/// \brief Invoked on `DgElementArray` to register all its elements with the
/// `Interpolator`.
///
/// Uses: nothing
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: nothing
///
/// When this struct is used as an action, the `apply` function will perform the
/// registration with the interpolator. However, this struct also offers the
/// static member functions `perform_registration` and `perform_deregistration`
/// that are needed for either registering when an element is added to a core
/// outside of initialization or deregistering when an element is being
/// eliminated from a core. The use of separate functions is necessary to
/// provide an interface usable outside of iterable actions, e.g. in specialized
/// `pup` functions.
struct RegisterElementWithInterpolator {
 private:
  template <typename ParallelComponent, typename RegisterOrDeregisterAction,
            typename Metavariables, typename ArrayIndex>
  static void register_or_deregister_impl(
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/) {
    auto& interpolator = *Parallel::local_branch(
        Parallel::get_parallel_component<::intrp::Interpolator<Metavariables>>(
            cache));
    Parallel::simple_action<RegisterOrDeregisterAction>(interpolator);
  }

 public:
  // the registration functions do not use the DataBox argument, but need to
  // keep it to conform to the interface used in the registration and
  // deregistration procedure in Algorithm.hpp, which also supports registration
  // and deregistration that does use the box.
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex>
  static void perform_registration(const db::DataBox<DbTagList>& /*box*/,
                                   Parallel::GlobalCache<Metavariables>& cache,
                                   const ArrayIndex& array_index) {
    register_or_deregister_impl<ParallelComponent, RegisterElement>(
        cache, array_index);
  }

  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex>
  static void perform_deregistration(
      const db::DataBox<DbTagList>& /*box*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index) {
    register_or_deregister_impl<ParallelComponent, DeregisterElement>(
        cache, array_index);
  }

  template <typename DbTagList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    perform_registration<ParallelComponent>(box, cache, array_index);
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace Actions
}  // namespace intrp
