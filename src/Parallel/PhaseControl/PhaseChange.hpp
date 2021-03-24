// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"


/// Contains utilities for determining control-flow among phases
namespace PhaseControl {
/// The possible options for instructing the Main chare in deciding the next
/// phase to jump to.
///
/// An object of this enum type is packaged with a requested phase in the
/// `PhaseChange::arbitrate_phase_change` function.
enum ArbitrationStrategy {
  /// Jump to the requested phase immediately, before considering other
  /// requested phases.
  ///
  /// This will ensure that the requested phase is always run, where
  /// alternative methods could have 'double-jumps' where the Main chare
  /// replaces a requested phase immediately without actually entering the
  /// phase.
  RunPhaseImmediately,
  /// After the requested phase is considered, continue considering other
  /// requests, potentially replacing this request.
  ///
  /// This will permit reprocessing the phase-jumps to help with cases where
  /// multiple phases are simultaneously requested.
  /// The `PermitAdditionalJumps` permits 'double-jumps' where a requested phase
  /// is immediately replaced by another phase to jump to.
  PermitAdditionalJumps
};
}  // namespace PhaseControl

/*!
 * \brief `PhaseChange` objects determine the storage types and logic for
 * moving between phases based on runtime data.
 *
 * The phase control flow must have the cooperation of each parallel component,
 * but make phase decisions centrally so that at any point, all components are
 * in the same phase. The operations needed by the parallel components and by
 * the Main chare, are:
 *
 * 1. Parallel components must select and/or compute the runtime data necessary
 *    for choosing the next phase, then contribute it to a global reduction to
 *    the Main component.
 *    The components must then halt at a globally-valid state for the phase
 *    change.
 *    The requirements for the state will vary depending on the phase choices,
 *    so triggers must be selected appropriately for the `PhaseChange` object.
 *    For instance, selecting a common slab will usually represent a globally
 *    well-behaved state for a `DgElementArray`.
 * 2. On the Main chare, the `PhaseChange` objects must use the collected
 *    reduction data, or other persistent data stored in
 *    `phase_change_decision_data` to decide on a phase to request and an
 *    `PhaseControl::ArbitrationStrategy` to determine how to resolve multiple
 *    simultaneous requests.
 *    Additionally, the `PhaseChange` objects must specify initialization
 *    functions to set the starting state of the tags in
 *    `phase_change_decision_data` for which they are responsible.
 *
 * In addition to the `options` type alias and `static constexpr Options::String
 * help` variable needed to be option-creatable, a derived class of
 * `PhaseChange` must specify the type aliases:
 * - `argument_tags`: A `tmpl::list` of tags from the
 *   \ref DataBoxGroup "DataBox" to be passed to `contribute_phase_data_impl` as
 *   const references.
 * - `return_tags`: A `tmpl::list` of mutable tags from the
 *   \ref DataBoxGroup "DataBox" to be passed to `contribute_phase_data_impl` as
 *   `gsl::not_null` pointers. This should be used only for tags that may be
 *   altered during the `contribute_phase_data_impl` function.
 * - `phase_change_tags_and_combines_list`: A `tmpl::list` of tags for
 *   populating the `phase_change_decision_data` in the Main chare. Each tag in
 *   this list must also define a `combine_method` and a `main_combine_method`
 *   for performing the aggregation during reduction.
 * - `participating_components` (templated on `Metavariables`): A `tmpl::list`
 *   of components that contribute data during this reduction. This can be used
 *   to screen out components that will not have the necessary information to
 *   contribute to the reduction. If all components should participate, this
 *   type alias can be set to simply `typename Metavariables::component_list`.
 *
 * And member functions with signatures:
 *
 * ```
 * template <typename... DecisionTags>
 * void initialize_phase_data_impl(
 *     const gsl::not_null<tuples::TaggedTuple<DecisionTags...>*>
 *         phase_change_decision_data) const noexcept;
 * ```
 * - Must set all tags in `phase_change_tags_and_combines_list` to useful
 *   initial states in the `phase_change_decision_data`.
 *
 * ```
 * template <typename ParallelComponent, typename ArrayIndex>
 * void contribute_phase_data_impl(
 *     [DataBox return tags...], [DataBox argument tags...],
 *     Parallel::GlobalCache<Metavariables>& cache,
 *     const ArrayIndex& array_index) const noexcept;
 * ```
 * - Should send any data relevant for the associated phase change decision made
 *   in `arbitrate_phase_change_impl` to the Main chare via function
 *   `Parallel::contribute_to_phase_change_reduction`.
 *
 * ```
 * template <typename... DecisionTags, typename Metavariables>
 * typename std::optional<
 *     std::pair<typename Metavariables::Phase, ArbitrationStrategy>>
 * arbitrate_phase_change_impl(
 *     const gsl::not_null<tuples::TaggedTuple<DecisionTags...>*>
 *         phase_change_decision_data,
 *     const typename Metavariables::Phase current_phase,
 *     const Parallel::GlobalCache<Metavariables>& cache) const noexcept;
 * ```
 * - Should examine the collected data in `phase_change_decision_data` and
 *   optionally return a `std::pair` with the desired `Metavariables::Phase` and
 *   an `PhaseControl::ArbitrationStrategy` indicating a method for arbitrating
 *   multiple simultaneous requests. Alternatively, it may return `std::nullopt`
 *   to abstain from the phase decision.
 *   The `arbitrate_phase_change_impl` may (and often will) mutate the
 *   `phase_change_decision_data`. For instance, it may be desirable to 'reset'
 *   the data to allow for future jumps associated with the same `PhaseChange`,
 *   or the `PhaseChange` will describe multiple changes in sequence, and the
 *   state of that sequential process can be recorded in
 *   `phase_change_decision_data`.
 */
template <typename PhaseChangeRegistrars>
struct PhaseChange : public PUP::able {
 protected:
  /// \cond
  PhaseChange() = default;
  PhaseChange(const PhaseChange&) = default;
  PhaseChange(PhaseChange&&) = default;
  PhaseChange& operator=(const PhaseChange&) = default;
  PhaseChange& operator=(PhaseChange&&) = default;
  /// \endcond

 public:
  ~PhaseChange() override = default;

  WRAPPED_PUPable_abstract(PhaseChange);  // NOLINT

  using creatable_classes = Registration::registrants<PhaseChangeRegistrars>;

  /// Send data from all `participating_components` to the Main chare for
  /// determining the next phase.
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  void contribute_phase_data(const gsl::not_null<db::DataBox<DbTags>*> box,
                             Parallel::GlobalCache<Metavariables>& cache,
                             const ArrayIndex& array_index) const noexcept {
    call_with_dynamic_type<
        void, creatable_classes>(this, [&box, &cache, &array_index](
                                           const auto* const
                                               phase_change) noexcept {
      using phase_change_t = typename std::decay_t<decltype(*phase_change)>;
      if constexpr (tmpl::list_contains_v<
                        typename phase_change_t::
                            template participating_components<Metavariables>,
                        ParallelComponent>) {
        db::mutate_apply<typename phase_change_t::return_tags,
                         typename phase_change_t::argument_tags>(
            [&phase_change, &cache, &array_index](auto&&... args) noexcept {
              phase_change
                  ->template contribute_phase_data_impl<ParallelComponent>(
                      args..., cache, array_index);
            },
            box);
      }
    });
  }

  /// Determine a phase request and `PhaseControl::ArbitrationStrategy` based on
  /// aggregated `phase_change_decision_data` on the Main Chare.
  template <typename... DecisionTags, typename Metavariables>
  std::optional<std::pair<typename Metavariables::Phase,
                          PhaseControl::ArbitrationStrategy>>
  arbitrate_phase_change(
      const gsl::not_null<tuples::TaggedTuple<DecisionTags...>*>
          phase_change_decision_data,
      const typename Metavariables::Phase current_phase,
      const Parallel::GlobalCache<Metavariables>& cache) const noexcept {
    return call_with_dynamic_type<
        std::optional<std::pair<typename Metavariables::Phase,
                                PhaseControl::ArbitrationStrategy>>,
        creatable_classes>(
        this, [&current_phase, &phase_change_decision_data,
               &cache](const auto* const phase_change) noexcept {
          return phase_change->arbitrate_phase_change_impl(
              phase_change_decision_data, current_phase, cache);
        });
  }

  /// Initialize the `phase_change_decision_data` on the main chare to starting
  /// values.
  template <typename... Tags>
  void initialize_phase_data(
      const gsl::not_null<tuples::TaggedTuple<Tags...>*>
          phase_change_decision_data) const noexcept {
    return call_with_dynamic_type<void, creatable_classes>(
        this, [&phase_change_decision_data](
                  const auto* const phase_change) noexcept {
          return phase_change->initialize_phase_data_impl(
              phase_change_decision_data);
        });
  }
};
