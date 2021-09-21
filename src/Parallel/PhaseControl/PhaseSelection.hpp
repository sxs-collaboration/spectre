// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Utilities/TMPL.hpp"

namespace PhaseControl {
namespace detail {
CREATE_HAS_TYPE_ALIAS(type)
CREATE_HAS_TYPE_ALIAS(combine_method)
CREATE_HAS_TYPE_ALIAS(main_combine_method)
}  // namespace detail

/*!
 * \brief Compile-time information for control-flow of phases.
 *
 * \details A class conforming to this protocol is placed in the metavariables
 * to provide information about the set of phases that can be dynamically chosen
 * during a simulation, and the runtime information that is used to arbitrate
 * those decisions. The conforming class must provide the following type
 * aliases:
 *
 * - `phase_changes`: A `tmpl::list` of the set of `PhaseControl` objects used
 * to change phases.
 * - `initialize_phase_change_decision_data`: A type with a static member
 * function with signature:
 * ```
 * template <typename... DecisionTags, typename Metavariables>
 * static void apply(
 *     const gsl::not_null<tuples::TaggedTuple<DecisionTags...>*>
 *         phase_change_decision_data,
 *     const Parallel::GlobalCache<Metavariables>& cache) noexcept;
 * ```
 * that initializes each of the tags in `phase_change_tags_and_combines` that
 * require initial values. This should almost always be an alias to
 * `PhaseControl::InitializePhaseChangeDecisionData`, but can be customized to
 * perform additional initialization if necessary.
 * - `phase_change_tags_and_combines_list`: The set of tags that contain type
 * aliases `combine_method` and `main_combine_method` for determining how the
 * reduction is performed. These tags are typically available from the type
 * alias `phase_change_tags_and_combines` from the individual `PhaseChange`
 * objects, and can be aggregated from a list of `PhaseChange`s via
 * `PhaseControl::get_phase_change_tags`.
 * - `template <typename ParallelComponent> registration_list`: A metafunction
 * that has the type alias `type` that is the list of registration actions that
 * are used to de-register and re-register when an element of
 * `ParallelComponent` is migrated. (Note: each item in these lists must be
 * actions that also specify `perform_deregistration` and `perform_registration`
 * static member functions that perform the de-registration and re-registration
 * when an element is migrated. See
 * `observers::Actions::RegisterEventsWithObservers` for an example of an action
 * that conforms to the required interface).
 *
 * static member functions:
 * - `static std::string phase_name(Phase phase) noexcept`: A function that
 * returns the names of each phase that needs to be optionally specified in the
 * input file (e.g. phases chosen using `PhaseControl::VisitAndReturn`.
 *
 * \note Two important parts of the phase choice architecture must also be
 * present in the wider metavariables:
 * - the `enum class Phase` enumerating the set of phases
 * - the static member function with the signature:
 * ```
 * template <typename... Tags>
 * static Phase determine_next_phase(
 *     const gsl::not_null<tuples::TaggedTuple<Tags...>*>
 *         phase_change_decision_data,
 *     const Phase& current_phase,
 *     const Parallel::CProxy_GlobalCache<EvolutionMetavars>&
 *         cache_proxy) noexcept
 * ```
 * The function should typically have body similar to (with possible alteration
 * for the set of sequenced phases in the executable):
 *
 * \snippet Test_AlgorithmPhaseControl.cpp determine_next_phase_example
 */
struct PhaseSelection {
  template <typename ConformingType>
  struct test {
    using phase_changes = typename ConformingType::phase_changes;
    using initialize_phase_change_decision_data =
        typename ConformingType::initialize_phase_change_decision_data;
    static_assert(
        std::is_function_v<initialize_phase_change_decision_data::apply>);

    using phase_change_tags_and_combines =
        typename ConformingType::phase_change_tags_and_combines;
    static_assert(
        tmpl::all<
            phase_change_tags_and_combines,
            tmpl::bind<tmpl::all, detail::has_type<tmpl::_1>,
                       detail::has_combine_method<tmpl::_1>,
                       detail::has_main_combine_method<tmpl::_1>>>::value);

    static_assert(
        std::is_same_v<std::void_t<typename ConformingType::
                                       template registration_list<NoSuchType>>,
                       std::void_t<>>);
    static_assert(std::is_function_v<ConformingType::phase_name>);
  };
};
}  // namespace PhaseControl
