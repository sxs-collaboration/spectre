// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Parallel/PhaseControl/PhaseChange.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/Serialize.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace PhaseControl {

/// A type for denoting a piece of data for deciding a phase change.
///
/// `Tag` is intended to be a tag (with a type) for indexing a
/// `tuples::TaggedTuple`, and `CombineMethod` is intended to be a
/// `Parallel::ReductionDatum`-compatible invokable for combining the `type` of
/// the `Tag`. The `MainCombineMethod` is provided to give flexibility for a
/// different method of combination at the top level of the hierarchy (so, in
/// the case of phase control reductions, performed by the main chare to combine
/// reductions from different chares)
template <typename Tag, typename CombineMethod,
          typename MainCombineMethod = CombineMethod>
struct TagAndCombine : Tag {
  using tag = Tag;
  using combine_method = CombineMethod;
  using main_combine_method = MainCombineMethod;
};

/// A flexible combine invokable that combines into a `tuples::TaggedTuple` a
/// new `tuples::TaggedTuple`, and combines according to type aliases
/// `combination_method`s that are required to be defined in each tag.
struct TaggedTupleCombine {
  template <typename... Tags>
  tuples::TaggedTuple<Tags...> operator()(
      tuples::TaggedTuple<Tags...> current_state,
      const tuples::TaggedTuple<Tags...>& element) {
    tmpl::for_each<tmpl::list<Tags...>>(
        [&current_state, &element](auto tag_v) noexcept {
          using tag = typename decltype(tag_v)::type;
          tuples::get<tag>(current_state) = typename tag::combine_method{}(
              tuples::get<tag>(current_state), tuples::get<tag>(element));
        });
    return current_state;
  }
};

/// A flexible combine invokable that combines into a `tuples::TaggedTuple` a
/// new `tuples::TaggedTuple` with a subset of the original tags, and combines
/// according to type aliases `main_combine_method`s that are required to be
/// defined in each tag.
///
/// \note This is _not_ usable with charm++ reductions; it mutates the current
/// state in-place. This is constructed for the use-case where the main chare
/// stores a persistent data structure and combines reduction data as it arrives
/// from the other chares.
struct TaggedTupleMainCombine {
  template <typename... CurrentTags, typename... CombineTags>
  static void apply(
      const gsl::not_null<tuples::TaggedTuple<CurrentTags...>*> current_state,
      const tuples::TaggedTuple<CombineTags...>& element) {
    tmpl::for_each<tmpl::list<CombineTags...>>(
        [&current_state, &element](auto tag_v) noexcept {
          using tag = typename decltype(tag_v)::type;
          tuples::get<tag>(*current_state) =
              typename tag::main_combine_method{}(
                  tuples::get<tag>(*current_state), tuples::get<tag>(element));
        });
  }
};

/// A `Parallel::ReductionData` with a single `Parallel::ReductionDatum` for a
/// given tagged tuple type determined by `TagsPresent`, and performs the
/// combine according to `TagsAndCombines`, which must be a `tmpl::list` of
/// `PhaseControl::TagAndCombine`s.
///
/// Each tag in the `TagsAndCombinesPresent` may either be a `TagsAndCombines`
/// or otherise define all three type traits `type`, `combine_method`, and
/// `main_combine_method`.
template <typename TagsAndCombinesPresent, typename TagsAndCombines>
using reduction_data = Parallel::ReductionData<Parallel::ReductionDatum<
    tuples::tagged_tuple_from_typelist<TagsAndCombinesPresent>,
    TaggedTupleCombine>>;

namespace OptionTags {
/// Option tag for the collection of triggers that indicate synchronization
/// points at which phase changes should be considered, and the associated
/// `PhaseChange` objects for making the phase change decisions.
///
/// When the phase control is arbitrated on the main chare, the `PhaseChange`
/// objects will be queried for their phase request in order of appearance in
/// the nested list (i.e. first all of the `PhaseChange`s associated with the
/// first trigger, in order, then those associated with the second trigger,
/// etc.). The order therefore determines the order of resolution of
/// simultaneous requests.
///
/// \note The nested collection types for this option tag gives the yaml
/// format the slightly unusual form:
///
/// ```
/// PhaseChangeAndTriggers:
///   - - Trigger1
///     - - PhaseChange1
///       - PhaseChange2
///   - - Trigger2
///     - - PhaseChange3
///       - PhaseChange4
/// ```
template <typename PhaseChangeRegistrars, typename TriggerRegistrars>
struct PhaseChangeAndTriggers {
  using phase_change_type = PhaseChange<PhaseChangeRegistrars>;
  using trigger_type = Trigger<TriggerRegistrars>;
  static constexpr Options::String help{
      "A collection of pairs of triggers and collections of phase change "
      "objects to determine runtime phase control-flow decisions. The order of "
      "the phase change objects determines the order of the requests processed "
      "by the Main chare during phase change arbitration."};

  using type =
      std::vector<std::pair<std::unique_ptr<trigger_type>,
                            std::vector<std::unique_ptr<phase_change_type>>>>;
};
}  // namespace OptionTags

namespace Tags {
/// Tag for the collection of triggers that indicate synchronization points at
/// which phase changes should be considered, and the associated `PhaseChange`
/// objects for making the phase change decisions.
template <typename PhaseChangeRegistrars, typename TriggerRegistrars>
struct PhaseChangeAndTriggers : db::SimpleTag {
  using phase_change_type = PhaseChange<PhaseChangeRegistrars>;
  using trigger_type = Trigger<TriggerRegistrars>;
  using type =
      std::vector<std::pair<std::unique_ptr<trigger_type>,
                            std::vector<std::unique_ptr<phase_change_type>>>>;

  using option_tags =
      tmpl::list<OptionTags::PhaseChangeAndTriggers<PhaseChangeRegistrars,
                                                    TriggerRegistrars>>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(
      const type& phase_control_and_triggers) noexcept {
    return deserialize<type>(
        serialize<type>(phase_control_and_triggers).data());
  }
};
}  // namespace Tags

namespace TagsAndCombines {
/// A tag for indicating that a halt was called by a trigger associated with
/// `PhaseChange`s.
///
/// This is needed to disambiguate different quiescence conditions in the main
/// chare. It is automatically included in
/// `PhaseControl::get_phase_change_tags`, so shouldn't be explicitly included
/// in the `phase_change_tags_and_combines_list` in derived classes of
/// `PhaseChange`.
struct UsePhaseChangeArbitration {
  using type = bool;
  using combine_method = funcl::Or<>;
  using main_combine_method = combine_method;
};
}  // namespace TagsAndCombines

namespace detail {
template <typename PhaseChangeDerived>
struct get_phase_change_tags_and_combines {
  using type = typename PhaseChangeDerived::phase_change_tags_and_combines;
};
}  // namespace detail

/// Metafunction for determining the merged collection of tags in
/// `phase_change_tags_and_combines_list`s from all `PhaseChange` derived
/// classes registered in `PhaseChangeRegistrars`
template <typename PhaseChangeRegistrars>
using get_phase_change_tags =
    tmpl::push_back<tmpl::flatten<tmpl::transform<
                        Registration::registrants<PhaseChangeRegistrars>,
                        detail::get_phase_change_tags_and_combines<tmpl::_1>>>,
                    TagsAndCombines::UsePhaseChangeArbitration>;
}  // namespace PhaseControl
