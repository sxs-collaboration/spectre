// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Parallel/Reduction.hpp"
#include "Parallel/Serialize.hpp"
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
    tmpl::for_each<tmpl::list<CombineTags...>>([&current_state,
                                                &element](auto tag_v) noexcept {
      using tag = typename decltype(tag_v)::type;
      tuples::get<tag>(*current_state) = typename tag::main_combine_method{}(
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
}  // namespace PhaseControl
