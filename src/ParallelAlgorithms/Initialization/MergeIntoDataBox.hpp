// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/HasInequivalence.hpp"

namespace Initialization {
/// The merge policy for resolving conflicts in `merge_into_databox` when adding
/// simple tags that already exist.
enum class MergePolicy {
  /// Overwrite the existing simple tags
  Overwrite,
  /// Cause an error if a simple tag is already in the DataBox.
  Error,
  /// Cause an error if a simple tag is in the DataBox but does not compare
  /// equal to the new value being added. Ignores types that do not have an
  /// equivalence operator.
  IgnoreIncomparable
};

namespace detail {
template <typename AddingAction, typename SimpleTag, MergePolicy Policy,
          typename DbTagsList,
          Requires<Policy == MergePolicy::Overwrite> = nullptr>
void merge_simple_tag_value(
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
    typename SimpleTag::type&& simple_tag_value) noexcept {
  db::mutate<SimpleTag>(
      box,
      [](const gsl::not_null<typename SimpleTag::type*> stored_simple_tag_value,
         typename SimpleTag::type&& local_simple_tag_value) noexcept {
        *stored_simple_tag_value = std::move(local_simple_tag_value);
      },
      std::move(simple_tag_value));
}

template <typename AddingAction, typename SimpleTag, MergePolicy Policy,
          typename DbTagsList,
          Requires<(Policy == MergePolicy::Error or
                    Policy == MergePolicy::IgnoreIncomparable) and
                   tt::has_inequivalence_v<typename SimpleTag::type>> = nullptr>
void merge_simple_tag_value(const gsl::not_null<db::DataBox<DbTagsList>*> box,
                            typename SimpleTag::type&& simple_tag) noexcept {
  if (db::get<SimpleTag>(*box) != simple_tag) {
    ERROR("While adding the simple tag "
          << db::tag_name<SimpleTag>()
          << " that is already in the DataBox we found that the value being "
             "set by the action "
          << pretty_type::get_name<AddingAction>()
          << " is not the same as what is already in the DataBox. The "
             "value in the DataBox is: "
          << db::get<SimpleTag>(*box) << " while the value being added is "
          << simple_tag);
  }
}

template <
    typename AddingAction, typename SimpleTag, MergePolicy Policy,
    typename DbTagsList,
    Requires<(Policy == MergePolicy::Error or
              Policy == MergePolicy::IgnoreIncomparable) and
             not tt::has_inequivalence_v<typename SimpleTag::type>> = nullptr>
void merge_simple_tag_value(
    const gsl::not_null<db::DataBox<DbTagsList>*> /*box*/,
    typename SimpleTag::type&& /*simple_tag*/) noexcept {
  static_assert(Policy != MergePolicy::Error,
                "The tag being added does not have an equivalence operator and "
                "is already in the DataBox. See the first template parameter "
                "of the 'merge_simple_tag_value' function in the error message "
                "for the action trying to re-add the simple tag and the second "
                "template parameter for the simple tag being added.");
}

template <typename AddingAction, typename ComputeTagsList, MergePolicy Policy,
          typename... SimpleTags, typename DbTagsList,
          typename... SimpleTagsToAdd, typename... SimpleTagsToCheck>
auto merge_into_databox_impl(
    db::DataBox<DbTagsList>&& box,
    tuples::TaggedTuple<SimpleTags...>&& simple_tags,
    tmpl::list<SimpleTagsToAdd...> /*meta*/,
    tmpl::list<SimpleTagsToCheck...> /*meta*/) noexcept {
  EXPAND_PACK_LEFT_TO_RIGHT(
      merge_simple_tag_value<AddingAction, SimpleTagsToCheck, Policy>(
          make_not_null(&box),
          std::move(tuples::get<SimpleTagsToCheck>(simple_tags))));

  // Only add compute tags that are not in the DataBox.
  using compute_tags_to_add =
      tmpl::remove_if<ComputeTagsList,
                      tmpl::bind<tmpl::list_contains,
                                 tmpl::pin<typename db::DataBox<
                                     DbTagsList>::immutable_item_creation_tags>,
                                 tmpl::_1>>;

  return db::create_from<db::RemoveTags<>,
                         db::AddSimpleTags<SimpleTagsToAdd...>,
                         compute_tags_to_add>(
      std::move(box), std::move(tuples::get<SimpleTagsToAdd>(simple_tags))...);
}
}  // namespace detail

/*!
 * \ingroup InitializationGroup
 * \brief Add tags that are not yet in the DataBox.
 *
 * How duplicate tags are handled depends on the `MergePolicy` passed. The
 * default merge policy is to error if the simple tag being added does not have
 * an equivalence operator and is being added again.
 * `MergePolicy::IgnoreIncomparable` means that nothing is done if the simple
 * tag being added does not have an equivalence operator. Tags that have an
 * equivalence operator are check that the value being added and that is already
 * in the DataBox are the same. If they aren't, an error occurs. Finally,
 * `MergePolicy::Overwrite` means the current simple tag in the DataBox is
 * overwritten with the new value passed in.
 *
 * Compute tags that are not in the DataBox are added, ones that are in the
 * DataBox already are ignored.
 */
template <typename AddingAction, typename SimpleTagsList,
          typename ComputeTagsList = tmpl::list<>,
          MergePolicy Policy = MergePolicy::Error, typename... SimpleTags,
          typename DbTagsList, typename... Args>
auto merge_into_databox(db::DataBox<DbTagsList>&& box,
                        Args&&... args) noexcept {
  using simple_tags_to_check = tmpl::filter<
      SimpleTagsList,
      tmpl::bind<tmpl::list_contains,
                 tmpl::pin<typename db::DataBox<DbTagsList>::mutable_item_tags>,
                 tmpl::_1>>;
  using simple_tags_to_add =
      tmpl::remove_if<SimpleTagsList,
                      tmpl::bind<tmpl::list_contains,
                                 tmpl::pin<simple_tags_to_check>, tmpl::_1>>;
  return detail::merge_into_databox_impl<AddingAction, ComputeTagsList, Policy>(
      std::move(box),
      tuples::tagged_tuple_from_typelist<SimpleTagsList>{
          std::forward<Args>(args)...},
      simple_tags_to_add{}, simple_tags_to_check{});
}
}  // namespace Initialization
