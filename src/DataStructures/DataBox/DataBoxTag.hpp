// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines classes SimpleTag, PrefixTag, ComputeTag and several
/// functions for retrieving tag info

#pragma once

#include <cstddef>
#include <ostream>
#include <string>

#include "DataStructures/DataBox/Subitems.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagTraits.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/CreateIsCallable.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

namespace Tags {
/*!
 * \ingroup DataBoxTagsGroup
 * \brief Tag used to retrieve the DataBox from the `db::get` function
 *
 * The main use of this tag is to allow fetching the DataBox from itself. The
 * primary use case is to allow an invokable to take a DataBox as an argument
 * when called through `db::apply`.
 *
 * \snippet Test_DataBox.cpp databox_self_tag_example
 */
struct DataBox {
  // Trick to get friend function declaration to compile but a const void& is
  // rather useless
  using type = void;
};
}  // namespace Tags

namespace db {

namespace detail {
template <typename TagList, typename Tag>
using list_of_matching_tags = tmpl::conditional_t<
    std::is_same_v<Tag, ::Tags::DataBox>, tmpl::list<::Tags::DataBox>,
    tmpl::filter<TagList, std::is_base_of<tmpl::pin<Tag>, tmpl::_1>>>;

template <typename Tag, typename TagList,
          typename MatchingTagsList = list_of_matching_tags<TagList, Tag>>
struct first_matching_tag_impl {
  using type = tmpl::front<MatchingTagsList>;
};

template <typename Tag, typename TagList>
struct first_matching_tag_impl<Tag, TagList, tmpl::list<>> {
  static_assert(std::is_same<Tag, NoSuchType>::value,
                "Could not find the DataBox tag in the list of DataBox tags. "
                "The first template parameter of 'first_matching_tag_impl' is "
                "the tag that cannot be found and the second is the list of "
                "tags being searched.");
  using type = NoSuchType;
};

template <typename TagList, typename Tag>
using first_matching_tag = typename first_matching_tag_impl<Tag, TagList>::type;

template <typename TagList, typename Tag>
constexpr auto number_of_matching_tags =
    tmpl::size<list_of_matching_tags<TagList, Tag>>::value;

template <typename TagList, typename Tag>
struct has_unique_matching_tag
    : std::integral_constant<bool, number_of_matching_tags<TagList, Tag> == 1> {
};

template <typename TagList, typename Tag>
using has_unique_matching_tag_t =
    typename has_unique_matching_tag<TagList, Tag>::type;

template <typename TagList, typename Tag>
constexpr bool has_unique_matching_tag_v =
    has_unique_matching_tag<TagList, Tag>::value;

template <typename TagList, typename Tag>
struct has_no_matching_tag
    : std::integral_constant<bool, number_of_matching_tags<TagList, Tag> == 0> {
};

template <typename TagList, typename Tag>
using has_no_matching_tag_t = typename has_no_matching_tag<TagList, Tag>::type;

template <typename TagList, typename Tag>
constexpr bool has_no_matching_tag_v = has_no_matching_tag<TagList, Tag>::value;

template <class T, class = void>
struct has_return_type_member : std::false_type {};
template <class T>
struct has_return_type_member<T, std::void_t<typename T::return_type>>
    : std::true_type {};

/*!
 * \brief `true` if `T` has nested type alias named `return_type`
 */
template <class T>
constexpr bool has_return_type_member_v = has_return_type_member<T>::value;

template <typename T>
struct ConvertToConst {
  using type = const T&;
};

template <typename T>
struct ConvertToConst<std::unique_ptr<T>> {
  using type = const T&;
};

template <typename T>
const T& convert_to_const_type(const T& item) noexcept {
  return item;
}

template <typename T>
const T& convert_to_const_type(const std::unique_ptr<T>& item) noexcept {
  return *item;
}

template <typename TagList, typename Tag>
struct storage_type_impl;

template <int SelectTagType>
struct dispatch_storage_type;

template <typename ArgsList>
struct compute_item_type;

template <typename TagList, typename Tag>
struct storage_type_impl {
  // storage_type_impl is intentionally a lazy metafunction rather than a
  // metaclosure or a metacontinuation. The reason is that it is quite likely we
  // call `db::item_type<Tag>` multiple times within the same translation unit
  // and so we want to memoize the resulting type. However, we do not want to
  // memoize the dispatch calls.
  using type = typename dispatch_storage_type<
      is_base_tag_v<Tag>
          ? 4
          : (is_compute_tag_v<Tag>and has_return_type_member_v<Tag>)
                ? 3
                : is_immutable_item_tag_v<Tag>
                      ? 2
                      : std::is_base_of_v<db::SimpleTag, Tag> ? 1 : 0>::
      template f<TagList, Tag>;
};

template <>
struct dispatch_storage_type<0> {
  // Tag is not a tag. This is necessary for SFINAE friendliness, specifically
  // if someone calls Requires<tt::is_a_v<std::vector, db::item_type<Tag>>>
  // with, say Tag = double, then this should probably SFINAE away, not fail to
  // compile.
  template <typename TagList, typename Tag>
  using f = NoSuchType;
};

template <>
struct dispatch_storage_type<1> {
  // simple item
  template <typename TagList, typename Tag>
  using f = typename Tag::type;
};

template <>
struct dispatch_storage_type<2> {
  // compute item
  template <typename TagList, typename Tag>
  using f = typename compute_item_type<typename Tag::argument_tags>::template f<
      TagList, Tag>;
};

template <>
struct dispatch_storage_type<3> {
  // mutating compute item
  template <typename TagList, typename Tag>
  using f = typename Tag::return_type;
};

template <typename TagList, typename Tag>
struct get_first_derived_tag_for_base_tag {
  static_assert(
      not std::is_same_v<TagList, NoSuchType>,
      "Can't retrieve the storage type of a base tag without the full tag "
      "list. If you're using 'item_type' or 'const_item_type' then make sure "
      "you pass the DataBox's tag list as the second template parameter to "
      "those metafunctions. The base tag for which the storage type is being"
      "retrieved is listed as the second template argument to the "
      "'get_first_derived_tag_for_base_tag' class below");
  using type = first_matching_tag<TagList, Tag>;
};

template <>
struct dispatch_storage_type<4> {
  // base tag item: retrieve the derived tag from the tag list then call
  // storage_type_impl on the result.
  // We do not check that there is only one matching tag in the DataBox because
  // the uniqueness is only checked in get and mutate. The reason for that is
  // that it is fine to have multiple derived tags in the DataBox as long as
  // they all have the same type. We do not check that the types are all the
  // same, it is undefined behavior if they are not and the user's
  // responsibility.
  template <typename TagList, typename Tag>
  using f = typename storage_type_impl<
      TagList,
      tmpl::type_from<get_first_derived_tag_for_base_tag<TagList, Tag>>>::type;
};

// The type internally stored in a simple or compute item.  For
// convenience in implementing the various tag type metafunctions,
// this will also look up types for db::BaseTags, even though they
// cannot actually store anything.
template <typename Tag, typename TagList>
using storage_type = typename storage_type_impl<TagList, Tag>::type;

template <typename TagList, typename Tag>
struct item_type_impl {
  static_assert(not is_compute_tag_v<Tag>,
                "Can't call item_type on a compute item because compute items "
                "cannot be modified.  You probably wanted const_item_type.");
  using type = detail::storage_type<Tag, TagList>;
};

// Get the type that is returned by `get<Tag>`. If it is a base tag then a
// `TagList` must be passed as a second argument.
template <typename Tag, typename TagList = NoSuchType>
using const_item_type =
    typename ConvertToConst<std::decay_t<storage_type<Tag, TagList>>>::type;

// Get the type that can be written to the `Tag`. If it is a base tag then a
// `TagList` must be passed as a second argument.
template <typename Tag, typename TagList = NoSuchType>
using item_type = typename item_type_impl<TagList, Tag>::type;

CREATE_IS_CALLABLE(function)
CREATE_IS_CALLABLE_V(function)

template <typename Tag, typename TagList, typename TagTypesList>
struct check_compute_item_is_invokable;

template <typename Tag, typename... Tags, typename... TagTypes>
struct check_compute_item_is_invokable<Tag, tmpl::list<Tags...>,
                                       tmpl::list<TagTypes...>> {
  static_assert(
      is_function_callable_v<Tag, TagTypes...>,
      "The compute item is not callable with the types that the tags hold. The "
      "compute item tag that is the problem should be shown in the first line "
      " after the static assert error: "
      "'check_compute_item_is_invokable<TheItemThatsFailingToBeCalled, "
      "brigand::list<PassedTags...>, "
      "brigand::list<const_item_type<PassedTags>...>>'");
};

template <typename... Args>
struct compute_item_type<tmpl::list<Args...>> {
  template <typename TagList, typename Tag>
  using f = decltype(
#ifdef SPECTRE_DEBUG
      (void)check_compute_item_is_invokable<
          Tag, tmpl::list<Args...>,
          tmpl::list<const_item_type<Args, TagList>...>>{},
#endif  // SPECTRE_DEBUG
      Tag::function(std::declval<const_item_type<Args, TagList>>()...));
};
}  // namespace detail
}  // namespace db
