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
  // Trick to get friend function declaration to compile but a const
  // NoSuchtype****& is rather useless
  using type = NoSuchType****;
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

template <typename T>
struct ConvertToConst {
  using type = const T&;
};

template <typename T>
struct ConvertToConst<std::unique_ptr<T>> {
  using type = const T&;
};

template <typename Tag, typename TagsList, bool = db::is_base_tag_v<Tag>>
struct const_item_type_impl {
  using type = typename db::detail::ConvertToConst<
      std::decay_t<typename Tag::type>>::type;
};

template <typename Tag, typename TagsList>
struct const_item_type_impl<Tag, TagsList, true> {
  using type = typename db::detail::ConvertToConst<std::decay_t<
      typename db::detail::first_matching_tag<TagsList, Tag>::type>>::type;
};

template <typename Tag, typename TagsList>
using const_item_type = typename const_item_type_impl<Tag, TagsList>::type;

}  // namespace detail
}  // namespace db
