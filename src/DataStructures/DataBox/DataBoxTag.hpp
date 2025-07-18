// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <type_traits>

#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataBox/TagTraits.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/TMPL.hpp"
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
template <typename TagsList,
          typename MatchingTagsList = tmpl::filter<
              TagsList, tt::is_a<Parallel::Tags::MetavariablesImpl, tmpl::_1>>>
struct metavars_tag_impl {
  static_assert(tmpl::size<MatchingTagsList>::value == 1);
  using type = tmpl::front<MatchingTagsList>;
};

template <typename TagsList>
struct metavars_tag_impl<TagsList, tmpl::list<>> {
  using type = NoSuchType;
};

template <typename TagList, typename Tag>
using list_of_matching_tags = tmpl::conditional_t<
    std::is_same_v<Tag, ::Tags::DataBox>, tmpl::list<::Tags::DataBox>,
    tmpl::conditional_t<
        std::is_same_v<Tag, Parallel::Tags::Metavariables>,
        tmpl::list<Parallel::Tags::Metavariables>,
        tmpl::filter<TagList, std::is_base_of<tmpl::pin<Tag>, tmpl::_1>>>>;

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

template <typename TagsList>
struct const_item_type_impl<Parallel::Tags::Metavariables, TagsList, false> {
  using type = const typename detail::metavars_tag_impl<TagsList>::type::type&;
};

template <typename Tag, typename TagsList>
struct const_item_type_impl<Tag, TagsList, true> {
  using type = typename db::detail::ConvertToConst<std::decay_t<
      typename db::detail::first_matching_tag<TagsList, Tag>::type>>::type;
};
}  // namespace detail

template <typename Tag, typename TagsList>
using const_item_type =
    typename detail::const_item_type_impl<Tag, TagsList>::type;
}  // namespace db
