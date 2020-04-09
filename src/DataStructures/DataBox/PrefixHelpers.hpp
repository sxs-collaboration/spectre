// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
namespace db {
struct PrefixTag;
struct SimpleTag;
}  // namespace db

namespace Tags {
template <typename TagsList>
struct Variables;
}  // namespace Tags
/// \endcond

namespace db {

/// \ingroup DataBoxTagsGroup
/// \brief Create a new list of Tags by wrapping each tag in `TagList` using the
/// `Wrapper`.
template <template <typename...> class Wrapper, typename TagList,
          typename... Args>
using wrap_tags_in =
    tmpl::transform<TagList, tmpl::bind<Wrapper, tmpl::_1, tmpl::pin<Args>...>>;

namespace DataBox_detail {
enum class DispatchTagType {
  Variables,
  Prefix,
  Other,
};

template <typename Tag>
constexpr DispatchTagType tag_type =
    tt::is_a_v<Tags::Variables, Tag>
        ? DispatchTagType::Variables
        : std::is_base_of_v<db::PrefixTag, Tag> ? DispatchTagType::Prefix
                                                : DispatchTagType::Other;

template <DispatchTagType TagType>
struct add_tag_prefix_impl;

// Call the appropriate impl based on the type of the tag being
// prefixed.
template <template <typename...> class Prefix, typename Tag, typename... Args>
using dispatch_add_tag_prefix_impl =
    typename add_tag_prefix_impl<tag_type<Tag>>::template f<Prefix, Tag,
                                                            Args...>;

template <>
struct add_tag_prefix_impl<DispatchTagType::Other> {
  template <template <typename...> class Prefix, typename Tag, typename... Args>
  using f = Tag;
};

template <>
struct add_tag_prefix_impl<DispatchTagType::Prefix> {
  template <template <typename...> class Prefix, typename Tag, typename... Args>
  struct prefix_wrapper_helper;

  template <template <typename...> class Prefix,
            template <typename...> class InnerPrefix, typename InnerTag,
            typename... InnerArgs, typename... Args>
  struct prefix_wrapper_helper<Prefix, InnerPrefix<InnerTag, InnerArgs...>,
                               Args...> {
    static_assert(
        std::is_same_v<typename InnerPrefix<InnerTag, InnerArgs...>::tag,
                       InnerTag>,
        "Inconsistent values of prefixed tag");
    using type =
        InnerPrefix<dispatch_add_tag_prefix_impl<Prefix, InnerTag, Args...>,
                    InnerArgs...>;
  };

  template <template <typename...> class Prefix, typename Tag, typename... Args>
  using f = typename prefix_wrapper_helper<Prefix, Tag, Args...>::type;
};

template <>
struct add_tag_prefix_impl<DispatchTagType::Variables> {
  template <template <typename...> class Prefix, typename Tag, typename... Args>
  using f =
      Tags::Variables<wrap_tags_in<Prefix, typename Tag::tags_list, Args...>>;
};

// Implementation of remove_tag_prefix
template <typename>
struct remove_tag_prefix_impl;

template <DispatchTagType TagType>
struct remove_variables_prefix;

template <typename Tag>
using dispatch_remove_variables_prefix =
    typename remove_variables_prefix<tag_type<Tag>>::template f<Tag>;

template <>
struct remove_variables_prefix<DispatchTagType::Other> {
  template <typename Tag>
  using f = Tag;
};

template <>
struct remove_variables_prefix<DispatchTagType::Prefix> {
  template <typename Tag>
  struct helper;

  template <template <typename...> class Prefix, typename Tag, typename... Args>
  struct helper<Prefix<Tag, Args...>> {
    using type = Prefix<dispatch_remove_variables_prefix<Tag>, Args...>;
  };

  template <typename Tag>
  using f = typename helper<Tag>::type;
};

template <>
struct remove_variables_prefix<DispatchTagType::Variables> {
  template <typename Tag>
  using f = Tags::Variables<tmpl::transform<typename Tag::tags_list,
                                            remove_tag_prefix_impl<tmpl::_1>>>;
};

template <typename UnprefixedTag, template <typename...> class Prefix,
          typename... Args>
struct remove_tag_prefix_impl<Prefix<UnprefixedTag, Args...>> {
  static_assert(std::is_base_of_v<db::SimpleTag, UnprefixedTag>,
                "Unwrapped tag is not a DataBoxTag");
  using type = dispatch_remove_variables_prefix<UnprefixedTag>;
};
}  // namespace DataBox_detail

/// \ingroup DataBoxTagsGroup
/// Wrap `Tag` in `Prefix<_, Args...>`, also wrapping variables tags
/// if `Tag` is a `Tags::Variables`.
template <template <typename...> class Prefix, typename Tag, typename... Args>
using add_tag_prefix =
    Prefix<DataBox_detail::dispatch_add_tag_prefix_impl<Prefix, Tag, Args...>,
           Args...>;

/// \ingroup DataBoxTagsGroup
/// Remove a prefix from `Tag`, also removing it from the variables
/// tags if the unwrapped tag is a `Tags::Variables`.
template <typename Tag>
using remove_tag_prefix =
    typename DataBox_detail::remove_tag_prefix_impl<Tag>::type;

namespace DataBox_detail {
template <class Tag, bool IsPrefix>
struct remove_all_prefixes_impl;
}  // namespace DataBox_detail

/// \ingroup DataBoxGroup
/// Completely remove all prefix tags from a Tag
template <typename Tag>
using remove_all_prefixes = typename DataBox_detail::remove_all_prefixes_impl<
    Tag, std::is_base_of_v<db::PrefixTag, Tag>>::type;

namespace DataBox_detail {
template <class Tag>
struct remove_all_prefixes_impl<Tag, false> {
  using type = Tag;
};

template <class Tag>
struct remove_all_prefixes_impl<Tag, true> {
  using type = remove_all_prefixes<remove_tag_prefix<Tag>>;
};

// Implementation of variables_tag_with_tags_list
template <DispatchTagType TagType>
struct variables_tag_with_tags_list_impl;
}  // namespace DataBox_detail

/// \ingroup DataBoxGroup
/// Change the tags contained in a possibly prefixed Variables tag.
/// \example
/// \snippet Test_PrefixHelpers.cpp variables_tag_with_tags_list
template <typename Tag, typename NewTagsList>
using variables_tag_with_tags_list =
    typename DataBox_detail::variables_tag_with_tags_list_impl<
        DataBox_detail::tag_type<Tag>>::template f<Tag, NewTagsList>;

namespace DataBox_detail {
// Implementation of variables_tag_with_tags_list
template <>
struct variables_tag_with_tags_list_impl<DispatchTagType::Variables> {
  template <typename Tag, typename NewTagsList>
  using f = Tags::Variables<NewTagsList>;
};

template <>
struct variables_tag_with_tags_list_impl<DispatchTagType::Prefix> {
  template <typename Tag, typename NewTagsList>
  struct helper;

  template <template <typename...> class Prefix, typename Tag, typename... Args,
            typename NewTagsList>
  struct helper<Prefix<Tag, Args...>, NewTagsList> {
    using type =
        Prefix<variables_tag_with_tags_list<Tag, NewTagsList>, Args...>;
  };

  template <typename Tag, typename NewTagsList>
  using f = typename helper<Tag, NewTagsList>::type;
};

// Implementation of get_variables_tags_list
template <DispatchTagType TagType>
struct get_variables_tags_list_impl;
}  // namespace DataBox_detail

template <typename Tag>
using get_variables_tags_list =
    typename DataBox_detail::get_variables_tags_list_impl<
        DataBox_detail::tag_type<Tag>>::template f<Tag>;

namespace DataBox_detail {
// Implementation of get_variables_tags_list
template <>
struct get_variables_tags_list_impl<DispatchTagType::Variables> {
  template <typename Tag>
  using f = typename Tag::tags_list;
};

template <>
struct get_variables_tags_list_impl<DispatchTagType::Prefix> {
  template <typename Tag>
  using f = get_variables_tags_list<typename Tag::tag>;
};
}  // namespace DataBox_detail
}  // namespace db
