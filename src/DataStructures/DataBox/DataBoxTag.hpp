// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines classes SimpleTag, PrefixTag, ComputeTag and several
/// functions for retrieving tag info

#pragma once

#include <cstddef>
#include <ostream>
#include <string>

#include "ErrorHandling/Assert.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
namespace Tags {
template <typename TagsList>
struct Variables;
}  // namespace Tags
/// \endcond

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
/*!
 * \ingroup DataBoxGroup
 * \brief Tags for the DataBox inherit from this type
 *
 * \details
 * Used to mark a type as being a SimpleTag so that it can be used in a
 * DataBox.
 *
 * \derivedrequires
 * - type alias `type` of the type this SimpleTag represents
 * - static `std::string name()` method that returns a runtime name for the tag.
 *
 * \example
 * \snippet Test_DataBox.cpp databox_tag_example
 *
 * \see DataBox PrefixTag get_tag_name
 */
struct SimpleTag {};

/*!
 * \ingroup DataBoxGroup
 * \brief Tags that are base tags, i.e. a simple or compute tag must derive
 * off them for them to be useful
 *
 * Base tags do not need to contain type information, unlike simple
 * tags which must contain the type information. Base tags are designed so
 * that retrieving items from the DataBox or setting argument tags in compute
 * items can be done without any knowledge of the type of the item.
 *
 * To use the base mechanism the base tag must inherit off of
 * `BaseTag` and NOT `SimpleTag`. This is very important for the
 * implementation. Inheriting off both and not making the tag either a simple
 * item or compute item is undefined behavior and is likely to end in extremely
 * complicated compiler errors.
 */
struct BaseTag {};

/*!
 * \ingroup DataBoxGroup
 * \brief Marks an item as being a prefix to another tag
 *
 * \details
 * Used to mark a type as being a DataBoxTag where the `label` is a prefix to
 * the DataBoxTag that is a member type alias `tag`. A prefix tag must contain a
 * type alias named `type` with the type of the Tag it is a prefix to, as well
 * as a type alias `tag` that is the type of the Tag that this prefix tag is
 * a prefix for. A prefix tag must also have a `label` equal to the name of
 * the struct (tag).
 *
 * \derivedrequires
 * - type alias `tag` of the DataBoxTag that this tag is a prefix to
 * - type alias `type` that is the type that this PrefixTag holds
 * - static `std::string name()` method that returns a runtime name for the tag.
 *
 * \example
 * A PrefixTag tag has the structure:
 * \snippet Test_DataBox.cpp databox_prefix_tag_example
 *
 * The name used to retrieve a prefix tag from the DataBox is:
 * \snippet Test_DataBox.cpp databox_name_prefix
 *
 *
 * \see DataBox DataBoxTag get_tag_name ComputeTag
 */
struct PrefixTag {};

/*!
 * \ingroup DataBoxGroup
 * \brief Marks a DataBoxTag as being a compute item that executes a function
 *
 * \details
 * Compute items come in two forms: mutating and non-mutating. Mutating
 * compute items modify a stored value in order to reduce the number of memory
 * allocations done. For example, if a function would return a `Variables` or
 * `Tensor<DataVector...>` and is called every time step, then it would be
 * preferable to use a mutating compute item so that the values in the already
 * allocated memory can just be changed.
 * In contrast, non-mutating compute items simply return the new value after a
 * call (if the value is out-of-date), which is fine for infrequently called
 * compute items or ones that do not allocate data on the heap.
 *
 * A compute item tag contains a member named `function` that is either a
 * function pointer, or a static constexpr function. The compute item tag
 * must also have a `label`, same as the DataBox tags, and a type alias
 * `argument_tags` that is a typelist of the tags that will
 * be retrieved from the DataBox and whose data will be passed to the function
 * (pointer). Mutating compute item tags must also contain a type alias named
 * `return_type` that is the type the function is mutating. The type must be
 * default constructible.
 *
 * \example
 * Most non-mutating compute item tags will look similar to:
 * \snippet Test_DataBox.cpp databox_compute_item_tag_example
 * Note that the arguments can be empty:
 * \snippet Test_DataBox.cpp compute_item_tag_no_tags
 *
 * Mutating compute item tags are of the form:
 * \snippet Test_DataBox.cpp databox_mutating_compute_item_tag
 * where the function is:
 * \snippet Test_DataBox.cpp databox_mutating_compute_item_function
 *
 * You can also have `function` be a function instead of a function pointer,
 * which offers a lot of simplicity for very simple compute items.
 * \snippet Test_DataBox.cpp compute_item_tag_function
 *
 * \see DataBox SimpleTag get_tag_name PrefixTag
 */
struct ComputeTag {};

namespace DataBox_detail {
template <typename TagList, typename Tag>
using list_of_matching_tags = tmpl::conditional_t<
    cpp17::is_same_v<Tag, ::Tags::DataBox>, tmpl::list<::Tags::DataBox>,
    tmpl::filter<TagList, std::is_base_of<tmpl::pin<Tag>, tmpl::_1>>>;

template <typename TagList, typename Tag>
using first_matching_tag = tmpl::front<list_of_matching_tags<TagList, Tag>>;

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
}  // namespace DataBox_detail

/*!
 * \ingroup DataBoxGroup
 * \brief Get the name of a DataBoxTag, including prefixes
 *
 * \details
 * Given a DataBoxTag returns the name of the DataBoxTag as a std::string. If
 * the DataBoxTag is also a PrefixTag then the prefix is added.
 *
 * \tparam Tag the DataBoxTag whose name to get
 * \return string holding the DataBoxTag's name
 */
template <typename Tag>
std::string get_tag_name() {
  return Tag::name();
}

// @{
/*!
 * \ingroup DataBoxGroup
 * \brief Check if `Tag` derives off of db::ComputeTag
 */
template <typename Tag, typename = std::nullptr_t>
struct is_compute_item : std::false_type {};
/// \cond HIDDEN_SYMBOLS
template <typename Tag>
struct is_compute_item<Tag, Requires<cpp17::is_base_of_v<db::ComputeTag, Tag>>>
    : std::true_type {};
/// \endcond

template <typename Tag>
constexpr bool is_compute_item_v = is_compute_item<Tag>::value;
// @}

// @{
/*!
 * \ingroup DataBoxGroup
 * \brief Check if `Tag` is a non-base DataBox tag. I.e. a SimpleTag or a
 * ComputeTag
 */
template <typename Tag, typename = std::nullptr_t>
struct is_non_base_tag : std::false_type {};
/// \cond
template <typename Tag>
struct is_non_base_tag<Tag, Requires<cpp17::is_base_of_v<db::ComputeTag, Tag> or
                                     cpp17::is_base_of_v<db::SimpleTag, Tag>>>
    : std::true_type {};
/// \endcond

template <typename Tag>
constexpr bool is_non_base_tag_v = is_non_base_tag<Tag>::value;
// @}

// @{
/*!
 * \ingroup DataBoxGroup
 * \brief Check if `Tag` is a BaseTag, SimpleTag, or ComputeTag
 */
template <typename Tag, typename = std::nullptr_t>
struct is_tag : std::false_type {};
/// \cond
template <typename Tag>
struct is_tag<Tag, Requires<cpp17::is_base_of_v<db::ComputeTag, Tag> or
                            cpp17::is_base_of_v<db::SimpleTag, Tag> or
                            cpp17::is_base_of_v<db::BaseTag, Tag>>>
    : std::true_type {};
/// \endcond

template <typename Tag>
constexpr bool is_tag_v = is_tag<Tag>::value;
// @}

// @{
/*!
 * \ingroup DataBoxGroup
 * \brief Check if `Tag` is a base DataBox tag
 */
template <typename Tag, typename = std::nullptr_t>
struct is_base_tag : std::false_type {};
/// \cond HIDDEN_SYMBOLS
template <typename Tag>
struct is_base_tag<Tag, Requires<cpp17::is_base_of_v<db::BaseTag, Tag> and
                                 not cpp17::is_base_of_v<db::SimpleTag, Tag> and
                                 not is_compute_item_v<Tag>>> : std::true_type {
};
/// \endcond

template <typename Tag>
constexpr bool is_base_tag_v = is_base_tag<Tag>::value;
// @}

template <class T, class = void>
struct has_return_type_member : std::false_type {};
template <class T>
struct has_return_type_member<T, cpp17::void_t<typename T::return_type>>
    : std::true_type {};

/*!
 * \brief `true` if `T` has nested type alias named `return_type`
 */
template <class T>
constexpr bool has_return_type_member_v = has_return_type_member<T>::value;

namespace DataBox_detail {
template <typename TagList, typename Tag>
struct item_type_impl;

template <int SelectTagType>
struct dispatch_item_type;

template <typename ArgsList>
struct compute_item_type;

template <typename TagList, typename Tag>
struct item_type_impl {
  // item_type_impl is intentionally a lazy metafunction rather than a
  // metaclosure or a metacontinuation. The reason is that it is quite likely we
  // call `db::item_type<Tag>` multiple times within the same translation unit
  // and so we want to memoize the resulting type. However, we do not want to
  // memoize the dispatch calls.
  using type = typename dispatch_item_type<
      is_base_tag_v<Tag>
          ? 4
          : (is_compute_item_v<Tag>and has_return_type_member_v<Tag>)
                ? 3
                : is_compute_item_v<Tag>
                      ? 2
                      : cpp17::is_base_of_v<db::SimpleTag, Tag> ? 1 : 0>::
      template f<TagList, Tag>;
};

template <>
struct dispatch_item_type<0> {
  // Tag is not a tag. This is necessary for SFINAE friendliness, specifically
  // if someone calls Requires<tt::is_a_v<std::vector, db::item_type<Tag>>>
  // with, say Tag = double, then this should probably SFINAE away, not fail to
  // compile.
  template <typename TagList, typename Tag>
  using f = NoSuchType;
};

template <>
struct dispatch_item_type<1> {
  // simple item
  template <typename TagList, typename Tag>
  using f = typename Tag::type;
};

template <>
struct dispatch_item_type<2> {
  // compute item
  template <typename TagList, typename Tag>
  using f = typename compute_item_type<typename Tag::argument_tags>::template f<
      TagList, Tag>;
};

template <>
struct dispatch_item_type<3> {
  // mutating compute item
  template <typename TagList, typename Tag>
  using f = typename Tag::return_type;
};

template <>
struct dispatch_item_type<4> {
  // base tag item: retrieve the derived tag from the tag list then call
  // item_type_impl on the result.
  // We do not check that there is only one matching tag in the DataBox because
  // the uniqueness is only checked in get and mutate. The reason for that is
  // that it is fine to have multiple derived tags in the DataBox as long as
  // they all have the same type. We do not check that the types are all the
  // same, it is undefined behavior if they are not and the user's
  // responsibility.
  template <typename TagList, typename Tag>
  using f =
      typename item_type_impl<TagList, first_matching_tag<TagList, Tag>>::type;
};

CREATE_IS_CALLABLE(function)

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
      "brigand::list<item_type<PassedTags>...>>'");
};

template <typename... Args>
struct compute_item_type<tmpl::list<Args...>> {
  template <typename TagList, typename Tag>
  using f = decltype(
#ifdef SPECTRE_DEBUG
      (void)check_compute_item_is_invokable<
          Tag, tmpl::list<Args...>,
          tmpl::list<typename item_type_impl<TagList, Args>::type...>>{},
#endif  // SPECTRE_DEBUG
      Tag::function(
          std::declval<typename item_type_impl<TagList, Args>::type>()...));
};
}  // namespace DataBox_detail

/*!
 * \ingroup DataBoxGroup
 * \brief Get the type that is returned by the `Tag`. If it is a base tag then a
 * `TagList` must be passed as a second argument.
 */
template <typename Tag, typename TagList = NoSuchType>
using item_type = typename DataBox_detail::item_type_impl<TagList, Tag>::type;

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
        : cpp17::is_base_of_v<db::PrefixTag, Tag> ? DispatchTagType::Prefix
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
        cpp17::is_same_v<typename InnerPrefix<InnerTag, InnerArgs...>::tag,
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
  static_assert(cpp17::is_base_of_v<db::SimpleTag, UnprefixedTag>,
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
    Tag, cpp17::is_base_of_v<db::PrefixTag, Tag>>::type;

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
/// \snippet Test_DataBoxTag.cpp variables_tag_with_tags_list
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

/// \ingroup DataBoxGroup
/// Struct that can be specialized to allow DataBox items to have
/// subitems.  Specializations must define:
/// * `using type = tmpl::list<...>` listing the subtags of `Tag`
/// * A static member function to initialize a subitem of a simple
///   item:
///   ```
///   template <typename Subtag>
///   static void create_item(
///       const gsl::not_null<item_type<Tag>*> parent_value,
///       const gsl::not_null<item_type<Subtag>*> sub_value) noexcept;
///   ```
///   Mutating the subitems must also modify the main item.
/// * A static member function evaluating a subitem of a compute
///   item:
///   ```
///   template <typename Subtag>
///   static item_type<Subtag> create_compute_item(
///       const item_type<Tag>& parent_value) noexcept;
///   ```
template <typename TagList, typename Tag, typename = std::nullptr_t>
struct Subitems {
  using type = tmpl::list<>;
};

/// \ingroup DataBoxGroup
/// Split a tag into its subitems.  `Tag` cannot be a base tag.
template <typename Tag, typename TagList = NoSuchType>
using split_tag = tmpl::conditional_t<
    tmpl::size<typename Subitems<TagList, Tag>::type>::value == 0,
    tmpl::list<Tag>, typename Subitems<TagList, Tag>::type>;

/// \ingroup DataBoxTagsGroup
/// \brief `true_type` if the prefix tag wraps the specified tag, `false_type`
/// otherwise. Can be used with `tmpl::filter` to extract a subset of a
/// `tmpl::list` of prefix tags which wrap a specified tag.
///
/// \snippet Test_DataBoxTag.cpp prefix_tag_wraps_specified_tag
template <typename PrefixTag, typename Tag>
struct prefix_tag_wraps_specified_tag
    : std::is_same<Tag, typename PrefixTag::tag> {};
}  // namespace db
