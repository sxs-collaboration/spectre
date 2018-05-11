// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines classes SimpleTag, PrefixTag, ComputeItemTag and several
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
 * \brief The string used to give a runtime name to a DataBoxTag
 */
using DataBoxString = const char* const;

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
 * - `static constexpr DataBoxString` that is the same as the type name
 *    and named `label`
 *
 * \example
 * \snippet Test_DataBox.cpp databox_tag_example
 *
 * \see DataBox PrefixTag DataBoxString get_tag_name
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
 * - DataBoxString `label` that is the prefix to the `tag`
 *
 * \example
 * A PrefixTag tag has the structure:
 * \snippet Test_DataBox.cpp databox_prefix_tag_example
 *
 * The name used to retrieve a prefix tag from the DataBox is:
 * \snippet Test_DataBox.cpp databox_name_prefix
 *
 *
 * \see DataBox DataBoxTag DataBoxString get_tag_name ComputeItemTag
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
 * \see DataBox SimpleTag DataBoxString get_tag_name PrefixTag
 */
struct ComputeItemTag {};

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

// @{
/*!
 * \ingroup DataBoxGroup
 * \brief Check if a Tag has a label
 *
 * \details
 * Check if a type `T` has a static member variable named `label`.
 *
 * \usage
 * For any type `T`
 * \code
 * using result = db::detail::tag_has_label<T>;
 * \endcode
 * \metareturns
 * cpp17::bool_constant
 *
 * \see tag_label_correct_type
 * \tparam T the type to check
 */
template <typename T, typename = void>
struct tag_has_label : std::false_type {};
/// \cond HIDDEN_SYMBOLS
template <typename T>
struct tag_has_label<T, cpp17::void_t<decltype(T::label)>> : std::true_type {};
/// \endcond

template <typename Tag>
constexpr bool tag_has_label_v = tag_has_label<Tag>::value;
// @}

// @{
/*!
 * \ingroup DataBoxGroup
 * \brief Check if a Tag label has type DataBoxString
 *
 * \details
 * For a type `T`, check that the static member variable named `label` has type
 * DataBoxString
 *
 * \usage
 * For any type `T`
 * \code
 * using result = db::tag_label_correct_type<T>;
 * \endcode
 * \metareturns
 * cpp17::bool_constant
 *
 * \tparam T the type to check
 */
template <typename T, typename = std::nullptr_t>
struct tag_label_correct_type : std::false_type {};
/// \cond HIDDEN_SYMBOLS
template <typename T>
struct tag_label_correct_type<
    T, Requires<cpp17::is_same_v<DataBoxString, decltype(T::label)>>>
    : std::true_type {};
/// \endcond
// @}

struct check_tag_labels {
  template <typename T>
  void operator()(tmpl::type_<T> /*meta*/) {
    ASSERT(pretty_type::get_name<T>().find(std::string(T::label)) !=
               std::string::npos,
           "Failed to match the Tag label " << std::string(T::label)
                                            << " with its type name "
                                            << pretty_type::get_name<T>());
  }
};
}  // namespace DataBox_detail

// @{
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
template <typename Tag,
          Requires<not cpp17::is_base_of_v<PrefixTag, Tag>> = nullptr>
std::string get_tag_name() {
  return std::string(Tag::label);
}
/// \cond HIDDEN_SYMBOLS
template <typename Tag, Requires<cpp17::is_base_of_v<PrefixTag, Tag>> = nullptr>
std::string get_tag_name() {
  return std::string(Tag::label) + get_tag_name<typename Tag::tag>();
}
/// \endcond
// @}

namespace detail {
// @{
/*!
 * \ingroup DataBoxGroup
 * \brief Compute a hash of the `label` of a DataBoxTag
 *
 * \details
 * Given a DataBoxTag returns a `value` of type `value_type` which is the hash
 * of the label of the DataBoxTag, including any prefix. The hashing algorithm
 * is based on what is used for std::typeinfo in libcxx v3.9.0.
 *
 * \tparam Tag the DataBoxTag whose name to get
 * \metareturns
 * value member of type `value_type` representing the hash of the DataBoxTag
 */
template <typename Tag, typename = std::nullptr_t>
struct hash_databox_tag {
  using value_type = size_t;
  static constexpr value_type value = cstring_hash(Tag::label);
  using type = std::integral_constant<value_type, value>;
};
/// \cond HIDDEN_SYMBOLS
template <typename Tag>
struct hash_databox_tag<Tag, Requires<cpp17::is_base_of_v<PrefixTag, Tag>>> {
  using value_type = size_t;
  static constexpr value_type value =
      cstring_hash(Tag::label) * hash_databox_tag<typename Tag::tag>::value;
  using type = std::integral_constant<value_type, value>;
};

template <typename Tag>
struct hash_databox_tag<Tag, Requires<tt::is_a_v<::Tags::Variables, Tag>>> {
  using value_type = size_t;
  using reduced_hash = tmpl::fold<
      typename Tag::type::tags_list, tmpl::size_t<0>,
      tmpl::bind<tmpl::plus, tmpl::_state, hash_databox_tag<tmpl::_element>>>;
  static constexpr value_type value =
      cstring_hash(Tag::label) * reduced_hash::value;
  using type = std::integral_constant<value_type, value>;
};
/// \endcond
// @}
}  // namespace detail

// @{
/*!
 * \ingroup DataBoxGroup
 * \brief Check if `Tag` derives off of db::ComputeItemTag
 */
template <typename Tag, typename = std::nullptr_t>
struct is_compute_item : std::false_type {};
/// \cond HIDDEN_SYMBOLS
template <typename Tag>
struct is_compute_item<Tag,
                       Requires<cpp17::is_base_of_v<db::ComputeItemTag, Tag>>>
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
struct is_non_base_tag<Tag,
                       Requires<cpp17::is_base_of_v<db::ComputeItemTag, Tag> or
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
struct is_tag<Tag, Requires<cpp17::is_base_of_v<db::ComputeItemTag, Tag> or
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

namespace DataBox_detail {
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

namespace detail {
template <bool IsVariables>
struct add_tag_prefix_impl;

template <>
struct add_tag_prefix_impl<false> {
  template <template <typename...> class Prefix, typename Tag, typename... Args>
  using f = Prefix<Tag, Args...>;
};

template <>
struct add_tag_prefix_impl<true> {
  template <template <typename...> class Prefix, typename Tag, typename... Args>
  using f = Prefix<
      Tags::Variables<wrap_tags_in<Prefix, typename Tag::tags_list, Args...>>,
      Args...>;
};

template <typename>
struct remove_tag_prefix_impl;

template <typename UnprefixedTag, template <typename...> class Prefix,
          typename... Args>
struct remove_tag_prefix_impl<Prefix<UnprefixedTag, Args...>> {
  static_assert(cpp17::is_base_of_v<db::SimpleTag, UnprefixedTag>,
                "Unwrapped tag is not a DataBoxTag");
  using type = UnprefixedTag;
};

template <typename... VariablesTags, template <typename...> class Prefix,
          typename... Args>
struct remove_tag_prefix_impl<
    Prefix<Tags::Variables<tmpl::list<VariablesTags...>>, Args...>> {
  using type = Tags::Variables<
      tmpl::list<typename remove_tag_prefix_impl<VariablesTags>::type...>>;
};
}  // namespace detail

/// \ingroup DataBoxTagsGroup
/// Wrap `Tag` in `Prefix<_, Args...>`, also wrapping variables tags
/// if `Tag` is a `Tags::Variables`.
template <template <typename...> class Prefix, typename Tag, typename... Args>
using add_tag_prefix = typename detail::add_tag_prefix_impl<
    tt::is_a_v<Tags::Variables, Tag>>::template f<Prefix, Tag, Args...>;

/// \ingroup DataBoxTagsGroup
/// Remove a prefix from `Tag`, also removing it from the variables
/// tags if the unwrapped tag is a `Tags::Variables`.
template <typename Tag>
using remove_tag_prefix = typename detail::remove_tag_prefix_impl<Tag>::type;

namespace databox_detail {
template <class Tag, bool IsPrefix = false>
struct remove_all_prefixes {
  using type = Tag;
};

template <template <class...> class F, class Tag, class... Args>
struct remove_all_prefixes<F<Tag, Args...>, true> {
  using type = typename remove_all_prefixes<
      Tag, cpp17::is_base_of_v<db::PrefixTag, Tag>>::type;
};
}  // namespace databox_detail

/// \ingroup DataBoxGroup
/// Completely remove all prefix tags from a Tag
template <typename Tag>
using remove_all_prefixes = typename databox_detail::remove_all_prefixes<
    Tag, cpp17::is_base_of_v<db::PrefixTag, Tag>>::type;

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
}  // namespace db
