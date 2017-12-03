// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines classes DataBoxTag, DataBoxPrefix, ComputeItemTag and several
/// functions for retrieving tag info

#pragma once

#include <ostream>
#include <string>

#include "ErrorHandling/Assert.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <typename TagsList>
class Variables;

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
 * Used to mark a type as being a DataBoxTag so that it can be used in a
 * DataBox.
 *
 * \derivedrequires
 * - type alias `type` of the type this DataBoxTag represents
 * - `static constexpr DataBoxString` that is the same as the type name
 *    and named `label`
 *
 * \example
 * \snippet Test_DataBox.cpp databox_tag_example
 *
 * \see DataBox DataBoxPrefix DataBoxString get_tag_name
 */
struct DataBoxTag {};

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
 * - type alias `type` that is the type that this DataBoxPrefix holds
 * - DataBoxString `label` that is the prefix to the `tag`
 *
 * \example
 * A DataBoxPrefix tag has the structure:
 * \snippet Test_DataBox.cpp databox_prefix_tag_example
 *
 * The name used to retrieve a prefix tag from the DataBox is:
 * \snippet Test_DataBox.cpp databox_name_prefix
 *
 *
 * \see DataBox DataBoxTag DataBoxString get_tag_name ComputeItemTag
 */
struct DataBoxPrefix : DataBoxTag {};

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
 * \see DataBox DataBoxTag DataBoxString get_tag_name DataBoxPrefix
 */
struct ComputeItemTag : DataBoxTag {};

namespace detail {
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
    T, Requires<std::is_same<DataBoxString, decltype(T::label)>::value>>
    : std::true_type {};
/// \endcond
// @}

struct check_tag_labels {
  using value_type = bool;
  value_type value{false};
  template <typename T>
  void operator()(tmpl::type_<T> /*meta*/) {
    bool correct = pretty_type::get_name<T>().find(std::string(T::label)) !=
                   std::string::npos;
    value |= correct;
    ASSERT(correct,
           "Failed to match the Tag label " << std::string(T::label)
                                            << " with its type name "
                                            << pretty_type::get_name<T>());
  }
};
}  // namespace detail

// @{
/*!
 * \ingroup DataBoxGroup
 * \brief Get the name of a DataBoxTag, including prefixes
 *
 * \details
 * Given a DataBoxTag returns the name of the DataBoxTag as a std::string. If
 * the DataBoxTag is also a DataBoxPrefix then the prefix is added.
 *
 * \tparam Tag the DataBoxTag whose name to get
 * \return string holding the DataBoxTag's name
 */
template <typename Tag,
          Requires<not std::is_base_of<DataBoxPrefix, Tag>::value> = nullptr>
std::string get_tag_name() {
  return std::string(Tag::label);
}
/// \cond HIDDEN_SYMBOLS
template <typename Tag,
          Requires<std::is_base_of<DataBoxPrefix, Tag>::value> = nullptr>
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
struct hash_databox_tag<Tag,
                        Requires<std::is_base_of<DataBoxPrefix, Tag>::value>> {
  using value_type = size_t;
  static constexpr value_type value =
      cstring_hash(Tag::label) * hash_databox_tag<typename Tag::tag>::value;
  using type = std::integral_constant<value_type, value>;
};

template <typename Tag>
struct hash_databox_tag<Tag,
                        Requires<tt::is_a<::Tags::Variables, Tag>::value>> {
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

/*!
 * \ingroup DataBoxGroup
 *
 * \details
 * Predicate that inherits from std::true_type if
 * `hash_databox_tag<DataBoxTag1>::value <
 * hash_databox_tag<DataBoxTag2>::value` otherwise inherits from
 * std::false_type.
 *
 * \tparam DataBoxTag1 the left operand
 * \tparam DataBoxTag2 the right operand
 */
template <typename DataBoxTag1, typename DataBoxTag2>
struct databox_tag_less : tmpl::bool_<(hash_databox_tag<DataBoxTag1>::value <
                                       hash_databox_tag<DataBoxTag2>::value)> {
};
}  // namespace detail

/*!
 * \ingroup DataBoxGroup
 * \brief Check if `T` derives off of db::ComputeItemTag
 */
template <typename T, typename = std::nullptr_t>
struct is_compute_item : std::false_type {};
/// \cond HIDDEN_SYMBOLS
template <typename T>
struct is_compute_item<T,
                       Requires<std::is_base_of<db::ComputeItemTag, T>::value>>
    : std::true_type {};
/// \endcond

/*!
 * \ingroup DataBoxGroup
 * \brief Check if `T` derives off of db::ComputeItemTag
 */
template <typename T>
constexpr bool is_compute_item_v = is_compute_item<T>::value;

namespace detail {
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

template <typename Tag, typename = std::nullptr_t>
struct compute_item_result_impl;

template <typename Tag, typename ArgsList>
struct compute_item_result_helper;

template <typename Tag, template <typename...> class ArgumentsList,
          typename... Args>
struct compute_item_result_helper<Tag, ArgumentsList<Args...>> {
  static_assert(
      tt::is_callable<decltype(Tag::function),
                      typename compute_item_result_impl<Args>::type...>::value,
      "The compute item is not callable with the types that the tags hold. The "
      "compute item tag that is the problem should be shown in the first line "
      "after the static assert error: "
      R"("detail::compute_item_result_helper<TheItemThatsFailingToBeCalled,)"
      R"(list<PassedTags...> >")");
  using type =
      decltype(std::declval<std::remove_pointer_t<decltype(Tag::function)>>()(
          std::declval<typename compute_item_result_impl<Args>::type>()...));
};

template <typename Tag>
struct compute_item_result_impl<
    Tag,
    Requires<is_compute_item_v<Tag> and not has_return_type_member_v<Tag>>> {
  using type =
      typename compute_item_result_helper<Tag,
                                          typename Tag::argument_tags>::type;
};

template <typename Tag>
struct compute_item_result_impl<
    Tag, Requires<is_compute_item_v<Tag> and has_return_type_member_v<Tag>>> {
  using type = typename Tag::return_type;
};

template <typename Tag>
struct compute_item_result_impl<
    Tag, Requires<not is_compute_item_v<Tag> and
                  std::is_base_of<db::DataBoxTag, Tag>::value>> {
  using type = typename Tag::type;
};

template <typename Tag>
struct compute_item_result_impl<
    Tag, Requires<not std::is_base_of<db::DataBoxTag, Tag>::value>> {
  using type = NoSuchType;
};
}  // namespace detail

/*!
 * \ingroup DataBoxGroup
 * \brief Get the type that is returned by a Compute Item
 */
template <typename T>
using item_type = typename detail::compute_item_result_impl<T>::type;

/*!
 * \ingroup DataBoxGroup
 * \brief Check if a Compute Item is "simple"
 */
template <typename T, typename = std::nullptr_t>
struct is_simple_compute_item : std::false_type {};
template <typename T>
struct is_simple_compute_item<T,
                              Requires<is_compute_item<T>::value and
                                       not tt::is_a_v<Variables, item_type<T>>>>
    : std::true_type {};

/*!
 * \ingroup DataBoxGroup
 * \brief Check if a Compute Item returns a Variables class
 */
template <typename T, typename = std::nullptr_t>
struct is_variables_compute_item : std::false_type {};
template <typename T>
struct is_variables_compute_item<
    T,
    Requires<is_compute_item<T>::value and tt::is_a_v<Variables, item_type<T>>>>
    : std::true_type {};

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
  static_assert(cpp17::is_base_of_v<db::DataBoxTag, UnprefixedTag>,
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
      Tag, cpp17::is_base_of_v<db::DataBoxPrefix, Tag>>::type;
};
}  // namespace databox_detail

/// \ingroup DataBoxGroup
/// Completely remove all prefix tags from a Tag
template <typename Tag>
using remove_all_prefixes = typename databox_detail::remove_all_prefixes<
    Tag, cpp17::is_base_of_v<db::DataBoxPrefix, Tag>>::type;
}  // namespace db
