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

namespace db {

/*!
 * \ingroup DataBoxGroup
 * \brief The string used to give a runtime name to a DataBoxTag
 */
using DataBoxString_t = const char* const;

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
 * - `static constexpr DataBoxString_t` that is the same as the type name
 *    and named `label`
 *
 * \example
 * \snippet Test_DataBox.cpp databox_tag_example
 *
 * \see DataBox DataBoxPrefix DataBoxString_t get_tag_name
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
 * - DataBoxString_t `label` that is the prefix to the `tag`
 *
 * \example
 * A DataBoxPrefix tag has the structure:
 * \snippet Test_DataBox.cpp databox_prefix_tag_example
 *
 * The name used to retrieve a prefix tag from the DataBox is:
 * \snippet Test_DataBox.cpp databox_name_prefix
 *
 *
 * \see DataBox DataBoxTag DataBoxString_t get_tag_name ComputeItemTag
 */
struct DataBoxPrefix : DataBoxTag {};

/*!
 * \ingroup DataBoxGroup
 * \brief Marks a DataBoxTag as being a compute item that executes a function
 *
 * \details
 * A compute item tag contains a member named `function` that is either a
 * function pointer, or a static constexpr function. The compute item tag
 * must also have a `label`, same as the DataBox tags, and a type alias
 * `argument_tags` that is a typelist of the tags that will
 * be retrieved from the DataBox and whose data will be passed to the function
 * (pointer).
 *
 * \example
 * Most compute item tags will look similar to:
 * \snippet Test_DataBox.cpp databox_compute_item_tag_example
 * Note that the arguments can be empty:
 * \snippet Test_DataBox.cpp compute_item_tag_no_tags
 *
 * You can also have `function` be a function instead of a function pointer,
 * which offers a lot of simplicity for very simple compute items.
 * \snippet Test_DataBox.cpp compute_item_tag_function
 *
 * \see DataBox DataBoxTag DataBoxString_t get_tag_name DataBoxPrefix
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
 * \brief Check if a Tag label has type DataBoxString_t
 *
 * \details
 * For a type `T`, check that the static member variable named `label` has type
 * DataBoxString_t
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
    T, Requires<std::is_same<DataBoxString_t, decltype(T::label)>::value>>
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
struct compute_item_result_impl<Tag, Requires<is_compute_item<Tag>::value>> {
  using type =
      typename compute_item_result_helper<Tag,
                                          typename Tag::argument_tags>::type;
};

template <typename Tag>
struct compute_item_result_impl<Tag,
                                Requires<not is_compute_item<Tag>::value>> {
  using type = typename Tag::type;
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
}  // namespace db
