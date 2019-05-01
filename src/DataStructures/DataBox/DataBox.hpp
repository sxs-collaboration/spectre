// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines classes and functions used for manipulating DataBox's

#pragma once

#include <algorithm>
#include <cassert>
#include <functional>
#include <pup.h>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Deferred.hpp"
#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "ErrorHandling/StaticAssert.hpp"
#include "Utilities/BoostHelpers.hpp"  // for pup variant
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/*!
 * \ingroup DataBoxGroup
 * \brief Namespace for DataBox related things
 */
namespace db {

// Forward declarations
/// \cond
template <typename TagsList>
class DataBox;
/// \endcond

// @{
/*!
 * \ingroup TypeTraitsGroup DataBoxGroup
 * \brief Determines if a type `T` is as db::DataBox
 *
 * \effects Inherits from std::true_type if `T` is a specialization of
 * db::DataBox, otherwise inherits from std::false_type
 * \example
 */
// \snippet Test_DataBox.cpp
template <typename T>
struct is_databox : std::false_type {};
/// \cond HIDDEN_SYMBOLS
template <typename... Tags>
struct is_databox<DataBox<tmpl::list<Tags...>>> : std::true_type {};
/// \endcond
// @}

namespace DataBox_detail {
template <class Tag, class Type>
class DataBoxLeaf;

template <class Tag, class Type>
class DataBoxLeaf {
  using value_type = Deferred<Type>;
  value_type value_;

  template <class T>
  static constexpr bool can_bind_reference() noexcept {
    using rem_ref_value_type = typename std::remove_reference<value_type>::type;
    using rem_ref_T = typename std::remove_reference<T>::type;
    using is_lvalue_type = std::integral_constant<
        bool, cpp17::is_lvalue_reference_v<T> or
                  cpp17::is_same_v<std::reference_wrapper<rem_ref_value_type>,
                                   rem_ref_T> or
                  cpp17::is_same_v<std::reference_wrapper<
                                       std::remove_const_t<rem_ref_value_type>>,
                                   rem_ref_T>>;
    return not cpp17::is_reference_v<value_type> or
           (cpp17::is_lvalue_reference_v<value_type> and
            is_lvalue_type::value) or
           (cpp17::is_rvalue_reference_v<value_type> and
            not cpp17::is_lvalue_reference_v<T>);
  }

 public:
  constexpr DataBoxLeaf() noexcept(
      cpp17::is_nothrow_default_constructible_v<value_type>)
      : value_() {
    static_assert(!cpp17::is_reference_v<value_type>,
                  "Cannot default construct a reference element in a "
                  "DataBox");
  }

  // clang-tidy: forwarding references are hard
  template <class T,
            Requires<not cpp17::is_same_v<std::decay_t<T>, DataBoxLeaf> and
                     cpp17::is_constructible_v<value_type, T&&>> = nullptr>
  constexpr explicit DataBoxLeaf(T&& t) noexcept(  // NOLINT
      cpp17::is_nothrow_constructible_v<value_type, T&&>)
      : value_(std::forward<T>(t)) {  // NOLINT
    static_assert(can_bind_reference<T>(),
                  "Cannot construct an lvalue reference with an rvalue");
  }

  constexpr DataBoxLeaf(DataBoxLeaf const& /*rhs*/) = default;
  constexpr DataBoxLeaf(DataBoxLeaf&& /*rhs*/) = default;
  constexpr DataBoxLeaf& operator=(DataBoxLeaf const& rhs) noexcept(
      noexcept(value_ = rhs.value_)) {
    if (this != &rhs) {
      value_ = rhs.value_;
    }
    return *this;
  }
  constexpr DataBoxLeaf& operator=(DataBoxLeaf&& rhs) noexcept(
      noexcept(value_ = std::move(rhs.value_))) {
    if (this != &rhs) {
      value_ = std::move(rhs.value_);
    }
    return *this;
  }

  ~DataBoxLeaf() = default;

  constexpr value_type& get() noexcept { return value_; }
  constexpr const value_type& get() const noexcept { return value_; }

  // clang-tidy: runtime-references
  void pup(PUP::er& p) { p | value_; }  // NOLINT
};

template <typename Element>
struct extract_expand_simple_subitems {
  using type =
      tmpl::push_front<typename Subitems<NoSuchType, Element>::type, Element>;
};

// Given a typelist of items List, returns a new typelist containing
// the items and all of their subitems.
template <typename List>
using expand_simple_subitems = tmpl::flatten<
    tmpl::transform<List, extract_expand_simple_subitems<tmpl::_1>>>;

namespace detail {
constexpr int select_expand_subitems_impl(const size_t pack_size) noexcept {
  // selects the appropriate fast track based on the pack size. Fast tracks for
  // 2 and 4 limit the DataBox to about 800 items. This could be increased by
  // adding fast tracks for say 8 and 64.
  return pack_size >= 4 ? 3 : pack_size >= 2 ? 2 : static_cast<int>(pack_size);
}

// expand_subitems_from_list_impl is a left fold, but Brigand doesn't do folds
// through aliases, so it's cheaper this way
template <int FastTrackSelector, template <typename...> class F>
struct expand_subitems_impl;

template <typename FulltagList, typename TagList, typename Element>
using expand_subitems_impl_helper = tmpl::append<
    tmpl::push_back<TagList, Element>,
    typename Subitems<tmpl::push_back<TagList, Element>, Element>::type>;

template <template <typename...> class F>
struct expand_subitems_impl<0, F> {
  template <typename FullTagList, typename TagList>
  using f = TagList;
};

template <template <typename...> class F>
struct expand_subitems_impl<1, F> {
  // The compile time here could be improved by having Subitems have a nested
  // type alias that generates the list currently retrieved by `::type` on to
  // the next call of `expand_subitems_impl`
  template <typename FullTagList, typename TagList, typename Element,
            typename... Rest>
  using f = typename expand_subitems_impl<
      select_expand_subitems_impl(sizeof...(Rest)),
      F>::template f<FullTagList, F<FullTagList, TagList, Element>, Rest...>;
};

template <template <typename...> class F>
struct expand_subitems_impl<2, F> {
  template <typename FullTagList, typename TagList, typename Element0,
            typename Element1, typename... Rest>
  using f = typename expand_subitems_impl<
      select_expand_subitems_impl(sizeof...(Rest)), F>::
      template f<FullTagList,
                 F<FullTagList, F<FullTagList, TagList, Element0>, Element1>,
                 Rest...>;
};

template <template <typename...> class F>
struct expand_subitems_impl<3, F> {
  template <typename FullTagList, typename TagList, typename Element0,
            typename Element1, typename Element2, typename Element3,
            typename... Rest>
  using f = typename expand_subitems_impl<
      select_expand_subitems_impl(sizeof...(Rest)), F>::
      template f<
          FullTagList,
          F<FullTagList,
            F<FullTagList,
              F<FullTagList, F<FullTagList, TagList, Element0>, Element1>,
              Element2>,
            Element3>,
          Rest...>;
};

/*!
 * Expands the `ComputeTagsList` into a parameter pack, and also makes the
 * decision about expanding the simple tags. The `ComputeTagsList` is expanded
 * into a parameter pack so that we can use type aliases to do the computation,
 * rather than structs. This turns out to be better for compilation speed.
 *
 * Once the `ComputeTagsList` is turned into a parameter pack, the
 * `expand_subitems_impl` struct's nested type alias is used for the recursive
 * computation. This recursive nature limits us to about 800 items in the
 * DataBox without the need for fast-tracking. If more are necessary, using
 * fast-tracking will easily allow it.
 */
template <class ComputeTagsList, bool ExpandSimpleTags>
struct expand_subitems;

template <class... ComputeTags>
struct expand_subitems<tmpl::list<ComputeTags...>, true> {
  template <typename SimpleTagsList>
  using f = typename detail::expand_subitems_impl<select_expand_subitems_impl(
                                                      sizeof...(ComputeTags)),
                                                  expand_subitems_impl_helper>::
      template f<tmpl::list<>, expand_simple_subitems<SimpleTagsList>,
                 ComputeTags...>;
};

template <class... ComputeTags>
struct expand_subitems<tmpl::list<ComputeTags...>, false> {
  template <typename SimpleTagsList>
  using f = typename detail::expand_subitems_impl<
      select_expand_subitems_impl(sizeof...(ComputeTags)),
      expand_subitems_impl_helper>::template f<tmpl::list<>, SimpleTagsList,
                                               ComputeTags...>;
};

// The compile time here could be improved by having Subitems have a nested
// type alias that generates the list currently retrieved by `::type` on to
// the next call of `expand_subitems_from_list_impl`
template <typename FullTagList, typename BuildingTagList, typename Element>
using expand_subitems_from_list_impl_helper =
    tmpl::append<tmpl::push_back<BuildingTagList, Element>,
                 typename Subitems<FullTagList, Element>::type>;

template <typename ComputeTagsList>
struct expanded_list_from_full_list_impl;

template <typename... ComputeTags>
struct expanded_list_from_full_list_impl<tmpl::list<ComputeTags...>> {
  template <typename FullTagList>
  using f =
      typename expand_subitems_impl<select_expand_subitems_impl(
                                        sizeof...(ComputeTags)),
                                    expand_subitems_from_list_impl_helper>::
          template f<FullTagList, tmpl::list<>, ComputeTags...>;
};

template <>
struct expanded_list_from_full_list_impl<tmpl::list<>> {
  template <typename FullTagList>
  using f = tmpl::list<>;
};
}  // namespace detail

/*!
 * \brief Returns a list of all the tags with the subitems expanded, but where
 * the types for compute items are grabbed from the FullTagList instead of the
 * tag list that is being built up.
 *
 * This is useful for generating a list with subitem-expanded compute items,
 * without having it be prefixed with the full tags list or the simple tags
 * list.
 */
template <typename FullTagList, typename TagsList>
using expand_subitems_from_list =
    typename detail::expanded_list_from_full_list_impl<TagsList>::template f<
        FullTagList>;

/*!
 * Expand on the subitems in SimpleTagsList and ComputeTagsList. For a subitem
 * `Varibles<Tag0, Tag1>` the order of the expanded tags is `Variables<Tag0,
 * Tag1>, Tag0, Tag1`. The simple tag list is only expanded if
 * `ExpandSimpleTags` is set to `true`, so if you already have an expanded
 * simple tag list, you can avoid double expansion.
 */
template <typename SimpleTagsList, typename ComputeTagsList,
          bool ExpandSimpleTags>
using expand_subitems = typename detail::expand_subitems<
    ComputeTagsList, ExpandSimpleTags>::template f<SimpleTagsList>;

template <typename TagList, typename Tag>
using has_subitems = tmpl::not_<
    std::is_same<typename Subitems<TagList, Tag>::type, tmpl::list<>>>;

template <typename ComputeTag, typename ArgumentTag,
          typename FoundComputeItemInBox>
struct report_missing_compute_item_argument {
  static_assert(cpp17::is_same_v<ComputeTag, void>,
                "A compute item's argument could not be found in the "
                "DataBox or was found multiple times.  See the first "
                "template argument for the compute item and the second "
                "for the missing argument.");
};

template <typename ComputeTag, typename ArgumentTag>
struct report_missing_compute_item_argument<ComputeTag, ArgumentTag,
                                            std::true_type> {
  using type = void;
};

template <typename TagList, typename ComputeTag>
struct create_dependency_graph {
#ifdef SPECTRE_DEBUG
  using argument_check_assertion =
      tmpl::transform<typename ComputeTag::argument_tags,
                      report_missing_compute_item_argument<
                          tmpl::pin<ComputeTag>, tmpl::_1,
                          DataBox_detail::has_unique_matching_tag<
                              tmpl::pin<TagList>, tmpl::_1>>>;
#endif  // SPECTRE_DEBUG
  // These edges record that a compute item's value depends on the
  // values of it's arguments.
  using compute_tag_argument_edges =
      tmpl::transform<typename ComputeTag::argument_tags,
                      tmpl::bind<tmpl::edge,
                                 tmpl::bind<DataBox_detail::first_matching_tag,
                                            tmpl::pin<TagList>, tmpl::_1>,
                                 tmpl::pin<ComputeTag>>>;
  // These edges record that the values of the subitems of a compute
  // item depend on the value of the compute item itself.
  using subitem_reverse_edges =
      tmpl::transform<typename Subitems<TagList, ComputeTag>::type,
                      tmpl::bind<tmpl::edge, tmpl::pin<ComputeTag>, tmpl::_1>>;

  using type = tmpl::append<compute_tag_argument_edges, subitem_reverse_edges>;
};
}  // namespace DataBox_detail

namespace DataBox_detail {
// Check if a tag has a name method
template <typename Tag, typename = std::nullptr_t>
struct tag_has_name {
  static_assert(cpp17::is_same_v<Tag, const void* const*>,
                "The tag does not have a static method 'name()' that returns a "
                "std::string. See the first template parameter of "
                "db::DataBox_detail::tag_has_name to see the problematic tag.");
};
template <typename Tag>
struct tag_has_name<
    Tag, Requires<cpp17::is_same_v<decltype(Tag::name()), std::string>>> {};

template <typename Tag, typename = std::nullptr_t>
struct check_simple_or_compute_tag {
  static_assert(cpp17::is_same_v<Tag, const void* const*>,
                "All tags added to a DataBox must derive off of db::SimpleTag "
                "or db::ComputeTag, you cannot add a base tag itself. See the "
                "first template parameter of "
                "db::DataBox_detail::check_simple_or_compute_tag to see "
                "the problematic tag.");
};
template <typename Tag>
struct check_simple_or_compute_tag<Tag, Requires<is_non_base_tag_v<Tag>>> {};
}  // namespace DataBox_detail

/*!
 * \ingroup DataBoxGroup
 * \brief A DataBox stores objects that can be retrieved by using Tags
 * \warning
 * The order of the tags in DataBoxes returned by db::create and
 * db::create_from depends on implementation-defined behavior, and
 * therefore should not be specified in source files. If explicitly
 * naming a DataBox type is necessary they should be generated using
 * db::compute_databox_type.
 *
 * \see db::create db::create_from
 *
 * @tparam Tags list of DataBoxTag's
 */
template <typename... Tags>
class DataBox<tmpl::list<Tags...>>
    : private DataBox_detail::DataBoxLeaf<
          Tags, db::item_type<Tags, tmpl::list<Tags...>>>... {
#ifdef SPECTRE_DEBUG
  static_assert(
      tmpl2::flat_all_v<is_non_base_tag_v<Tags>...>,
      "All structs used to Tag (compute) items in a DataBox must derive off of "
      "db::SimpleTag. Another static_assert will tell you which tag is the "
      "problematic one. Look for check_simple_or_compute_tag.");
#endif  // ifdef SPECTRE_DEBUG

 public:
  /*!
   * \brief A typelist (`tmpl::list`) of Tags that the DataBox holds
   */
  using tags_list = tmpl::list<Tags...>;

  /// A list of all the compute item tags, excluding their subitems
  using compute_item_tags =
      tmpl::filter<tags_list, db::is_compute_item<tmpl::_1>>;

  /// A list of all the compute items, including subitems from the compute items
  using compute_with_subitems_tags =
      DataBox_detail::expand_subitems_from_list<tags_list, compute_item_tags>;

  /// A list of all the simple items, including subitems from the simple
  /// items
  using simple_item_tags =
      tmpl::list_difference<tags_list, compute_with_subitems_tags>;

  /// A list of the simple items that have subitems, without expanding the
  /// subitems out
  using simple_subitems_tags = tmpl::filter<
      simple_item_tags,
      tmpl::bind<DataBox_detail::has_subitems, tmpl::pin<tags_list>, tmpl::_1>>;

  /// A list of the expanded simple subitems, not including the main Subitem
  /// tags themselves.
  ///
  /// Specifically, if there is a `Variables<Tag0, Tag1>`, then this list would
  /// contain `Tag0, Tag1`.
  using simple_only_expanded_subitems_tags = tmpl::flatten<tmpl::transform<
      simple_subitems_tags, db::Subitems<tmpl::pin<tags_list>, tmpl::_1>>>;

  /// \cond HIDDEN_SYMBOLS
  /*!
   * \note the default constructor is only used for serialization
   */
  DataBox() = default;
  DataBox(DataBox&& rhs) noexcept(
      tmpl2::flat_all_v<
          cpp17::is_nothrow_move_constructible_v<DataBox_detail::DataBoxLeaf<
              Tags, db::item_type<Tags, tmpl::list<Tags...>>>>...>) = default;
  DataBox& operator=(DataBox&& rhs) noexcept(
      tmpl2::flat_all_v<
          cpp17::is_nothrow_move_assignable_v<DataBox_detail::DataBoxLeaf<
              Tags, db::item_type<Tags, tmpl::list<Tags...>>>>...>) {
    if (&rhs != this) {
      ::expand_pack((get_deferred<Tags>() =
                         std::move(rhs.template get_deferred<Tags>()))...);
    }
    return *this;
  }
  DataBox(const DataBox& rhs) = delete;
  DataBox& operator=(const DataBox& rhs) = delete;
#ifdef SPECTRE_DEBUG
  // Destructor is used for triggering assertions
  ~DataBox() noexcept {
    EXPAND_PACK_LEFT_TO_RIGHT(DataBox_detail::tag_has_name<Tags>{});
    EXPAND_PACK_LEFT_TO_RIGHT(
        DataBox_detail::check_simple_or_compute_tag<Tags>{});
  }
#else   // ifdef SPECTRE_DEBUG
  ~DataBox() = default;
#endif  // ifdef SPECTRE_DEBUG

  /// \endcond

  /// \cond HIDDEN_SYMBOLS
  /// Retrieve the tag `Tag`, should be called by the free function db::get
  template <typename Tag,
            Requires<not cpp17::is_same_v<Tag, ::Tags::DataBox>> = nullptr>
  auto get() const noexcept -> const item_type<Tag, tags_list>&;

  /// Retrieve the tag `Tag`, should be called by the free function db::get
  template <typename Tag,
            Requires<cpp17::is_same_v<Tag, ::Tags::DataBox>> = nullptr>
  auto get() const noexcept -> const DataBox<tags_list>&;
  /// \endcond

  // clang-tidy: no non-const references
  void pup(PUP::er& p) noexcept {  // NOLINT
    using non_subitems_tags =
        tmpl::list_difference<simple_item_tags,
                              simple_only_expanded_subitems_tags>;

    // We do not send subitems for both simple items and compute items since
    // they can be reconstructed very cheaply.
    pup_impl(p, non_subitems_tags{}, compute_item_tags{});
  }

  template <typename Box, typename... KeepTags, typename... AddTags,
            typename... AddComputeTags, typename... Args>
  constexpr DataBox(Box&& old_box, tmpl::list<KeepTags...> /*meta*/,
                    tmpl::list<AddTags...> /*meta*/,
                    tmpl::list<AddComputeTags...> /*meta*/,
                    Args&&... args) noexcept;

  template <typename... TagsInArgsOrder, typename... FullItems,
            typename... ComputeTags, typename... FullComputeItems,
            typename... Args>
  constexpr DataBox(tmpl::list<TagsInArgsOrder...> /*meta*/,
                    tmpl::list<FullItems...> /*meta*/,
                    tmpl::list<ComputeTags...> /*meta*/,
                    tmpl::list<FullComputeItems...> /*meta*/,
                    Args&&... args) noexcept;
  /// \endcond

 private:
  template <typename... MutateTags, typename TagList, typename Invokable,
            typename... Args>
  // clang-tidy: redundant declaration
  friend void mutate(gsl::not_null<DataBox<TagList>*> box,             // NOLINT
                     Invokable&& invokable, Args&&... args) noexcept;  // NOLINT

  template <typename... SimpleTags>
  SPECTRE_ALWAYS_INLINE void copy_simple_items(
      const DataBox& box, tmpl::list<SimpleTags...> /*meta*/) noexcept;

  // Creates a copy with no aliasing of items.
  template <typename SimpleItemTags>
  DataBox deep_copy() const noexcept;

  template <typename TagsList_>
  // clang-tidy: redundant declaration
  friend SPECTRE_ALWAYS_INLINE constexpr DataBox<TagsList_>  // NOLINT
  create_copy(                                               // NOLINT
      const DataBox<TagsList_>& box) noexcept;

  /*!
   * \requires Type `T` is one of the Tags corresponding to an object stored in
   * the DataBox
   *
   * \note This should not be used outside of implementation details
   *
   * @return The lazy object corresponding to the Tag `T`
   */
  template <typename T>
  const Deferred<item_type<T, tags_list>>& get_deferred() const noexcept {
    return static_cast<const DataBox_detail::DataBoxLeaf<
        T, db::item_type<T, tags_list>>&>(*this)
        .get();
  }

  template <typename T>
  Deferred<item_type<T, tags_list>>& get_deferred() noexcept {
    return static_cast<
               DataBox_detail::DataBoxLeaf<T, db::item_type<T, tags_list>>&>(
               *this)
        .get();
  }

  // Adding compute items
  template <typename ParentTag>
  SPECTRE_ALWAYS_INLINE constexpr void add_sub_compute_item_tags_to_box(
      tmpl::list<> /*meta*/, std::false_type /*meta*/) noexcept {}
  template <typename ParentTag>
  SPECTRE_ALWAYS_INLINE constexpr void add_sub_compute_item_tags_to_box(
      tmpl::list<> /*meta*/, std::true_type /*meta*/) noexcept {}

  template <typename ParentTag, typename... Subtags>
  SPECTRE_ALWAYS_INLINE constexpr void add_sub_compute_item_tags_to_box(
      tmpl::list<Subtags...> /*meta*/,
      std::false_type /*has_return_type_member*/) noexcept;
  template <typename ParentTag, typename... Subtags>
  SPECTRE_ALWAYS_INLINE constexpr void add_sub_compute_item_tags_to_box(
      tmpl::list<Subtags...> /*meta*/,
      std::true_type /*has_return_type_member*/) noexcept;

  template <typename ComputeItem, typename FullTagList,
            typename... ComputeItemArgumentsTags>
  constexpr void add_compute_item_to_box_impl(
      tmpl::list<ComputeItemArgumentsTags...> /*meta*/) noexcept;

  template <typename Tag, typename FullTagList>
  constexpr void add_compute_item_to_box() noexcept;
  // End adding compute items

  // Adding simple items
  template <typename ParentTag>
  constexpr void add_subitem_tags_to_box(tmpl::list<> /*meta*/) noexcept {}

  template <typename ParentTag, typename... Subtags>
  constexpr void add_subitem_tags_to_box(
      tmpl::list<Subtags...> /*meta*/) noexcept;

  template <size_t ArgsIndex, typename Tag, typename... Ts>
  constexpr cpp17::void_type add_item_to_box(
      std::tuple<Ts...>& tupull) noexcept;
  // End adding simple items

  template <typename FullTagList, typename... Ts, typename... AddItemTags,
            typename... AddComputeTags, size_t... Is,
            bool... DependenciesAddedBefore>
  void add_items_to_box(std::tuple<Ts...>& tupull,
                        tmpl::list<AddItemTags...> /*meta*/,
                        std::index_sequence<Is...> /*meta*/,
                        tmpl::list<AddComputeTags...> /*meta*/) noexcept;

  // Merging DataBox's using create_from requires that all instantiations of
  // DataBox be friends with each other.
  template <typename OtherTags>
  friend class DataBox;

  template <typename... OldTags, typename... TagsToCopy>
  constexpr void merge_old_box(
      const db::DataBox<tmpl::list<OldTags...>>& old_box,
      tmpl::list<TagsToCopy...> /*meta*/) noexcept;

  template <typename... OldTags, typename... TagsToCopy>
  constexpr void merge_old_box(db::DataBox<tmpl::list<OldTags...>>&& old_box,
                               tmpl::list<TagsToCopy...> /*meta*/) noexcept;

  // Serialization of DataBox
  // make_deferred_helper is used to expand the parameter pack
  // ComputeItemArgumentsTags
  template <typename Tag, typename... ComputeItemArgumentsTags>
  Deferred<db::item_type<Tag>> make_deferred_helper(
      tmpl::list<ComputeItemArgumentsTags...> /*meta*/) noexcept;

  // clang-tidy: no non-const references
  template <typename... NonSubitemsTags, typename... ComputeTags>
  void pup_impl(PUP::er& p, tmpl::list<NonSubitemsTags...> /*meta*/,  // NOLINT
                tmpl::list<ComputeTags...> /*meta*/) noexcept;
  // End serialization of DataBox

  // Mutating items in the DataBox
  template <typename ParentTag>
  constexpr void mutate_subitem_tags_in_box(tmpl::list<> /*meta*/) noexcept {}

  template <typename ParentTag, typename... Subtags>
  constexpr void mutate_subitem_tags_in_box(
      tmpl::list<Subtags...> /*meta*/) noexcept;

  template <typename ComputeItem,
            Requires<not db::is_compute_item_v<ComputeItem>> = nullptr>
  constexpr void add_reset_compute_item_to_box(tmpl::list<> /*meta*/) noexcept {
  }

  template <typename ComputeItem, typename... ComputeItemArgumentsTags,
            Requires<db::is_compute_item_v<ComputeItem>> = nullptr>
  constexpr void add_reset_compute_item_to_box(
      tmpl::list<ComputeItemArgumentsTags...> /*meta*/) noexcept;

  template <typename... ComputeItemsToReset>
  SPECTRE_ALWAYS_INLINE constexpr void reset_compute_items_after_mutate(
      tmpl::list<ComputeItemsToReset...> /*meta*/) noexcept;

  SPECTRE_ALWAYS_INLINE constexpr void reset_compute_items_after_mutate(
      tmpl::list<> /*meta*/) noexcept {}
  // End mutating items in the DataBox

  using edge_list = tmpl::join<tmpl::transform<
      compute_item_tags,
      DataBox_detail::create_dependency_graph<tmpl::pin<tags_list>, tmpl::_1>>>;

  bool mutate_locked_box_{false};
};

// Adding compute items
namespace DataBox_detail {
template <bool IsMutating>
struct compute_item_function_pointer_type_impl;

template <>
struct compute_item_function_pointer_type_impl<false> {
  // get function pointer type for a non-mutating compute item
  template <typename FullTagList, typename ComputeItem,
            typename... ComputeItemArgumentsTags>
  using f = db::item_type<ComputeItem, FullTagList> (*)(
      std::add_lvalue_reference_t<std::add_const_t<
          db::item_type<ComputeItemArgumentsTags, FullTagList>>>...);
};

template <>
struct compute_item_function_pointer_type_impl<true> {
  // get function pointer type for a mutating compute item
  template <typename FullTagList, typename ComputeItem,
            typename... ComputeItemArgumentsTags>
  using f =
      void (*)(gsl::not_null<
                   std::add_pointer_t<db::item_type<ComputeItem, FullTagList>>>,
               std::add_lvalue_reference_t<std::add_const_t<
                   db::item_type<ComputeItemArgumentsTags, FullTagList>>>...);
};

// Computes the function pointer type of the compute item
template <typename FullTagList, typename ComputeItem,
          typename... ComputeItemArgumentsTags>
using compute_item_function_pointer_type =
    typename compute_item_function_pointer_type_impl<has_return_type_member_v<
        ComputeItem>>::template f<FullTagList, ComputeItem,
                                  ComputeItemArgumentsTags...>;

template <bool IsComputeTag>
struct get_argument_list_impl {
  template <class Tag>
  using f = tmpl::list<>;
};

template <>
struct get_argument_list_impl<true> {
  template <class Tag>
  using f = typename Tag::argument_tags;
};

/// Returns the argument_tags of a compute item. If the Tag is not a compute tag
/// then it returns tmpl::list<>
template <class Tag>
using get_argument_list = typename get_argument_list_impl<
    ::db::is_compute_item_v<Tag>>::template f<Tag>;
}  // namespace DataBox_detail

template <typename... Tags>
template <typename ParentTag, typename... Subtags>
SPECTRE_ALWAYS_INLINE constexpr void
DataBox<tmpl::list<Tags...>>::add_sub_compute_item_tags_to_box(
    tmpl::list<Subtags...> /*meta*/,
    std::false_type /*has_return_type_member*/) noexcept {
  const auto helper = [lazy_function = get_deferred<ParentTag>()](
                          auto tag) noexcept->decltype(auto) {
    return Subitems<tmpl::list<Tags...>, ParentTag>::
        template create_compute_item<decltype(tag)>(lazy_function.get());
  };
  EXPAND_PACK_LEFT_TO_RIGHT(
      (get_deferred<Subtags>() =
           make_deferred_for_subitem<decltype(helper(Subtags{}))>(helper,
                                                                  Subtags{})));
}

template <typename... Tags>
template <typename ParentTag, typename... Subtags>
SPECTRE_ALWAYS_INLINE constexpr void
DataBox<tmpl::list<Tags...>>::add_sub_compute_item_tags_to_box(
    tmpl::list<Subtags...> /*meta*/,
    std::true_type /*has_return_type_member*/) noexcept {
  const auto helper = [lazy_function = get_deferred<ParentTag>()](
      const auto result, auto tag) noexcept {
    Subitems<tmpl::list<Tags...>, ParentTag>::template create_compute_item<
        decltype(tag)>(result, lazy_function.get());
  };
  EXPAND_PACK_LEFT_TO_RIGHT(
      (get_deferred<Subtags>() =
           make_deferred_for_subitem<db::item_type<Subtags>>(helper,
                                                             Subtags{})));
}

namespace DataBox_detail {
// This function exists so that the user can look at the template
// arguments to find out what triggered the static_assert.
template <typename ComputeItem, typename Argument, typename FullTagList>
constexpr cpp17::void_type check_compute_item_argument_exists() noexcept {
  using compute_item_index = tmpl::index_of<FullTagList, ComputeItem>;
  static_assert(
      tmpl::less<tmpl::index_if<FullTagList,
                                std::is_same<tmpl::pin<Argument>, tmpl::_1>,
                                compute_item_index>,
                 compute_item_index>::value,
      "The dependencies of a ComputeItem must be added before the "
      "ComputeItem itself. This is done to ensure no cyclic "
      "dependencies arise.  See the first and second template "
      "arguments of the instantiation of this function for the "
      "compute item and missing dependency.");
  return cpp17::void_type{};
}
}  // namespace DataBox_detail

template <typename... Tags>
template <typename ComputeItem, typename FullTagList,
          typename... ComputeItemArgumentsTags>
SPECTRE_ALWAYS_INLINE constexpr void
DataBox<tmpl::list<Tags...>>::add_compute_item_to_box_impl(
    tmpl::list<ComputeItemArgumentsTags...> /*meta*/) noexcept {
  DEBUG_STATIC_ASSERT(
      tmpl2::flat_all_v<is_tag_v<ComputeItemArgumentsTags>...>,
      "Cannot have non-DataBoxTag arguments to a ComputeItem. Please make "
      "sure all the specified argument_tags in the ComputeItem derive from "
      "db::SimpleTag.");
  DEBUG_STATIC_ASSERT(
      not tmpl2::flat_any_v<
          cpp17::is_same_v<ComputeItemArgumentsTags, ComputeItem>...>,
      "A ComputeItem cannot take its own Tag as an argument.");
  expand_pack(DataBox_detail::check_compute_item_argument_exists<
              ComputeItem, ComputeItemArgumentsTags, FullTagList>()...);

  get_deferred<ComputeItem>() =
      make_deferred<db::item_type<ComputeItem, FullTagList>>(
          DataBox_detail::compute_item_function_pointer_type<
              FullTagList, ComputeItem, ComputeItemArgumentsTags...>{
              ComputeItem::function},
          get_deferred<ComputeItemArgumentsTags>()...);
}

template <typename... Tags>
template <typename Tag, typename FullTagList>
SPECTRE_ALWAYS_INLINE constexpr void
db::DataBox<tmpl::list<Tags...>>::add_compute_item_to_box() noexcept {
  add_compute_item_to_box_impl<Tag, FullTagList>(
      tmpl::transform<typename Tag::argument_tags,
                      tmpl::bind<DataBox_detail::first_matching_tag,
                                 tmpl::pin<tmpl::list<Tags...>>, tmpl::_1>>{});
  add_sub_compute_item_tags_to_box<Tag>(
      typename Subitems<tmpl::list<Tags...>, Tag>::type{},
      typename has_return_type_member<
          Subitems<tmpl::list<Tags...>, Tag>>::type{});
}
// End adding compute items

// Adding simple items
template <typename... Tags>
template <typename ParentTag, typename... Subtags>
SPECTRE_ALWAYS_INLINE constexpr void
db::DataBox<tmpl::list<Tags...>>::add_subitem_tags_to_box(
    tmpl::list<Subtags...> /*meta*/) noexcept {
  const auto helper = [this](auto tag_v) {
    (void)this;  // Compiler bug warns this is unused
    using tag = decltype(tag_v);
    get_deferred<tag>() = Deferred<db::item_type<tag>>(db::item_type<tag>{});
    Subitems<tmpl::list<Tags...>, ParentTag>::template create_item<tag>(
        make_not_null(&get_deferred<ParentTag>().mutate()),
        make_not_null(&get_deferred<tag>().mutate()));
  };

  EXPAND_PACK_LEFT_TO_RIGHT(helper(Subtags{}));
}

template <typename... Tags>
template <size_t ArgsIndex, typename Tag, typename... Ts>
SPECTRE_ALWAYS_INLINE constexpr cpp17::void_type
db::DataBox<tmpl::list<Tags...>>::add_item_to_box(
    std::tuple<Ts...>& tupull) noexcept {
  using ArgType = std::tuple_element_t<ArgsIndex, std::tuple<Ts...>>;
  static_assert(not tt::is_a<Deferred, std::decay_t<ArgType>>::value,
                "Cannot pass a Deferred into the DataBox as an Item. This "
                "functionally can trivially be added, however it is "
                "intentionally omitted because users of DataBox are not "
                "supposed to deal with Deferred.");
  get_deferred<Tag>() = Deferred<item_type<Tag>>(
      std::forward<ArgType>(std::get<ArgsIndex>(tupull)));
  add_subitem_tags_to_box<Tag>(
      typename Subitems<tmpl::list<Tags...>, Tag>::type{});
  return cpp17::void_type{};  // must return in constexpr function
}
// End adding simple items

// Add items or compute items to the TaggedDeferredTuple `data`. If
// `AddItemTags...` is an empty pack then only compute items are added, while if
// `AddComputeTags...` is an empty pack only items are added. Items are
// always added before compute items.
template <typename... Tags>
template <typename FullTagList, typename... Ts, typename... AddItemTags,
          typename... AddComputeTags, size_t... Is,
          bool... DependenciesAddedBefore>
SPECTRE_ALWAYS_INLINE void DataBox<tmpl::list<Tags...>>::add_items_to_box(
    std::tuple<Ts...>& tupull, tmpl::list<AddItemTags...> /*meta*/,
    std::index_sequence<Is...> /*meta*/,
    tmpl::list<AddComputeTags...> /*meta*/) noexcept {
  expand_pack(add_item_to_box<Is, AddItemTags>(tupull)...);
  EXPAND_PACK_LEFT_TO_RIGHT(
      add_compute_item_to_box<AddComputeTags, FullTagList>());
}

namespace DataBox_detail {
// This function (and its unused template argument) exist so that
// users can see what tag has the wrong type when the static_assert
// fails.
template <typename Tag, typename TagType, typename SuppliedType>
constexpr int check_argument_type() noexcept {
  static_assert(cpp17::is_same_v<TagType, SuppliedType>,
                "The type of each Tag must be the same as the type being "
                "passed into the function creating the new DataBox.  See the "
                "function template parameters for the tag, expected type, and "
                "supplied type.");
  return 0;
}
}  // namespace DataBox_detail

/// \cond
template <typename... Tags>
template <typename... TagsInArgsOrder, typename... FullItems,
          typename... ComputeTags, typename... FullComputeItems,
          typename... Args>
constexpr DataBox<tmpl::list<Tags...>>::DataBox(
    tmpl::list<TagsInArgsOrder...> /*meta*/, tmpl::list<FullItems...> /*meta*/,
    tmpl::list<ComputeTags...> /*meta*/,
    tmpl::list<FullComputeItems...> /*meta*/, Args&&... args) noexcept {
  DEBUG_STATIC_ASSERT(
      sizeof...(Tags) == sizeof...(FullItems) + sizeof...(FullComputeItems),
      "Must pass in as many (compute) items as there are Tags.");
  DEBUG_STATIC_ASSERT(sizeof...(TagsInArgsOrder) == sizeof...(Args),
                      "Must pass in as many arguments as AddTags");
  DEBUG_STATIC_ASSERT(
      not tmpl2::flat_any_v<is_databox<std::decay_t<Args>>::value...>,
      "Cannot store a DataBox inside a DataBox.");
#ifdef SPECTRE_DEBUG
  // The check_argument_type call is very expensive compared to the majority of
  // DataBox
  expand_pack(
      DataBox_detail::check_argument_type<TagsInArgsOrder,
                                          typename TagsInArgsOrder::type,
                                          std::decay_t<Args>>()...);
#endif  // SPECTRE_DEBUG

  std::tuple<Args...> args_tuple(std::forward<Args>(args)...);
  add_items_to_box<tmpl::list<FullItems..., FullComputeItems...>>(
      args_tuple, tmpl::list<TagsInArgsOrder...>{},
      std::make_index_sequence<sizeof...(TagsInArgsOrder)>{},
      tmpl::list<ComputeTags...>{});
}

////////////////////////////////////////////////////////////////
// Construct DataBox from an existing one
template <typename... Tags>
template <typename... OldTags, typename... TagsToCopy>
constexpr void DataBox<tmpl::list<Tags...>>::merge_old_box(
    const db::DataBox<tmpl::list<OldTags...>>& old_box,
    tmpl::list<TagsToCopy...> /*meta*/) noexcept {
  EXPAND_PACK_LEFT_TO_RIGHT(get_deferred<TagsToCopy>() =
                                old_box.template get_deferred<TagsToCopy>());
}

template <typename... Tags>
template <typename... OldTags, typename... TagsToCopy>
constexpr void DataBox<tmpl::list<Tags...>>::merge_old_box(
    db::DataBox<tmpl::list<OldTags...>>&& old_box,
    tmpl::list<TagsToCopy...> /*meta*/) noexcept {
  (void)std::initializer_list<char>{
      (void(get_deferred<TagsToCopy>() =
                std::move(old_box.template get_deferred<TagsToCopy>())),
       '0')...};
}

template <typename... Tags>
template <typename Box, typename... KeepTags, typename... AddTags,
          typename... AddComputeTags, typename... Args>
constexpr DataBox<tmpl::list<Tags...>>::DataBox(
    Box&& old_box, tmpl::list<KeepTags...> /*meta*/,
    tmpl::list<AddTags...> /*meta*/, tmpl::list<AddComputeTags...> /*meta*/,
    Args&&... args) noexcept {
  expand_pack(
      DataBox_detail::check_argument_type<AddTags, typename AddTags::type,
                                          std::decay_t<Args>>()...);

  merge_old_box(std::forward<Box>(old_box), tmpl::list<KeepTags...>{});

  // Add in new simple and compute tags
  std::tuple<Args...> args_tuple(std::forward<Args>(args)...);
  add_items_to_box<tmpl::list<Tags...>>(
      args_tuple, tmpl::list<AddTags...>{},
      std::make_index_sequence<sizeof...(AddTags)>{},
      tmpl::list<AddComputeTags...>{});
}

////////////////////////////////////////////////////////////////
// Create a copy of the DataBox with no aliasing items.
template <typename... Tags>
template <typename... SimpleTags>
SPECTRE_ALWAYS_INLINE void DataBox<tmpl::list<Tags...>>::copy_simple_items(
    const DataBox& box, tmpl::list<SimpleTags...> /*meta*/) noexcept {
  EXPAND_PACK_LEFT_TO_RIGHT((get_deferred<SimpleTags>() =
                                 box.get_deferred<SimpleTags>().deep_copy()));
}

template <typename... Tags>
template <typename SimpleItemTags>
DataBox<tmpl::list<Tags...>> DataBox<tmpl::list<Tags...>>::deep_copy() const
    noexcept {
  DataBox new_box{};
  new_box.copy_simple_items(*this, simple_item_tags{});

  std::tuple<> empty_tuple{};
  new_box.add_items_to_box<tmpl::list<Tags...>>(empty_tuple, tmpl::list<>{},
                                                std::make_index_sequence<0>{},
                                                compute_item_tags{});
  return new_box;
}
/// \endcond

////////////////////////////////////////////////////////////////
// Serialization of DataBox

// Function used to expand the parameter pack ComputeItemArgumentsTags
template <typename... Tags>
template <typename Tag, typename... ComputeItemArgumentsTags>
Deferred<db::item_type<Tag>> DataBox<tmpl::list<Tags...>>::make_deferred_helper(
    tmpl::list<ComputeItemArgumentsTags...> /*meta*/) noexcept {
  return make_deferred<db::item_type<Tag>>(
      Tag::function, get_deferred<ComputeItemArgumentsTags>()...);
}

template <typename... Tags>
template <typename... NonSubitemsTags, typename... ComputeTags>
void DataBox<tmpl::list<Tags...>>::pup_impl(
    PUP::er& p, tmpl::list<NonSubitemsTags...> /*meta*/,
    tmpl::list<ComputeTags...> /*meta*/) noexcept {
  const auto pup_simple_item = [&p, this ](auto current_tag) noexcept {
    (void)this;  // Compiler bug warning this capture is not used
    using tag = decltype(current_tag);
    if (p.isUnpacking()) {
      db::item_type<tag> t{};
      p | t;
      get_deferred<tag>() = Deferred<db::item_type<tag>>(std::move(t));
      add_subitem_tags_to_box<tag>(
          typename Subitems<tmpl::list<Tags...>, tag>::type{});
    } else {
      p | get_deferred<tag>().mutate();
    }
  };
  (void)pup_simple_item;  // Silence GCC warning about unused variable
  EXPAND_PACK_LEFT_TO_RIGHT(pup_simple_item(NonSubitemsTags{}));

  const auto pup_compute_item = [&p, this ](auto current_tag) noexcept {
    (void)this;  // Compiler bug warns this isn't used
    using tag = decltype(current_tag);
    if (p.isUnpacking()) {
      get_deferred<tag>() =
          make_deferred_helper<tag>(typename tag::argument_tags{});
    }
    get_deferred<tag>().pack_unpack_lazy_function(p);
    if (p.isUnpacking()) {
      add_sub_compute_item_tags_to_box<tag>(
          typename Subitems<tmpl::list<Tags...>, tag>::type{},
          typename has_return_type_member<
              Subitems<tmpl::list<Tags...>, tag>>::type{});
    }
  };
  (void)pup_compute_item;  // Silence GCC warning about unused variable
  EXPAND_PACK_LEFT_TO_RIGHT(pup_compute_item(ComputeTags{}));
}

////////////////////////////////////////////////////////////////
// Mutating items in the DataBox
// Classes and functions necessary for db::mutate to work
template <typename... Tags>
template <typename ComputeItem, typename... ComputeItemArgumentsTags,
          Requires<db::is_compute_item_v<ComputeItem>>>
SPECTRE_ALWAYS_INLINE constexpr void
DataBox<tmpl::list<Tags...>>::add_reset_compute_item_to_box(
    tmpl::list<ComputeItemArgumentsTags...> /*meta*/) noexcept {
  get_deferred<ComputeItem>().reset();
  mutate_subitem_tags_in_box<ComputeItem>(
      typename Subitems<tmpl::list<Tags...>, ComputeItem>::type{});
}

template <typename... Tags>
template <typename... ComputeItemsToReset>
SPECTRE_ALWAYS_INLINE constexpr void
db::DataBox<tmpl::list<Tags...>>::reset_compute_items_after_mutate(
    tmpl::list<ComputeItemsToReset...> /*meta*/) noexcept {
  EXPAND_PACK_LEFT_TO_RIGHT(add_reset_compute_item_to_box<ComputeItemsToReset>(
      DataBox_detail::get_argument_list<ComputeItemsToReset>{}));

  using next_compute_tags_to_reset =
      tmpl::transform<tmpl::append<tmpl::filter<
                          typename DataBox<tmpl::list<Tags...>>::edge_list,
                          std::is_same<tmpl::pin<ComputeItemsToReset>,
                                       tmpl::get_source<tmpl::_1>>>...>,
                      tmpl::get_destination<tmpl::_1>>;
  reset_compute_items_after_mutate(next_compute_tags_to_reset{});
}

template <typename... Tags>
template <typename ParentTag, typename... Subtags>
SPECTRE_ALWAYS_INLINE constexpr void
db::DataBox<tmpl::list<Tags...>>::mutate_subitem_tags_in_box(
    tmpl::list<Subtags...> /*meta*/) noexcept {
  const auto helper = [this](auto tag_v) {
    (void)this;  // Compiler bug warns about unused this capture
    using tag = decltype(tag_v);
    if (is_compute_item_v<ParentTag>) {
      get_deferred<tag>().reset();
    } else {
      Subitems<tmpl::list<Tags...>, ParentTag>::template create_item<tag>(
          make_not_null(&get_deferred<ParentTag>().mutate()),
          make_not_null(&get_deferred<tag>().mutate()));
    }
  };

  EXPAND_PACK_LEFT_TO_RIGHT(helper(Subtags{}));
}

/*!
 * \ingroup DataBoxGroup
 * \brief Allows changing the state of one or more non-computed elements in
 * the DataBox
 *
 * `mutate()`'s first argument is the DataBox from which to retrieve the tags
 * `MutateTags`. The objects corresponding to the `MutateTags` are then passed
 * to `invokable`, which is a lambda or a function object taking as many
 * arguments as there are `MutateTags` and with the arguments being of types
 * `gsl::not_null<db::item_type<MutateTags>*>...`. Inside the `invokable` no
 * items can be retrieved from the DataBox `box`. This is to avoid confusing
 * subtleties with order of evaluation of compute items, as well as dangling
 * references. If an `invokable` needs read access to items in `box` they should
 * be passed as additional arguments to `mutate`. Capturing them by reference in
 * a lambda does not work because of a bug in GCC 6.3 and earlier. For a
 * function object the read-only items can also be stored as const references
 * inside the object by passing `db::get<TAG>(t)` to the constructor.
 *
 * \example
 * \snippet Test_DataBox.cpp databox_mutate_example
 */
template <typename... MutateTags, typename TagList, typename Invokable,
          typename... Args>
void mutate(const gsl::not_null<DataBox<TagList>*> box, Invokable&& invokable,
            Args&&... args) noexcept {
  static_assert(
      tmpl2::flat_all_v<
          DataBox_detail::has_unique_matching_tag_v<TagList, MutateTags>...>,
      "One of the tags being mutated could not be found in the DataBox or "
      "is a base tag identifying more than one tag.");
  static_assert(
      not tmpl2::flat_any_v<db::is_compute_item_v<
          DataBox_detail::first_matching_tag<TagList, MutateTags>>...>,
      "Cannot mutate a compute item");
  if (UNLIKELY(box->mutate_locked_box_)) {
    ERROR(
        "Unable to mutate a DataBox that is already being mutated. This "
        "error occurs when mutating a DataBox from inside the invokable "
        "passed to the mutate function.");
  }
  box->mutate_locked_box_ = true;
  invokable(
      make_not_null(
          &box->template get_deferred<
                  DataBox_detail::first_matching_tag<TagList, MutateTags>>()
               .mutate())...,
      std::forward<Args>(args)...);
  using mutate_tags_list =
      tmpl::list<DataBox_detail::first_matching_tag<TagList, MutateTags>...>;
  // For all the tags in the DataBox, check if one of their subtags is
  // being mutated and if so add the parent to the list of tags
  // being mutated. Then, remove any tags that would be passed
  // multiple times.
  using extra_mutated_tags = tmpl::list_difference<
      tmpl::filter<
          TagList,
          tmpl::bind<
              tmpl::found, Subitems<tmpl::pin<TagList>, tmpl::_1>,
              tmpl::pin<tmpl::bind<tmpl::list_contains,
                                   tmpl::pin<mutate_tags_list>, tmpl::_1>>>>,
      mutate_tags_list>;
  // Extract the subtags inside the MutateTags and reset compute items
  // depending on those too.
  using full_mutated_items = tmpl::append<
      DataBox_detail::expand_subitems_from_list<TagList, mutate_tags_list>,
      extra_mutated_tags>;

  using first_compute_items_to_reset =
      tmpl::transform<tmpl::filter<typename DataBox<TagList>::edge_list,
                                   tmpl::bind<tmpl::list_contains,
                                              tmpl::pin<full_mutated_items>,
                                              tmpl::get_source<tmpl::_1>>>,
                      tmpl::get_destination<tmpl::_1>>;

  EXPAND_PACK_LEFT_TO_RIGHT(
      box->template mutate_subitem_tags_in_box<MutateTags>(
          typename Subitems<TagList, MutateTags>::type{}));
  box->template reset_compute_items_after_mutate(
      first_compute_items_to_reset{});

  box->mutate_locked_box_ = false;
}

////////////////////////////////////////////////////////////////
// Retrieving items from the DataBox

/// \cond
template <typename... Tags>
template <typename Tag, Requires<not cpp17::is_same_v<Tag, ::Tags::DataBox>>>
SPECTRE_ALWAYS_INLINE auto DataBox<tmpl::list<Tags...>>::get() const noexcept
    -> const item_type<Tag, tags_list>& {
  DEBUG_STATIC_ASSERT(
      not DataBox_detail::has_no_matching_tag_v<tags_list, Tag>,
      "Found no tags in the DataBox that match the tag being retrieved.");
  DEBUG_STATIC_ASSERT(
      DataBox_detail::has_unique_matching_tag_v<tags_list, Tag>,
      "Found more than one tag in the DataBox that matches the tag "
      "being retrieved. This happens because more than one tag with the same "
      "base (class) tag was added to the DataBox.");
  using derived_tag = DataBox_detail::first_matching_tag<tags_list, Tag>;
  if (UNLIKELY(mutate_locked_box_)) {
    ERROR("Unable to retrieve a (compute) item '"
          << derived_tag::name()
          << "' from the DataBox from within a "
             "call to mutate. You must pass these either through the capture "
             "list of the lambda or the constructor of a class, this "
             "restriction exists to avoid complexity.");
  }
  return get_deferred<derived_tag>().get();
}

template <typename... Tags>
template <typename Tag, Requires<cpp17::is_same_v<Tag, ::Tags::DataBox>>>
SPECTRE_ALWAYS_INLINE auto DataBox<tmpl::list<Tags...>>::get() const noexcept
    -> const DataBox<tags_list>& {
  if (UNLIKELY(mutate_locked_box_)) {
    ERROR(
        "Unable to retrieve a (compute) item 'DataBox' from the DataBox from "
        "within a call to mutate. You must pass these either through the "
        "capture list of the lambda or the constructor of a class, this "
        "restriction exists to avoid complexity.");
  }
  return *this;
}
/// \endcond

/*!
 * \ingroup DataBoxGroup
 * \brief Retrieve the item with tag `Tag` from the DataBox
 * \requires Type `Tag` is one of the Tags corresponding to an object stored in
 * the DataBox
 *
 * \return The object corresponding to the tag `Tag`
 */
template <typename Tag, typename TagList>
SPECTRE_ALWAYS_INLINE const auto& get(const DataBox<TagList>& box) noexcept {
  return box.template get<Tag>();
}

/*!
 * \ingroup DataBoxGroup
 * \brief List of Tags to remove from the DataBox
 */
template <typename... Tags>
using RemoveTags = tmpl::flatten<tmpl::list<Tags...>>;

/*!
 * \ingroup DataBoxGroup
 * \brief List of Tags to add to the DataBox
 */
template <typename... Tags>
using AddSimpleTags = tmpl::flatten<tmpl::list<Tags...>>;

/*!
 * \ingroup DataBoxGroup
 * \brief List of Tags to mutate in the DataBox
 */
template <typename... Tags>
using MutateTags = tmpl::flatten<tmpl::list<Tags...>>;

/*!
 * \ingroup DataBoxGroup
 * \brief List of Tags to get from the DataBox to be used as arguments
 */
template <typename... Tags>
using ArgumentTags = tmpl::flatten<tmpl::list<Tags...>>;

/*!
 * \ingroup DataBoxGroup
 * \brief List of Compute Item Tags to add to the DataBox
 */
template <typename... Tags>
using AddComputeTags = tmpl::flatten<tmpl::list<Tags...>>;

/*!
 * \ingroup DataBoxGroup
 * \brief Create a new DataBox
 *
 * \details
 * Creates a new DataBox holding types Tags::type filled with the arguments
 * passed to the function. Compute items must be added so that the dependencies
 * of a compute item are added before the compute item. For example, say you
 * have compute items `A` and `B` where `B` depends on `A`, then you must
 * add them using `db::AddComputeTags<A, B>`.
 *
 * \example
 * \snippet Test_DataBox.cpp create_databox
 *
 * \see create_from
 *
 * \tparam AddSimpleTags the tags of the args being added
 * \tparam AddComputeTags list of \ref ComputeTag "compute item tags"
 * to add to the DataBox
 *  \param args the data to be added to the DataBox
 */
template <typename AddSimpleTags, typename AddComputeTags = tmpl::list<>,
          typename... Args>
SPECTRE_ALWAYS_INLINE constexpr auto create(Args&&... args) {
  static_assert(tt::is_a_v<tmpl::list, AddComputeTags>,
                "AddComputeTags must be a tmpl::list");
  static_assert(tt::is_a_v<tmpl::list, AddSimpleTags>,
                "AddSimpleTags must be a tmpl::list");
  static_assert(
      not tmpl::any<AddSimpleTags, is_compute_item<tmpl::_1>>::value,
      "Cannot add any ComputeTags in the AddSimpleTags list, must use the "
      "AddComputeTags list.");
  static_assert(
      tmpl::all<AddComputeTags, is_compute_item<tmpl::_1>>::value,
      "Cannot add any SimpleTags in the AddComputeTags list, must use the "
      "AddSimpleTags list.");

  using tag_list =
      DataBox_detail::expand_subitems<AddSimpleTags, AddComputeTags, true>;
  using full_items =
      DataBox_detail::expand_subitems<AddSimpleTags, tmpl::list<>, true>;
  using full_compute_items =
      DataBox_detail::expand_subitems_from_list<tag_list, AddComputeTags>;

  return DataBox<tmpl::append<full_items, full_compute_items>>(
      AddSimpleTags{}, full_items{}, AddComputeTags{}, full_compute_items{},
      std::forward<Args>(args)...);
}

namespace DataBox_detail {
template <typename RemoveTags, typename AddTags, typename AddComputeTags,
          typename Box, typename... Args>
SPECTRE_ALWAYS_INLINE constexpr auto create_from(Box&& box,
                                                 Args&&... args) noexcept {
  static_assert(tmpl::size<AddTags>::value == sizeof...(Args),
                "Must pass in as many arguments as AddTags to db::create_from");

  // 1. Full list of old tags, and the derived tags list of the RemoveTags
  using old_box_tags = typename std::decay_t<Box>::tags_list;
  static_assert(
      tmpl::all<RemoveTags, DataBox_detail::has_unique_matching_tag<
                                tmpl::pin<old_box_tags>, tmpl::_1>>::value,
      "One of the tags being removed could not be found in the DataBox or "
      "is a base tag identifying more than one tag.");
  using remove_tags =
      tmpl::transform<RemoveTags,
                      tmpl::bind<DataBox_detail::first_matching_tag,
                                 tmpl::pin<old_box_tags>, tmpl::_1>>;

  // 2. Expand simple remove tags and compute remove tags
  using compute_tags_to_remove =
      tmpl::filter<remove_tags, db::is_compute_item<tmpl::_1>>;
  using compute_tags_to_remove_with_subitems =
      DataBox_detail::expand_subitems_from_list<old_box_tags,
                                                compute_tags_to_remove>;
  using simple_tags_to_remove =
      tmpl::list_difference<remove_tags, compute_tags_to_remove>;
  using simple_tags_to_remove_with_subitems =
      DataBox_detail::expand_subitems<tmpl::list<>, simple_tags_to_remove,
                                      false>;

  // 3. Expand AddTags (these are just the simple tags)
  using simple_tags_to_add_with_subitems =
      DataBox_detail::expand_subitems<AddTags, tmpl::list<>, true>;

  // 4. Create new list of tags by removing all the remove tags, and adding all
  // the AddTags, including subitems
  using simple_tags_to_keep =
      tmpl::list_difference<typename std::decay_t<Box>::simple_item_tags,
                            simple_tags_to_remove_with_subitems>;
  using new_simple_tags =
      tmpl::append<simple_tags_to_keep, simple_tags_to_add_with_subitems>;

  // 5. Create the list of compute items with the RemoveTags removed
  using compute_tags_to_keep = tmpl::list_difference<
      typename std::decay_t<Box>::compute_with_subitems_tags,
      compute_tags_to_remove_with_subitems>;

  // 6. List of the old tags that are being kept
  using old_tags_to_keep =
      tmpl::append<simple_tags_to_keep, compute_tags_to_keep>;

  // 7. List of the new tags, we only need to expand the AddComputeTags now
  using new_tag_list = DataBox_detail::expand_subitems<
      tmpl::append<new_simple_tags, compute_tags_to_keep>, AddComputeTags,
      false>;

  DEBUG_STATIC_ASSERT(
      tmpl::size<tmpl::list_difference<AddTags, RemoveTags>>::value ==
          tmpl::size<AddTags>::value,
      "Use db::mutate to mutate simple items, do not remove and add them with "
      "db::create_from.");

#ifdef SPECTRE_DEBUG
  // Check that we're not removing a subitem itself, should remove the parent.
  using compute_subitems_tags =
      tmpl::filter<typename std::decay_t<Box>::compute_item_tags,
                   tmpl::bind<DataBox_detail::has_subitems,
                              tmpl::pin<old_box_tags>, tmpl::_1>>;

  using compute_only_expand_subitems_tags = tmpl::flatten<tmpl::transform<
      compute_subitems_tags, db::Subitems<tmpl::pin<old_box_tags>, tmpl::_1>>>;
  using all_only_subitems_tags = tmpl::append<
      typename std::decay_t<Box>::simple_only_expanded_subitems_tags,
      compute_only_expand_subitems_tags>;
  using non_expand_subitems_remove_tags =
      tmpl::list_difference<RemoveTags, all_only_subitems_tags>;
  static_assert(tmpl::size<non_expand_subitems_remove_tags>::value ==
                    tmpl::size<RemoveTags>::value,
                "You are not allowed to remove part of a Subitem from the "
                "DataBox using db::create_from.");
#endif  // ifdef SPECTRE_DEBUG

  return DataBox<new_tag_list>(std::forward<Box>(box), old_tags_to_keep{},
                               AddTags{}, AddComputeTags{},
                               std::forward<Args>(args)...);
}
}  // namespace DataBox_detail

/*!
 * \ingroup DataBoxGroup
 * \brief Create a new DataBox from an existing one adding or removing items
 * and compute items
 *
 * When passed an lvalue this function will return a const DataBox
 * whose members cannot be modified.  When passed a (mutable) rvalue
 * this function will return a mutable DataBox.
 *
 * Note that in the const lvalue case the output DataBox shares all
 * items that were not removed with the input DataBox. This means if an item is
 * mutated in the input DataBox it is also mutated in the output DataBox.
 * Similarly, if a compute item is evaluated in either the returned DataBox or
 * the input DataBox it is evaluated in both (at the cost of only evaluating it
 * once).
 *
 * \example
 * Removing an item or compute item is done using:
 * \snippet Test_DataBox.cpp create_from_remove
 * Adding an item is done using:
 * \snippet Test_DataBox.cpp create_from_add_item
 * Adding a compute item is done using:
 * \snippet Test_DataBox.cpp create_from_add_compute_item
 *
 * \see create DataBox
 *
 * \tparam RemoveTags typelist of Tags to remove
 * \tparam AddTags typelist of Tags corresponding to the arguments to be
 * added
 * \tparam AddComputeTags list of \ref ComputeTag "compute item tags"
 * to add to the DataBox
 * \param box the DataBox the new box should be based off
 * \param args the values for the items to add to the DataBox
 * \return DataBox like `box` but altered by RemoveTags and AddTags
 *@{
 */
template <typename RemoveTags, typename AddTags = tmpl::list<>,
          typename AddComputeTags = tmpl::list<>, typename TagsList,
          typename... Args>
SPECTRE_ALWAYS_INLINE constexpr auto create_from(db::DataBox<TagsList>&& box,
                                                 Args&&... args) noexcept {
  return DataBox_detail::create_from<RemoveTags, AddTags, AddComputeTags>(
      std::move(box), std::forward<Args>(args)...);
}

/// \cond HIDDEN_SYMBOLS
// Clang warns that the const qualifier on the return type has no
// effect.  It does have an effect.
#ifdef __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#endif
template <typename RemoveTags, typename AddTags = tmpl::list<>,
          typename AddComputeTags = tmpl::list<>, typename TagsList,
          typename... Args>
SPECTRE_ALWAYS_INLINE constexpr const auto create_from(
    const db::DataBox<TagsList>& box, Args&&... args) noexcept {
  return DataBox_detail::create_from<RemoveTags, AddTags, AddComputeTags>(
      box, std::forward<Args>(args)...);
}
#ifdef __clang__
#pragma GCC diagnostic pop
#endif
/// \endcond
/**@}*/

/*!
 * \ingroup DataBoxGroup
 * \brief Create a non-aliasing copy of the DataBox. That is, the new DataBox
 * will not share items with the old one.
 *
 * \warning Currently all compute items will be reset in the new DataBox because
 * copying of DataBoxes shouldn't be done in general. This does not lead to
 * incorrect behavior, but is less efficient.
 *
 * \see db::create_from
 */
template <typename TagsList>
SPECTRE_ALWAYS_INLINE constexpr DataBox<TagsList> create_copy(
    const DataBox<TagsList>& box) noexcept {
  return box.template deep_copy<typename DataBox<TagsList>::simple_item_tags>();
}

namespace DataBox_detail {
template <typename Type, typename... Tags, typename... TagsInBox>
const Type& get_item_from_box(const DataBox<tmpl::list<TagsInBox...>>& box,
                              const std::string& tag_name,
                              tmpl::list<Tags...> /*meta*/) {
  DEBUG_STATIC_ASSERT(
      sizeof...(Tags) != 0,
      "No items with the requested type were found in the DataBox");
  const Type* result = nullptr;
  const auto helper = [&box, &tag_name, &result ](auto current_tag) noexcept {
    using tag = decltype(current_tag);
    if (get_tag_name<tag>() == tag_name) {
      result = &::db::get<tag>(box);
    }
  };
  EXPAND_PACK_LEFT_TO_RIGHT(helper(Tags{}));
  if (result == nullptr) {
    std::string tags_in_box;
    const auto print_helper = [&tags_in_box](auto tag) noexcept {
      tags_in_box += "  " + decltype(tag)::name() + "\n";
    };
    EXPAND_PACK_LEFT_TO_RIGHT(print_helper(Tags{}));
    ERROR("Could not find the tag named \""
          << tag_name << "\" in the DataBox. Available tags are:\n"
          << tags_in_box);
  }
  return *result;
}  // namespace db
}  // namespace DataBox_detail

/*!
 * \ingroup DataBoxGroup
 * \brief Retrieve an item from the DataBox that has a tag with label `tag_name`
 * and type `Type`
 *
 * \details
 * The type that the tag represents must be of the type `Type`, and the tag must
 * have the label `tag_name`. The function iterates over all tags in the DataBox
 * `box` that have the type `Type` searching linearly for one whose `label`
 * matches `tag_name`.
 *
 * \example
 * \snippet Test_DataBox.cpp get_item_from_box
 *
 * \tparam Type the type of the tag with the `label` `tag_name`
 * \param box the DataBox through which to search
 * \param tag_name the `label` of the tag to retrieve
 */
template <typename Type, typename TagList>
constexpr const Type& get_item_from_box(const DataBox<TagList>& box,
                                        const std::string& tag_name) noexcept {
  using tags = tmpl::filter<
      TagList, std::is_same<tmpl::bind<item_type, tmpl::_1>, tmpl::pin<Type>>>;
  return DataBox_detail::get_item_from_box<Type>(box, tag_name, tags{});
}

namespace DataBox_detail {
CREATE_IS_CALLABLE(apply)

template <typename TagsList>
struct Apply;

template <typename... Tags>
struct Apply<tmpl::list<Tags...>> {
  template <typename F, typename BoxTags, typename... Args,
            Requires<is_apply_callable_v<
                F, const std::add_lvalue_reference_t<db::item_type<Tags>>...,
                Args...>> = nullptr>
  static constexpr auto apply(F&& /*f*/, const DataBox<BoxTags>& box,
                              Args&&... args) {
    return F::apply(::db::get<Tags>(box)..., std::forward<Args>(args)...);
  }

  template <typename F, typename BoxTags, typename... Args,
            Requires<not is_apply_callable_v<
                F, const std::add_lvalue_reference_t<db::item_type<Tags>>...,
                Args...>> = nullptr>
  static constexpr auto apply(F&& f, const DataBox<BoxTags>& box,
                              Args&&... args) {
    static_assert(
        tt::is_callable_v<
            std::remove_pointer_t<F>,
            tmpl::conditional_t<cpp17::is_same_v<Tags, ::Tags::DataBox>,
                                const DataBox<BoxTags>&, item_type<Tags>>...,
            Args...>,
        "Cannot call the function f with the list of tags and "
        "arguments specified. Check that the Tags::type and the "
        "types of the Args match the function f.");
    return std::forward<F>(f)(::db::get<Tags>(box)...,
                              std::forward<Args>(args)...);
  }
};
}  // namespace DataBox_detail

/*!
 * \ingroup DataBoxGroup
 * \brief Apply the function `f` with argument Tags `TagsList` from
 * DataBox `box`
 *
 * \details
 * `f` must either be invokable with the arguments of type
 * `db::item_type<TagsList>..., Args...` where the first pack expansion
 * is over the elements in the type list `TagsList`, or have a static
 * `apply` function that is callable with the same types.
 *
 * \usage
 * Given a function `func` that takes arguments of types
 * `T1`, `T2`, `A1` and `A2`. Let the Tags for the quantities of types `T1` and
 * `T2` in the DataBox `box` be `Tag1` and `Tag2`, and objects `a1` of type
 * `A1` and `a2` of type `A2`, then
 * \code
 * auto result = db::apply<tmpl::list<Tag1, Tag2>>(func, box, a1, a2);
 * \endcode
 * \return `decltype(func(db::get<Tag1>(box), db::get<Tag2>(box), a1, a2))`
 *
 * \semantics
 * For tags `Tags...` in a DataBox `box`, and a function `func` that takes
 * `sizeof...(Tags)` arguments of types `db::item_type<Tags>...`,  and
 * `sizeof...(Args)` arguments of types `Args...`,
 * \code
 * result = func(box, db::get<Tags>(box)..., args...);
 * \endcode
 *
 * \example
 * \snippet Test_DataBox.cpp apply_example
 * Using a struct with an `apply` method:
 * \snippet Test_DataBox.cpp apply_struct_example
 *
 * \see DataBox
 * \tparam TagsList typelist of Tags in the order that they are to be passed
 * to `f`
 * \param f the function to apply
 * \param box the DataBox out of which to retrieve the Tags and to pass to `f`
 * \param args the arguments to pass to the function that are not in the
 * DataBox, `box`
 */
template <typename TagsList, typename F, typename BoxTags, typename... Args>
inline constexpr auto apply(F&& f, const DataBox<BoxTags>& box,
                            Args&&... args) {
  return DataBox_detail::Apply<TagsList>::apply(std::forward<F>(f), box,
                                                std::forward<Args>(args)...);
}

namespace DataBox_detail {
template <typename... ReturnTags, typename... ArgumentTags, typename F,
          typename BoxTags, typename... Args,
          Requires<is_apply_callable_v<
              F, const gsl::not_null<db::item_type<ReturnTags>*>...,
              const std::add_lvalue_reference_t<
                  db::item_type<ArgumentTags, BoxTags>>...,
              Args...>> = nullptr>
inline constexpr auto mutate_apply(
    F&& /*f*/, const gsl::not_null<db::DataBox<BoxTags>*> box,
    tmpl::list<ReturnTags...> /*meta*/, tmpl::list<ArgumentTags...> /*meta*/,
    Args&&... args) noexcept {
  static_assert(
      not tmpl2::flat_any_v<
          cpp17::is_same_v<ArgumentTags, Tags::DataBox>...> and
          not tmpl2::flat_any_v<cpp17::is_same_v<ReturnTags, Tags::DataBox>...>,
      "Cannot pass a DataBox to mutate_apply since the db::get won't work "
      "inside mutate_apply.");
  ::db::mutate<ReturnTags...>(
      box,
      [](const gsl::not_null<db::item_type<ReturnTags>*>... mutated_items,
         const db::item_type<ArgumentTags, BoxTags>&... args_items,
         decltype(std::forward<Args>(args))... l_args) noexcept {
        return std::decay_t<F>::apply(mutated_items..., args_items...,
                                      std::forward<Args>(l_args)...);
      },
      db::get<ArgumentTags>(*box)..., std::forward<Args>(args)...);
}

template <typename... ReturnTags, typename... ArgumentTags, typename F,
          typename BoxTags, typename... Args,
          Requires<::tt::is_callable_v<
              F, const gsl::not_null<db::item_type<ReturnTags>*>...,
              const std::add_lvalue_reference_t<
                  db::item_type<ArgumentTags, BoxTags>>...,
              Args...>> = nullptr>
inline constexpr auto mutate_apply(
    F&& f, const gsl::not_null<db::DataBox<BoxTags>*> box,
    tmpl::list<ReturnTags...> /*meta*/, tmpl::list<ArgumentTags...> /*meta*/,
    Args&&... args) noexcept {
  static_assert(
      not tmpl2::flat_any_v<
          cpp17::is_same_v<ArgumentTags, Tags::DataBox>...> and
          not tmpl2::flat_any_v<cpp17::is_same_v<ReturnTags, Tags::DataBox>...>,
      "Cannot pass a DataBox to mutate_apply since the db::get won't work "
      "inside mutate_apply.");
  ::db::mutate<ReturnTags...>(
      box,
      [&f](const gsl::not_null<db::item_type<ReturnTags>*>... mutated_items,
           const db::item_type<ArgumentTags, BoxTags>&... args_items,
           decltype(std::forward<Args>(args))... l_args) noexcept {
        return f(mutated_items..., args_items...,
                 std::forward<Args>(l_args)...);
      },
      db::get<ArgumentTags>(*box)..., std::forward<Args>(args)...);
}

template <typename Func, typename... Args>
constexpr void error_mutate_apply_not_callable() noexcept {
  static_assert(cpp17::is_same_v<Func, void>,
                "The function is not callable with the expected arguments.  "
                "See the first template parameter for the function type and "
                "the remaining arguments for the parameters that cannot be "
                "passed.");
}

template <
    typename... ReturnTags, typename... ArgumentTags, typename F,
    typename BoxTags, typename... Args,
    Requires<not(is_apply_callable_v<
                     F, const gsl::not_null<db::item_type<ReturnTags>*>...,
                     const std::add_lvalue_reference_t<
                         db::item_type<ArgumentTags, BoxTags>>...,
                     Args...> or
                 ::tt::is_callable_v<
                     F, const gsl::not_null<db::item_type<ReturnTags>*>...,
                     const std::add_lvalue_reference_t<
                         db::item_type<ArgumentTags, BoxTags>>...,
                     Args...>)> = nullptr>
inline constexpr auto mutate_apply(
    F /*f*/, const gsl::not_null<db::DataBox<BoxTags>*> /*box*/,
    tmpl::list<ReturnTags...> /*meta*/, tmpl::list<ArgumentTags...> /*meta*/,
    Args&&... /*args*/) noexcept {
  error_mutate_apply_not_callable<
      F, gsl::not_null<db::item_type<ReturnTags>*>...,
      const db::item_type<ArgumentTags, BoxTags>&..., Args&&...>();
}

template <typename Tag, typename BoxTags>
constexpr int check_mutate_apply_mutate_tag() noexcept {
  static_assert(tmpl::list_contains_v<BoxTags, Tag>,
                "A tag to mutate is not in the DataBox.  See the first "
                "template argument for the missing tag, and the second for the "
                "available tags.");
  return 0;
}

template <typename BoxTags, typename... MutateTags>
constexpr bool check_mutate_apply_mutate_tags(
    BoxTags /*meta*/, tmpl::list<MutateTags...> /*meta*/) noexcept {
  expand_pack(check_mutate_apply_mutate_tag<MutateTags, BoxTags>()...);
  return true;
}

template <typename Tag, typename BoxTags>
constexpr int check_mutate_apply_apply_tag() noexcept {
  // This static assert is triggered for the mutate_apply on line
  // 86 of ComputeNonConservativeBoundaryFluxes, with the tag Interface<Dirs,
  // Normalized...>, which is the base tag of InterfaceComputeItem<Dirs,
  // Normalized...> and so can be retrieved from the DataBox, but still triggers
  // this assert.

  //  static_assert(tmpl::list_contains_v<BoxTags, Tag>,
  //                "A tag to apply with is not in the DataBox.  See the first "
  //                "template argument for the missing tag, and the second for
  //                the " "available tags.");
  return 0;
}

template <typename BoxTags, typename... ApplyTags>
constexpr bool check_mutate_apply_argument_tags(
    BoxTags /*meta*/, tmpl::list<ApplyTags...> /*meta*/) noexcept {
  expand_pack(check_mutate_apply_apply_tag<ApplyTags, BoxTags>()...);
  return true;
}

template <typename F, typename = cpp17::void_t<>>
struct has_return_tags_and_argument_tags : std::false_type {};

template <typename F>
struct has_return_tags_and_argument_tags<
    F, cpp17::void_t<typename F::return_tags, typename F::argument_tags>>
    : std::true_type {};

template <typename F>
constexpr bool has_return_tags_and_argument_tags_v =
    has_return_tags_and_argument_tags<F>::value;
}  // namespace DataBox_detail

// @{
/*!
 * \ingroup DataBoxGroup
 * \brief Apply the invokable `f` mutating items `MutateTags` and taking as
 * additional arguments `ArgumentTags` and `args`.
 *
 * \details
 * `f` must either be invokable with the arguments of type
 * `gsl::not_null<db::item_type<MutateTags>*>...,
 * db::item_type<ArgumentTags>..., Args...`
 * where the first two pack expansions are over the elements in the typelists
 * `MutateTags` and `ArgumentTags`, or have a static `apply` function that is
 * callable with the same types. If the type of `f` specifies `return_tags` and
 * `argument_tags` typelists, these are used for the `MutateTags` and
 * `ArgumentTags`, respectively.
 *
 * \example
 * An example of using `mutate_apply` with a lambda:
 * \snippet Test_DataBox.cpp mutate_apply_lambda_example
 *
 * An example of a class with a static `apply` function
 * \snippet Test_DataBox.cpp mutate_apply_struct_definition_example
 * and how to use `mutate_apply` with the above class
 * \snippet Test_DataBox.cpp mutate_apply_struct_example_stateful
 * Note that the class exposes `return_tags` and `argument_tags` typelists, so
 * we don't specify the template parameters explicitly.
 * If the class `F` has no state, like in this example, you can also use the
 * stateless overload of `mutate_apply`:
 * \snippet Test_DataBox.cpp mutate_apply_struct_example_stateless
 *
 * \tparam MutateTags typelist of Tags to mutate
 * \tparam ArgumentTags typelist of additional items to retrieve from the
 * DataBox
 * \param f the function to apply
 * \param box the DataBox out of which to retrieve the Tags and to pass to `f`
 * \param args the arguments to pass to the function that are not in the
 * DataBox, `box`
 */
template <typename MutateTags, typename ArgumentTags, typename F,
          typename BoxTags, typename... Args,
          Requires<not DataBox_detail::has_return_tags_and_argument_tags_v<
              std::decay_t<F>>> = nullptr>
inline constexpr auto mutate_apply(
    F&& f, const gsl::not_null<DataBox<BoxTags>*> box,
    Args&&... args) noexcept(DataBox_detail::
                                 check_mutate_apply_mutate_tags(
                                     BoxTags{}, MutateTags{}) and
                             DataBox_detail::check_mutate_apply_argument_tags(
                                 BoxTags{}, ArgumentTags{}) and
                             noexcept(DataBox_detail::mutate_apply(
                                 f, box, MutateTags{}, ArgumentTags{},
                                 std::forward<Args>(args)...))) {
  // These checks are duplicated in the noexcept specification above
  // because the noexcept(DataBox_detail::mutate_apply(...)) can cause
  // a compilation error before the checks in the function body are
  // performed.
  DataBox_detail::check_mutate_apply_mutate_tags(BoxTags{}, MutateTags{});
  DataBox_detail::check_mutate_apply_argument_tags(BoxTags{}, ArgumentTags{});
  return DataBox_detail::mutate_apply(std::forward<F>(f), box, MutateTags{},
                                      ArgumentTags{},
                                      std::forward<Args>(args)...);
}

template <typename F, typename BoxTags, typename... Args,
          Requires<DataBox_detail::has_return_tags_and_argument_tags_v<
              std::decay_t<F>>> = nullptr,
          typename MutateTags = typename std::decay_t<F>::return_tags,
          typename ArgumentTags = typename std::decay_t<F>::argument_tags>
inline constexpr auto mutate_apply(
    F&& f, const gsl::not_null<DataBox<BoxTags>*> box,
    Args&&... args) noexcept(DataBox_detail::
                                 check_mutate_apply_mutate_tags(
                                     BoxTags{}, MutateTags{}) and
                             DataBox_detail::check_mutate_apply_argument_tags(
                                 BoxTags{}, ArgumentTags{}) and
                             noexcept(DataBox_detail::mutate_apply(
                                 f, box, MutateTags{}, ArgumentTags{},
                                 std::forward<Args>(args)...))) {
  // These checks are duplicated in the noexcept specification above
  // because the noexcept(DataBox_detail::mutate_apply(...)) can cause
  // a compilation error before the checks in the function body are
  // performed.
  DataBox_detail::check_mutate_apply_mutate_tags(BoxTags{}, MutateTags{});
  DataBox_detail::check_mutate_apply_argument_tags(BoxTags{}, ArgumentTags{});
  return DataBox_detail::mutate_apply(std::forward<F>(f), box, MutateTags{},
                                      ArgumentTags{},
                                      std::forward<Args>(args)...);
}
// @}

/*!
 * \ingroup DataBoxGroup
 * \brief Apply the stateless function `F::apply` mutating the `F::return_tags`
 * and taking as additional arguments the `F::argument_tags` and `args`.
 *
 * \details
 * `F` must have `tmpl::list` type aliases `return_tags` and `argument_tags`, as
 * well as a static `apply` function. The `apply` function must take the types
 * of the `return_tags` as `gsl::not_null` pointers, followed by the types of
 * the `argument_tags` as constant references. It can also take the `Args` as
 * additional arguments.
 *
 * \example
 * \snippet Test_DataBox.cpp mutate_apply_struct_definition_example
 * This is how to use `mutate_apply` with the above class:
 * \snippet Test_DataBox.cpp mutate_apply_struct_example_stateless
 *
 * \tparam F The function to apply
 * \param box The DataBox out of which to retrieve the Tags to pass to `F`
 * \param args The arguments to pass to the function that are not in the
 * DataBox, `box`
 */
template <typename F, typename BoxTags, typename... Args>
inline constexpr auto mutate_apply(const gsl::not_null<DataBox<BoxTags>*> box,
                                   Args&&... args) noexcept {
  static_assert(
      DataBox_detail::has_return_tags_and_argument_tags_v<F>,
      "The stateless mutator does not specify both 'argument_tags' and "
      "'return_tags'. Did you forget to add these tag lists to the mutator "
      "class? The class is listed as the first template parameter below.");
  mutate_apply(F{}, box, std::forward<Args>(args)...);
}

/*!
 * \ingroup DataBoxGroup
 * \brief Get all the Tags that are compute items from the `TagList`
 */
template <class TagList>
using get_compute_items = tmpl::filter<TagList, db::is_compute_item<tmpl::_1>>;

/*!
 * \ingroup DataBoxGroup
 * \brief Get all the Tags that are items from the `TagList`
 */
template <class TagList>
using get_items =
    tmpl::filter<TagList,
                 tmpl::not_<tmpl::bind<db::is_compute_item, tmpl::_1>>>;

namespace DataBox_detail {
template <class ItemsList, class ComputeItemsList>
struct compute_dbox_type;

template <class... ItemsPack, class ComputeItemsList>
struct compute_dbox_type<tmpl::list<ItemsPack...>, ComputeItemsList> {
  using type = decltype(db::create<tmpl::list<ItemsPack...>, ComputeItemsList>(
      std::declval<db::item_type<ItemsPack>>()...));
};
}  // namespace DataBox_detail

/*!
 * \ingroup DataBoxGroup
 * \brief Returns the type of the DataBox that would be constructed from the
 * `TagList` of tags.
 */
template <class TagList>
using compute_databox_type = typename DataBox_detail::compute_dbox_type<
    get_items<TagList>, get_compute_items<TagList>>::type;
}  // namespace db
