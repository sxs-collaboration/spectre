// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines classes and functions used for manipulating DataBox's

#pragma once

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <ostream>
#include <pup.h>
#include <string>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Deferred.hpp"
#include "DataStructures/DataBox/Subitems.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataBox/TagTraits.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/StaticAssert.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/CreateIsCallable.hpp"
#include "Utilities/TypeTraits/IsA.hpp"
#include "Utilities/TypeTraits/IsCallable.hpp"

// IWYU pragma: no_forward_declare brigand::get_destination
// IWYU pragma: no_forward_declare brigand::get_source

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

// @{
/// \ingroup DataBoxGroup
/// Equal to `true` if `Tag` can be retrieved from a `DataBox` of type
/// `DataBoxType`.
template <typename Tag, typename DataBoxType>
using tag_is_retrievable = tmpl::any<typename DataBoxType::tags_list,
                                     std::is_base_of<tmpl::pin<Tag>, tmpl::_1>>;

template <typename Tag, typename DataBoxType>
constexpr bool tag_is_retrievable_v =
    tag_is_retrievable<Tag, DataBoxType>::value;
// @}

namespace detail {
template <class Tag, class Type>
class DataBoxLeaf {
  using value_type = Deferred<Type>;
  value_type value_;

 public:
  constexpr DataBoxLeaf() noexcept(
      std::is_nothrow_default_constructible_v<value_type>)
      : value_() {
    static_assert(!std::is_reference_v<value_type>,
                  "Cannot default construct a reference element in a "
                  "DataBox");
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

template <typename Tag>
using append_subitem_tags =
    tmpl::push_front<typename db::Subitems<Tag>::type, Tag>;

template <typename TagsList>
struct expand_subitems_impl;

template <typename... Tags>
struct expand_subitems_impl<tmpl::list<Tags...>> {
  using type = tmpl::append<append_subitem_tags<Tags>...>;
};

template <typename TagsList>
using expand_subitems = typename expand_subitems_impl<TagsList>::type;

template <typename Tag>
using has_subitems =
    tmpl::not_<std::is_same<typename Subitems<Tag>::type, tmpl::list<>>>;

template <typename ComputeTag, typename ArgumentTag,
          typename FoundComputeItemInBox>
struct report_missing_compute_item_argument {
  static_assert(std::is_same_v<ComputeTag, void>,
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
  using argument_check_assertion = tmpl::transform<
      typename ComputeTag::argument_tags,
      report_missing_compute_item_argument<
          tmpl::pin<ComputeTag>, tmpl::_1,
          has_unique_matching_tag<tmpl::pin<TagList>, tmpl::_1>>>;
#endif  // SPECTRE_DEBUG
  // These edges record that a compute item's value depends on the
  // values of it's arguments.
  using compute_tag_argument_edges = tmpl::transform<
      typename ComputeTag::argument_tags,
      tmpl::bind<tmpl::edge,
                 tmpl::bind<first_matching_tag, tmpl::pin<TagList>, tmpl::_1>,
                 tmpl::pin<ComputeTag>>>;
  // These edges record that the values of the subitems of a compute
  // item depend on the value of the compute item itself.
  using subitem_reverse_edges =
      tmpl::transform<typename Subitems<ComputeTag>::type,
                      tmpl::bind<tmpl::edge, tmpl::pin<ComputeTag>, tmpl::_1>>;

  using type = tmpl::append<compute_tag_argument_edges, subitem_reverse_edges>;
};
}  // namespace detail

namespace detail {
template <typename Tag, typename = std::nullptr_t>
struct check_simple_or_compute_tag {
  static_assert(std::is_same_v<Tag, const void* const*>,
                "All tags added to a DataBox must derive off of db::SimpleTag "
                "or db::ComputeTag, you cannot add a base tag itself. See the "
                "first template parameter of "
                "db::detail::check_simple_or_compute_tag to see "
                "the problematic tag.");
};
template <typename Tag>
struct check_simple_or_compute_tag<Tag, Requires<is_non_base_tag_v<Tag>>> {};
}  // namespace detail

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
    : private detail::DataBoxLeaf<
          Tags, detail::storage_type<Tags, tmpl::list<Tags...>>>... {
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

  /// A list of all the immutable item tags used to create the DataBox
  ///
  /// \note This does not include subitems of immutable items
  using immutable_item_creation_tags =
      tmpl::filter<tags_list, db::is_immutable_item_tag<tmpl::_1>>;

  /// A list of all the immutable item tags, including their subitems
  using immutable_item_tags =
      detail::expand_subitems<immutable_item_creation_tags>;

  /// A list of all the mutable item tags, including their subitems
  using mutable_item_tags =
      tmpl::list_difference<tags_list, immutable_item_tags>;

  /// A list of the expanded simple subitems, not including the main Subitem
  /// tags themselves.
  ///
  /// Specifically, if there is a `Variables<Tag0, Tag1>`, then this list would
  /// contain `Tag0, Tag1`.
  using mutable_subitem_tags = tmpl::flatten<
      tmpl::transform<mutable_item_tags, db::Subitems<tmpl::_1>>>;

  /// \cond HIDDEN_SYMBOLS
  /*!
   * \note the default constructor is only used for serialization
   */
  DataBox() = default;
  DataBox(DataBox&& rhs) noexcept(
      tmpl2::flat_all_v<
          std::is_nothrow_move_constructible_v<detail::DataBoxLeaf<
              Tags, detail::storage_type<Tags, tmpl::list<Tags...>>>>...>) =
      default;
  DataBox& operator=(DataBox&& rhs) noexcept(
      tmpl2::flat_all_v<std::is_nothrow_move_assignable_v<detail::DataBoxLeaf<
          Tags, detail::storage_type<Tags, tmpl::list<Tags...>>>>...>) {
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
    EXPAND_PACK_LEFT_TO_RIGHT(detail::check_simple_or_compute_tag<Tags>{});
  }
#else   // ifdef SPECTRE_DEBUG
  ~DataBox() = default;
#endif  // ifdef SPECTRE_DEBUG

  /// \endcond

  /// \cond HIDDEN_SYMBOLS
  /// Retrieve the tag `Tag`, should be called by the free function db::get
  template <typename Tag,
            Requires<not std::is_same_v<Tag, ::Tags::DataBox>> = nullptr>
  auto get() const noexcept -> detail::const_item_type<Tag, tags_list>;

  /// Retrieve the tag `Tag`, should be called by the free function db::get
  template <typename Tag,
            Requires<std::is_same_v<Tag, ::Tags::DataBox>> = nullptr>
  auto get() const noexcept -> const DataBox<tags_list>&;
  /// \endcond

  // clang-tidy: no non-const references
  void pup(PUP::er& p) noexcept {  // NOLINT
    using non_subitems_tags =
        tmpl::list_difference<mutable_item_tags,
                              mutable_subitem_tags>;

    // We do not send subitems for both simple items and compute items since
    // they can be reconstructed very cheaply.
    pup_impl(p, non_subitems_tags{}, immutable_item_creation_tags{});
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

 private:
  template <typename... MutateTags, typename TagList, typename Invokable,
            typename... Args>
  // clang-tidy: redundant declaration
  friend void mutate(gsl::not_null<DataBox<TagList>*> box,             // NOLINT
                     Invokable&& invokable, Args&&... args) noexcept;  // NOLINT

  /*!
   * \requires Type `T` is one of the Tags corresponding to an object stored in
   * the DataBox
   *
   * \note This should not be used outside of implementation details
   *
   * @return The lazy object corresponding to the Tag `T`
   */
  template <typename T>
  const Deferred<detail::storage_type<T, tags_list>>& get_deferred() const
      noexcept {
    return static_cast<const detail::DataBoxLeaf<
        T, detail::storage_type<T, tags_list>>&>(*this)
        .get();
  }

  template <typename T>
  Deferred<detail::storage_type<T, tags_list>>& get_deferred() noexcept {
    return static_cast<
               detail::DataBoxLeaf<T, detail::storage_type<T, tags_list>>&>(
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

  template <typename ComputeItem, typename FullTagList, bool CheckArgs,
            typename... ComputeItemArgumentsTags>
  constexpr void add_compute_item_to_box_impl(
      tmpl::list<ComputeItemArgumentsTags...> /*meta*/) noexcept;

  template <typename Tag, typename FullTagList, bool CheckArgs = true>
  constexpr void add_compute_item_to_box() noexcept;
  // End adding compute items

  // Adding simple items
  template <typename ParentTag>
  constexpr void add_subitem_tags_to_box(tmpl::list<> /*meta*/) noexcept {}

  template <typename ParentTag, typename... Subtags>
  constexpr void add_subitem_tags_to_box(
      tmpl::list<Subtags...> /*meta*/) noexcept;

  template <size_t ArgsIndex, typename Tag, typename... Ts>
  constexpr char add_item_to_box(std::tuple<Ts...>& tupull) noexcept;
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
  constexpr void merge_old_box(db::DataBox<tmpl::list<OldTags...>>&& old_box,
                               tmpl::list<TagsToCopy...> /*meta*/) noexcept;

  // clang-tidy: no non-const references
  template <typename... NonSubitemsTags, typename... ComputeTags>
  void pup_impl(PUP::er& p, tmpl::list<NonSubitemsTags...> /*meta*/,  // NOLINT
                tmpl::list<ComputeTags...> /*meta*/) noexcept;

  // Mutating items in the DataBox
  template <typename ParentTag>
  constexpr void mutate_subitem_tags_in_box(tmpl::list<> /*meta*/) noexcept {}

  template <typename ParentTag, typename... Subtags>
  constexpr void mutate_subitem_tags_in_box(
      tmpl::list<Subtags...> /*meta*/) noexcept;

  template <typename ComputeItem,
            Requires<not db::is_compute_tag_v<ComputeItem>> = nullptr>
  constexpr void add_reset_compute_item_to_box(tmpl::list<> /*meta*/) noexcept {
  }

  template <typename ComputeItem, typename... ComputeItemArgumentsTags,
            Requires<db::is_compute_tag_v<ComputeItem>> = nullptr>
  constexpr void add_reset_compute_item_to_box(
      tmpl::list<ComputeItemArgumentsTags...> /*meta*/) noexcept;

  template <typename... ComputeItemsToReset>
  SPECTRE_ALWAYS_INLINE constexpr void reset_compute_items_after_mutate(
      tmpl::list<ComputeItemsToReset...> /*meta*/) noexcept;

  SPECTRE_ALWAYS_INLINE constexpr void reset_compute_items_after_mutate(
      tmpl::list<> /*meta*/) noexcept {}
  // End mutating items in the DataBox

  using edge_list = tmpl::join<tmpl::transform<
      immutable_item_creation_tags,
      detail::create_dependency_graph<tmpl::pin<tags_list>, tmpl::_1>>>;

  bool mutate_locked_box_{false};
};

// Adding compute items
namespace detail {
template <bool IsMutating>
struct compute_item_function_impl;

template <>
struct compute_item_function_impl<false> {
  template <typename FullTagList, typename ComputeItem,
            typename... ComputeItemArgumentsTags>
  static decltype(auto) apply(
      const storage_type<ComputeItemArgumentsTags,
                         FullTagList>&... args) noexcept {
    return ComputeItem::function(convert_to_const_type(args)...);
  }
};

template <>
struct compute_item_function_impl<true> {
  template <typename FullTagList, typename ComputeItem,
            typename... ComputeItemArgumentsTags>
  static decltype(auto) apply(
      const gsl::not_null<storage_type<ComputeItem, FullTagList>*> ret,
      const storage_type<ComputeItemArgumentsTags,
                         FullTagList>&... args) noexcept {
    return ComputeItem::function(ret, convert_to_const_type(args)...);
  }
};

template <typename FullTagList, typename ComputeItem,
          typename... ComputeItemArgumentsTags>
constexpr auto compute_item_function =
    &compute_item_function_impl<has_return_type_member_v<ComputeItem>>::
        template apply<FullTagList, ComputeItem, ComputeItemArgumentsTags...>;

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
    ::db::is_compute_tag_v<Tag>>::template f<Tag>;
}  // namespace detail

template <typename... Tags>
template <typename ParentTag, typename... Subtags>
SPECTRE_ALWAYS_INLINE constexpr void
DataBox<tmpl::list<Tags...>>::add_sub_compute_item_tags_to_box(
    tmpl::list<Subtags...> /*meta*/,
    std::false_type /*has_return_type_member*/) noexcept {
  const auto helper = [lazy_function = get_deferred<ParentTag>()](
                          auto tag) noexcept->decltype(auto) {
    return Subitems<ParentTag>::template create_compute_item<decltype(tag)>(
        lazy_function.get());
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
    Subitems<ParentTag>::template create_compute_item<decltype(tag)>(
        result, lazy_function.get());
  };
  EXPAND_PACK_LEFT_TO_RIGHT(
      (get_deferred<Subtags>() = make_deferred_for_subitem<
           detail::storage_type<Subtags, tmpl::list<Tags...>>>(helper,
                                                               Subtags{})));
}

namespace detail {
// This function exists so that the user can look at the template
// arguments to find out what triggered the static_assert.
template <typename ComputeItem, typename Argument, typename FullTagList>
constexpr char check_compute_item_argument_exists() noexcept {
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
  return '0';
}
}  // namespace detail

template <typename... Tags>
template <typename ComputeItem, typename FullTagList, bool CheckArgs,
          typename... ComputeItemArgumentsTags>
SPECTRE_ALWAYS_INLINE constexpr void
DataBox<tmpl::list<Tags...>>::add_compute_item_to_box_impl(
    tmpl::list<ComputeItemArgumentsTags...> /*meta*/) noexcept {
  // These checks can be expensive for the build, so the template argument
  // allows us to skip them in situations that they would be redundant.
  if constexpr (CheckArgs) {
    DEBUG_STATIC_ASSERT(
        tmpl2::flat_all_v<is_tag_v<ComputeItemArgumentsTags>...>,
        "Cannot have non-DataBoxTag arguments to a ComputeItem. Please make "
        "sure all the specified argument_tags in the ComputeItem derive from "
        "db::SimpleTag.");
    DEBUG_STATIC_ASSERT(
        not tmpl2::flat_any_v<
            std::is_same_v<ComputeItemArgumentsTags, ComputeItem>...>,
        "A ComputeItem cannot take its own Tag as an argument.");
    expand_pack(detail::check_compute_item_argument_exists<
                ComputeItem, ComputeItemArgumentsTags, FullTagList>()...);
  }
  get_deferred<ComputeItem>() =
      make_deferred<detail::storage_type<ComputeItem, FullTagList>>(
          detail::compute_item_function<FullTagList, ComputeItem,
                                        ComputeItemArgumentsTags...>,
          get_deferred<ComputeItemArgumentsTags>()...);
}

template <typename... Tags>
template <typename Tag, typename FullTagList, bool CheckArgs>
SPECTRE_ALWAYS_INLINE constexpr void
db::DataBox<tmpl::list<Tags...>>::add_compute_item_to_box() noexcept {
  add_compute_item_to_box_impl<Tag, FullTagList, CheckArgs>(
      tmpl::transform<typename Tag::argument_tags,
                      tmpl::bind<detail::first_matching_tag,
                                 tmpl::pin<tmpl::list<Tags...>>, tmpl::_1>>{});
  add_sub_compute_item_tags_to_box<Tag>(
      typename Subitems<Tag>::type{},
      typename detail::has_return_type_member<Subitems<Tag>>::type{});
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
    using ItemType = detail::storage_type<tag, tmpl::list<Tags...>>;
    get_deferred<tag>() = Deferred<ItemType>(ItemType{});
    Subitems<ParentTag>::template create_item<tag>(
        make_not_null(&get_deferred<ParentTag>().mutate()),
        make_not_null(&get_deferred<tag>().mutate()));
  };

  EXPAND_PACK_LEFT_TO_RIGHT(helper(Subtags{}));
}

template <typename... Tags>
template <size_t ArgsIndex, typename Tag, typename... Ts>
SPECTRE_ALWAYS_INLINE constexpr char
db::DataBox<tmpl::list<Tags...>>::add_item_to_box(
    std::tuple<Ts...>& tupull) noexcept {
  using ArgType = std::tuple_element_t<ArgsIndex, std::tuple<Ts...>>;
  static_assert(not tt::is_a<Deferred, std::decay_t<ArgType>>::value,
                "Cannot pass a Deferred into the DataBox as an Item. This "
                "functionally can trivially be added, however it is "
                "intentionally omitted because users of DataBox are not "
                "supposed to deal with Deferred.");
  get_deferred<Tag>() =
      Deferred<detail::storage_type<Tag, tmpl::list<Tags...>>>(
          std::forward<ArgType>(std::get<ArgsIndex>(tupull)));
  add_subitem_tags_to_box<Tag>(typename Subitems<Tag>::type{});
  return '0';  // must return in constexpr function
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

namespace detail {
// This function (and its unused template argument) exist so that
// users can see what tag has the wrong type when the static_assert
// fails.
template <typename Tag, typename TagType, typename SuppliedType>
constexpr int check_argument_type() noexcept {
  static_assert(std::is_same_v<TagType, SuppliedType>,
                "The type of each Tag must be the same as the type being "
                "passed into the function creating the new DataBox.  See the "
                "function template parameters for the tag, expected type, and "
                "supplied type.");
  return 0;
}
}  // namespace detail

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
  expand_pack(detail::check_argument_type<TagsInArgsOrder,
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
  expand_pack(detail::check_argument_type<AddTags, typename AddTags::type,
                                          std::decay_t<Args>>()...);

  merge_old_box(std::forward<Box>(old_box), tmpl::list<KeepTags...>{});

  // Add in new simple and compute tags

// Silence "maybe-uninitialized" warning for GCC-6. The warning only occurs in
// Release mode.
// Note that clang also defines `__GNUC__`, so we need to exclude it.
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 7
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif  // defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 7
  std::tuple<Args...> args_tuple(std::forward<Args>(args)...);
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 7
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 7
  add_items_to_box<tmpl::list<Tags...>>(
      args_tuple, tmpl::list<AddTags...>{},
      std::make_index_sequence<sizeof...(AddTags)>{},
      tmpl::list<AddComputeTags...>{});
}
/// \endcond

////////////////////////////////////////////////////////////////
// Serialization of DataBox

template <typename... Tags>
template <typename... NonSubitemsTags, typename... ComputeTags>
void DataBox<tmpl::list<Tags...>>::pup_impl(
    PUP::er& p, tmpl::list<NonSubitemsTags...> /*meta*/,
    tmpl::list<ComputeTags...> /*meta*/) noexcept {
  const auto pup_simple_item = [&p, this ](auto current_tag) noexcept {
    (void)this;  // Compiler bug warning this capture is not used
    using tag = decltype(current_tag);
    if (p.isUnpacking()) {
      detail::storage_type<tag, tmpl::list<Tags...>> t{};
      p | t;
      get_deferred<tag>() =
          Deferred<detail::storage_type<tag, tmpl::list<Tags...>>>(
              std::move(t));
      add_subitem_tags_to_box<tag>(typename Subitems<tag>::type{});
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
      add_compute_item_to_box<tag, tmpl::list<Tags...>, false>();
    }
    get_deferred<tag>().pack_unpack_lazy_function(p);
  };
  (void)pup_compute_item;  // Silence GCC warning about unused variable
  EXPAND_PACK_LEFT_TO_RIGHT(pup_compute_item(ComputeTags{}));
}

////////////////////////////////////////////////////////////////
// Mutating items in the DataBox
// Classes and functions necessary for db::mutate to work
template <typename... Tags>
template <typename ComputeItem, typename... ComputeItemArgumentsTags,
          Requires<db::is_compute_tag_v<ComputeItem>>>
SPECTRE_ALWAYS_INLINE constexpr void
DataBox<tmpl::list<Tags...>>::add_reset_compute_item_to_box(
    tmpl::list<ComputeItemArgumentsTags...> /*meta*/) noexcept {
  get_deferred<ComputeItem>().reset();
  mutate_subitem_tags_in_box<ComputeItem>(
      typename Subitems<ComputeItem>::type{});
}

template <typename... Tags>
template <typename... ComputeItemsToReset>
SPECTRE_ALWAYS_INLINE constexpr void
db::DataBox<tmpl::list<Tags...>>::reset_compute_items_after_mutate(
    tmpl::list<ComputeItemsToReset...> /*meta*/) noexcept {
  EXPAND_PACK_LEFT_TO_RIGHT(add_reset_compute_item_to_box<ComputeItemsToReset>(
      detail::get_argument_list<ComputeItemsToReset>{}));
  using current_tags_to_reset = tmpl::list<ComputeItemsToReset...>;
  using next_compute_tags_to_reset = tmpl::list_difference<
      tmpl::remove_duplicates<
          tmpl::transform<tmpl::append<tmpl::filter<
                              typename DataBox<tmpl::list<Tags...>>::edge_list,
                              std::is_same<tmpl::pin<ComputeItemsToReset>,
                                           tmpl::get_source<tmpl::_1>>>...>,
                          tmpl::get_destination<tmpl::_1>>>,
      current_tags_to_reset>;
  reset_compute_items_after_mutate(next_compute_tags_to_reset{});
}

template <typename... Tags>
template <typename ParentTag, typename... Subtags>
SPECTRE_ALWAYS_INLINE constexpr void
db::DataBox<tmpl::list<Tags...>>::mutate_subitem_tags_in_box(
    tmpl::list<Subtags...> /*meta*/) noexcept {
  const auto helper = make_overloader(
    [this](auto tag_v, std::true_type /*is_compute_tag*/) noexcept {
      (void)this;  // Compiler bug warns about unused this capture
      using tag = decltype(tag_v);
      get_deferred<tag>().reset();
    },
    [this](auto tag_v, std::false_type /*is_compute_tag*/) noexcept {
      (void)this;  // Compiler bug warns about unused this capture
      using tag = decltype(tag_v);
      Subitems<ParentTag>::template create_item<tag>(
          make_not_null(&get_deferred<ParentTag>().mutate()),
          make_not_null(&get_deferred<tag>().mutate()));
    });

  EXPAND_PACK_LEFT_TO_RIGHT(helper(Subtags{}, is_compute_tag<ParentTag>{}));
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
          detail::has_unique_matching_tag_v<TagList, MutateTags>...>,
      "One of the tags being mutated could not be found in the DataBox or "
      "is a base tag identifying more than one tag.");
  static_assert(not tmpl2::flat_any_v<tmpl::list_contains_v<
                    typename DataBox<TagList>::immutable_item_tags,
                    detail::first_matching_tag<TagList, MutateTags>>...>,
                "Cannot mutate a compute item");
  if (UNLIKELY(box->mutate_locked_box_)) {
    ERROR(
        "Unable to mutate a DataBox that is already being mutated. This "
        "error occurs when mutating a DataBox from inside the invokable "
        "passed to the mutate function.");
  }
  box->mutate_locked_box_ = true;
  invokable(
      make_not_null(&box->template get_deferred<
                            detail::first_matching_tag<TagList, MutateTags>>()
                         .mutate())...,
      std::forward<Args>(args)...);
  using mutate_tags_list =
      tmpl::list<detail::first_matching_tag<TagList, MutateTags>...>;
  // For all the tags in the DataBox, check if one of their subtags is
  // being mutated and if so add the parent to the list of tags
  // being mutated. Then, remove any tags that would be passed
  // multiple times.
  using extra_mutated_tags = tmpl::list_difference<
      tmpl::filter<TagList,
                   tmpl::bind<tmpl::found, Subitems<tmpl::_1>,
                              tmpl::pin<tmpl::bind<tmpl::list_contains,
                                                   tmpl::pin<mutate_tags_list>,
                                                   tmpl::_1>>>>,
      mutate_tags_list>;
  // Extract the subtags inside the MutateTags and reset compute items
  // depending on those too.
  using full_mutated_items =
      tmpl::append<detail::expand_subitems<mutate_tags_list>,
                   extra_mutated_tags>;

  using first_compute_items_to_reset = tmpl::remove_duplicates<
      tmpl::transform<tmpl::filter<typename DataBox<TagList>::edge_list,
                                   tmpl::bind<tmpl::list_contains,
                                              tmpl::pin<full_mutated_items>,
                                              tmpl::get_source<tmpl::_1>>>,
                      tmpl::get_destination<tmpl::_1>>>;

  EXPAND_PACK_LEFT_TO_RIGHT(
      box->template mutate_subitem_tags_in_box<MutateTags>(
          typename Subitems<MutateTags>::type{}));
  box->template reset_compute_items_after_mutate(
      first_compute_items_to_reset{});

  box->mutate_locked_box_ = false;
}

////////////////////////////////////////////////////////////////
// Retrieving items from the DataBox

/// \cond
template <typename... Tags>
template <typename Tag, Requires<not std::is_same_v<Tag, ::Tags::DataBox>>>
SPECTRE_ALWAYS_INLINE auto DataBox<tmpl::list<Tags...>>::get() const noexcept
    -> detail::const_item_type<Tag, tags_list> {
  DEBUG_STATIC_ASSERT(
      not detail::has_no_matching_tag_v<tags_list, Tag>,
      "Found no tags in the DataBox that match the tag being retrieved.");
  DEBUG_STATIC_ASSERT(
      detail::has_unique_matching_tag_v<tags_list, Tag>,
      "Found more than one tag in the DataBox that matches the tag "
      "being retrieved. This happens because more than one tag with the same "
      "base (class) tag was added to the DataBox.");
  using derived_tag = detail::first_matching_tag<tags_list, Tag>;
  if (UNLIKELY(mutate_locked_box_)) {
    ERROR("Unable to retrieve a (compute) item '"
          << db::tag_name<derived_tag>()
          << "' from the DataBox from within a "
             "call to mutate. You must pass these either through the capture "
             "list of the lambda or the constructor of a class, this "
             "restriction exists to avoid complexity.");
  }
  return detail::convert_to_const_type(get_deferred<derived_tag>().get());
}

template <typename... Tags>
template <typename Tag, Requires<std::is_same_v<Tag, ::Tags::DataBox>>>
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


namespace detail {
template <class ItemsList, class ComputeItemsList>
struct compute_dbox_type;

template <class... ItemsPack, class ComputeItemsList>
struct compute_dbox_type<tmpl::list<ItemsPack...>, ComputeItemsList> {
  using full_items = detail::expand_subitems<tmpl::list<ItemsPack...>>;
  using full_compute_items = detail::expand_subitems<ComputeItemsList>;
  using type = DataBox<tmpl::append<full_items, full_compute_items>>;
};
}  // namespace detail

/*!
 * \ingroup DataBoxGroup
 * \brief Get all the Tags that are compute items from the `TagList`
 */
template <class TagList>
using get_compute_items = tmpl::filter<TagList, db::is_compute_tag<tmpl::_1>>;

/*!
 * \ingroup DataBoxGroup
 * \brief Get all the Tags that are items from the `TagList`
 */
template <class TagList>
using get_items =
    tmpl::filter<TagList, tmpl::not_<tmpl::bind<db::is_compute_tag, tmpl::_1>>>;

/*!
 * \ingroup DataBoxGroup
 * \brief Returns the type of the DataBox that would be constructed from the
 * `TagList` of tags.
 */
template <class TagList>
using compute_databox_type =
    typename detail::compute_dbox_type<get_items<TagList>,
                                       get_compute_items<TagList>>::type;
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
      not tmpl::any<AddSimpleTags, is_compute_tag<tmpl::_1>>::value,
      "Cannot add any ComputeTags in the AddSimpleTags list, must use the "
      "AddComputeTags list.");
  static_assert(
      tmpl::all<AddComputeTags, is_compute_tag<tmpl::_1>>::value,
      "Cannot add any SimpleTags in the AddComputeTags list, must use the "
      "AddSimpleTags list.");

  using full_items = detail::expand_subitems<AddSimpleTags>;
  using full_compute_items = detail::expand_subitems<AddComputeTags>;
  using databox_type =
      compute_databox_type<tmpl::append<AddSimpleTags, AddComputeTags>>;

  return databox_type(AddSimpleTags{}, full_items{}, AddComputeTags{},
                      full_compute_items{}, std::forward<Args>(args)...);
}

namespace detail {
template <typename RemoveTags, typename AddTags, typename AddComputeTags,
          typename Box, typename... Args>
SPECTRE_ALWAYS_INLINE constexpr auto create_from(Box&& box,
                                                 Args&&... args) noexcept {
  static_assert(tmpl::size<AddTags>::value == sizeof...(Args),
                "Must pass in as many arguments as AddTags to db::create_from");

  // 1. Full list of old tags, and the derived tags list of the RemoveTags
  using old_box_tags = typename std::decay_t<Box>::tags_list;
  static_assert(
      tmpl::all<RemoveTags, has_unique_matching_tag<tmpl::pin<old_box_tags>,
                                                    tmpl::_1>>::value,
      "One of the tags being removed could not be found in the DataBox or "
      "is a base tag identifying more than one tag.");
  using remove_tags = tmpl::transform<
      RemoveTags,
      tmpl::bind<first_matching_tag, tmpl::pin<old_box_tags>, tmpl::_1>>;

  // 2. Expand simple remove tags and compute remove tags
  using compute_tags_to_remove =
      tmpl::filter<remove_tags, db::is_compute_tag<tmpl::_1>>;
  using compute_tags_to_remove_with_subitems =
      expand_subitems<compute_tags_to_remove>;
  using simple_tags_to_remove =
      tmpl::list_difference<remove_tags, compute_tags_to_remove>;
  using simple_tags_to_remove_with_subitems =
      expand_subitems<simple_tags_to_remove>;

  // 3. Expand AddTags (these are just the simple tags)
  using simple_tags_to_add_with_subitems = expand_subitems<AddTags>;
  using compute_tags_to_add_with_subitems = expand_subitems<AddComputeTags>;

  // 4. Create new list of tags by removing all the remove tags, and adding all
  // the AddTags, including subitems
  using simple_tags_to_keep =
      tmpl::list_difference<typename std::decay_t<Box>::mutable_item_tags,
                            simple_tags_to_remove_with_subitems>;
  using new_simple_tags =
      tmpl::append<simple_tags_to_keep, simple_tags_to_add_with_subitems>;

  // 5. Create the list of compute items with the RemoveTags removed
  using compute_tags_to_keep = tmpl::list_difference<
      typename std::decay_t<Box>::immutable_item_tags,
      compute_tags_to_remove_with_subitems>;

  // 6. List of the old tags that are being kept
  using old_tags_to_keep =
      tmpl::append<simple_tags_to_keep, compute_tags_to_keep>;

  // 7. List of the new tags, we only need to expand the AddComputeTags now
  using new_tag_list = tmpl::append<new_simple_tags, compute_tags_to_keep,
                                    compute_tags_to_add_with_subitems>;

  DEBUG_STATIC_ASSERT(
      tmpl::size<tmpl::list_difference<AddTags, RemoveTags>>::value ==
          tmpl::size<AddTags>::value,
      "Use db::mutate to mutate simple items, do not remove and add them with "
      "db::create_from.");

#ifdef SPECTRE_DEBUG
  // Check that we're not removing a subitem itself, should remove the parent.
  using compute_subitems_tags =
      tmpl::filter<typename std::decay_t<Box>::immutable_item_creation_tags,
                   tmpl::bind<has_subitems, tmpl::_1>>;

  using compute_only_expand_subitems_tags = tmpl::flatten<
      tmpl::transform<compute_subitems_tags, db::Subitems<tmpl::_1>>>;
  using all_only_subitems_tags = tmpl::append<
      typename std::decay_t<Box>::mutable_subitem_tags,
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
}  // namespace detail

/*!
 * \ingroup DataBoxGroup
 * \brief Create a new DataBox from an existing one adding or removing items
 * and compute items
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
 */
template <typename RemoveTags, typename AddTags = tmpl::list<>,
          typename AddComputeTags = tmpl::list<>, typename TagsList,
          typename... Args>
SPECTRE_ALWAYS_INLINE constexpr auto create_from(db::DataBox<TagsList>&& box,
                                                 Args&&... args) noexcept {
  return detail::create_from<RemoveTags, AddTags, AddComputeTags>(
      std::move(box), std::forward<Args>(args)...);
}

namespace detail {
CREATE_IS_CALLABLE(apply)
CREATE_IS_CALLABLE_V(apply)

template <typename Func, typename... Args>
constexpr void error_function_not_callable() noexcept {
  static_assert(std::is_same_v<Func, void>,
                "The function is not callable with the expected arguments.  "
                "See the first template parameter of "
                "error_function_not_callable for the function type and "
                "the remaining arguments for the parameters that cannot be "
                "passed.");
}

template <typename DataBoxTags, typename... TagsToRetrieve>
constexpr bool check_tags_are_in_databox(
    DataBoxTags /*meta*/, tmpl::list<TagsToRetrieve...> /*meta*/) noexcept {
  static_assert(
      (tag_is_retrievable_v<TagsToRetrieve, DataBox<DataBoxTags>> and ...),
      "A desired tag is not in the DataBox.  See the first template "
      "argument of check_tags_are_in_databox for the missing tag, and the "
      "second for the "
      "available tags.");
  return true;
}

template <typename... ArgumentTags, typename F, typename BoxTags,
          typename... Args>
static constexpr auto apply(F&& f, const DataBox<BoxTags>& box,
                            tmpl::list<ArgumentTags...> /*meta*/,
                            Args&&... args) noexcept {
  if constexpr (is_apply_callable_v<
                    F, const_item_type<ArgumentTags, BoxTags>..., Args...>) {
    return F::apply(::db::get<ArgumentTags>(box)...,
                    std::forward<Args>(args)...);
  } else if constexpr (::tt::is_callable_v<
                           std::remove_pointer_t<F>,
                           tmpl::conditional_t<
                               std::is_same_v<ArgumentTags, ::Tags::DataBox>,
                               const DataBox<BoxTags>&,
                               const_item_type<ArgumentTags, BoxTags>>...,
                           Args...>) {
    return std::forward<F>(f)(::db::get<ArgumentTags>(box)...,
                              std::forward<Args>(args)...);
  } else {
    error_function_not_callable<
        std::remove_pointer_t<F>,
        tmpl::conditional_t<std::is_same_v<ArgumentTags, ::Tags::DataBox>,
                            const DataBox<BoxTags>&,
                            const_item_type<ArgumentTags, BoxTags>>...,
        Args...>();
  }
}
}  // namespace detail

// @{
/*!
 * \ingroup DataBoxGroup
 * \brief Apply the invokable `f` with argument Tags `TagsList` from
 * DataBox `box`
 *
 * \details
 * `f` must either be invokable with the arguments of type
 * `db::const_item_type<TagsList>..., Args...` where the first pack expansion
 * is over the elements in the type list `ArgumentTags`, or have a static
 * `apply` function that is callable with the same types.
 * If the class that implements the static `apply` functions also provides an
 * `argument_tags` typelist, then it is used and no explicit `ArgumentTags`
 * template parameter should be specified.
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
 * `sizeof...(Tags)` arguments of types `db::const_item_type<Tags>...`,  and
 * `sizeof...(Args)` arguments of types `Args...`,
 * \code
 * result = func(box, db::get<Tags>(box)..., args...);
 * \endcode
 *
 * \example
 * \snippet Test_DataBox.cpp apply_example
 * Using a struct with an `apply` method:
 * \snippet Test_DataBox.cpp apply_struct_example
 * If the class `F` has no state, you can also use the stateless overload of
 * `apply`: \snippet Test_DataBox.cpp apply_stateless_struct_example
 *
 * \see DataBox
 * \tparam ArgumentTags typelist of Tags in the order that they are to be passed
 * to `f`
 * \tparam F The invokable to apply
 */
template <typename ArgumentTags, typename F, typename BoxTags, typename... Args>
SPECTRE_ALWAYS_INLINE constexpr auto apply(F&& f, const DataBox<BoxTags>& box,
                                           Args&&... args) noexcept {
  detail::check_tags_are_in_databox(
      BoxTags{}, tmpl::remove<ArgumentTags, ::Tags::DataBox>{});
  return detail::apply(std::forward<F>(f), box, ArgumentTags{},
                       std::forward<Args>(args)...);
}

template <typename F, typename BoxTags, typename... Args>
SPECTRE_ALWAYS_INLINE constexpr auto apply(F&& f, const DataBox<BoxTags>& box,
                                           Args&&... args) noexcept {
  return apply<typename std::decay_t<F>::argument_tags>(
      std::forward<F>(f), box, std::forward<Args>(args)...);
}

template <typename F, typename BoxTags, typename... Args>
SPECTRE_ALWAYS_INLINE constexpr auto apply(const DataBox<BoxTags>& box,
                                           Args&&... args) noexcept {
  return apply(F{}, box, std::forward<Args>(args)...);
}
// @}

namespace detail {
template <typename... ReturnTags, typename... ArgumentTags, typename F,
          typename BoxTags, typename... Args>
SPECTRE_ALWAYS_INLINE constexpr void mutate_apply(
    F&& f, const gsl::not_null<db::DataBox<BoxTags>*> box,
    tmpl::list<ReturnTags...> /*meta*/, tmpl::list<ArgumentTags...> /*meta*/,
    Args&&... args) noexcept {
  static_assert(
      not tmpl2::flat_any_v<std::is_same_v<ArgumentTags, Tags::DataBox>...> and
          not tmpl2::flat_any_v<std::is_same_v<ReturnTags, Tags::DataBox>...>,
      "Cannot pass a DataBox to mutate_apply since the db::get won't work "
      "inside mutate_apply.");
  if constexpr (is_apply_callable_v<
                    F, const gsl::not_null<item_type<ReturnTags, BoxTags>*>...,
                    const_item_type<ArgumentTags, BoxTags>..., Args...>) {
    ::db::mutate<ReturnTags...>(
        box,
        [
        ](const gsl::not_null<item_type<ReturnTags, BoxTags>*>... mutated_items,
          const_item_type<ArgumentTags, BoxTags>... args_items,
          decltype(std::forward<Args>(args))... l_args) noexcept {
          return std::decay_t<F>::apply(mutated_items..., args_items...,
                                        std::forward<Args>(l_args)...);
        },
        db::get<ArgumentTags>(*box)..., std::forward<Args>(args)...);
  } else if constexpr (
      ::tt::is_callable_v<
          F, const gsl::not_null<item_type<ReturnTags, BoxTags>*>...,
          const_item_type<ArgumentTags, BoxTags>..., Args...>) {
    ::db::mutate<ReturnTags...>(
        box,
        [&f](const gsl::not_null<
                 item_type<ReturnTags, BoxTags>*>... mutated_items,
             const_item_type<ArgumentTags, BoxTags>... args_items,
             decltype(std::forward<Args>(args))... l_args) noexcept {
          return f(mutated_items..., args_items...,
                   std::forward<Args>(l_args)...);
        },
        db::get<ArgumentTags>(*box)..., std::forward<Args>(args)...);
  } else {
    error_function_not_callable<
        F, gsl::not_null<item_type<ReturnTags, BoxTags>*>...,
        const_item_type<ArgumentTags, BoxTags>..., Args...>();
  }
}
}  // namespace detail

// @{
/*!
 * \ingroup DataBoxGroup
 * \brief Apply the invokable `f` mutating items `MutateTags` and taking as
 * additional arguments `ArgumentTags` and `args`.
 *
 * \details
 * `f` must either be invokable with the arguments of type
 * `gsl::not_null<db::item_type<MutateTags>*>...,
 * db::const_item_type<ArgumentTags>..., Args...`
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
 * If the class `F` has no state, like in this example,
 * \snippet Test_DataBox.cpp mutate_apply_struct_definition_example
 * you can also use the stateless overload of `mutate_apply`:
 * \snippet Test_DataBox.cpp mutate_apply_struct_example_stateless
 *
 * \tparam MutateTags typelist of Tags to mutate
 * \tparam ArgumentTags typelist of additional items to retrieve from the
 * DataBox
 * \tparam F The invokable to apply
 */
template <typename MutateTags, typename ArgumentTags, typename F,
          typename BoxTags, typename... Args>
SPECTRE_ALWAYS_INLINE constexpr void mutate_apply(
    F&& f, const gsl::not_null<DataBox<BoxTags>*> box,
    Args&&... args) noexcept {
  detail::check_tags_are_in_databox(BoxTags{}, MutateTags{});
  detail::check_tags_are_in_databox(BoxTags{}, ArgumentTags{});
  detail::mutate_apply(std::forward<F>(f), box, MutateTags{}, ArgumentTags{},
                       std::forward<Args>(args)...);
}

template <typename F, typename BoxTags, typename... Args>
SPECTRE_ALWAYS_INLINE constexpr void mutate_apply(
    F&& f, const gsl::not_null<DataBox<BoxTags>*> box,
    Args&&... args) noexcept {
  mutate_apply<typename std::decay_t<F>::return_tags,
               typename std::decay_t<F>::argument_tags>(
      std::forward<F>(f), box, std::forward<Args>(args)...);
}

template <typename F, typename BoxTags, typename... Args>
SPECTRE_ALWAYS_INLINE constexpr void mutate_apply(
    const gsl::not_null<DataBox<BoxTags>*> box, Args&&... args) noexcept {
  mutate_apply(F{}, box, std::forward<Args>(args)...);
}
// @}
}  // namespace db
