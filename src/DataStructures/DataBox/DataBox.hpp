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
#include <sstream>
#include <string>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/IsApplyCallable.hpp"
#include "DataStructures/DataBox/Item.hpp"
#include "DataStructures/DataBox/SubitemTag.hpp"
#include "DataStructures/DataBox/Subitems.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataBox/TagTraits.hpp"
#include "Utilities/CleanupRoutine.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/StaticAssert.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
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

/// @{
/// \ingroup DataBoxGroup
/// Equal to `true` if `Tag` can be retrieved from a `DataBox` of type
/// `DataBoxType`.
template <typename Tag, typename DataBoxType>
using tag_is_retrievable =
    tmpl::or_<std::is_same<Tag, ::Tags::DataBox>,
              tmpl::any<typename DataBoxType::tags_list,
                        std::is_base_of<tmpl::pin<Tag>, tmpl::_1>>>;

template <typename Tag, typename DataBoxType>
constexpr bool tag_is_retrievable_v =
    tag_is_retrievable<Tag, DataBoxType>::value;
/// @}

namespace detail {
template <typename TagsList, typename Tag>
struct creation_tag_impl {
  DEBUG_STATIC_ASSERT(
      not has_no_matching_tag_v<TagsList, Tag>,
      "Found no tags in the DataBox that match the tag being retrieved.");
  DEBUG_STATIC_ASSERT(
      has_unique_matching_tag_v<TagsList, Tag>,
      "Found more than one tag in the DataBox that matches the tag "
      "being retrieved. This happens because more than one tag with the same "
      "base (class) tag was added to the DataBox.");
  using normalized_tag = first_matching_tag<TagsList, Tag>;
  // A list with 0 or 1 elements.  This uses `Tag` rather than
  // `normalized_tag` because subitems of compute tags normalize to
  // `Tags::Subitem<...>`, which is not what is in the `Subitems`
  // list.
  using parent_of_subitem = tmpl::filter<
      TagsList, tmpl::lazy::list_contains<Subitems<tmpl::_1>, tmpl::pin<Tag>>>;
  using type = tmpl::front<tmpl::push_back<parent_of_subitem, normalized_tag>>;
};
}  // namespace detail

/*!
 * \ingroup DataBoxGroup
 * \brief The tag added to \p Box referred to by \p Tag.  This
 * resolves base tags and converts subitems to full items.
 */
template <typename Tag, typename Box>
using creation_tag =
    typename detail::creation_tag_impl<typename Box::tags_list, Tag>::type;

namespace detail {
template <typename Tag>
using has_subitems =
    tmpl::not_<std::is_same<typename Subitems<Tag>::type, tmpl::list<>>>;

template <typename Tag>
constexpr bool has_subitems_v = has_subitems<Tag>::value;

template <typename Tag, typename ParentTag>
struct make_subitem_tag {
  using type = ::Tags::Subitem<Tag, ParentTag>;
};

template <typename Tag, typename = std::nullptr_t>
struct append_subitem_tags {
  using type = tmpl::push_front<typename db::Subitems<Tag>::type, Tag>;
};

template <typename ParentTag>
struct append_subitem_tags<ParentTag,
                           Requires<has_subitems_v<ParentTag> and
                                    is_immutable_item_tag_v<ParentTag>>> {
  using type = tmpl::push_front<
      tmpl::transform<typename Subitems<ParentTag>::type,
                      make_subitem_tag<tmpl::_1, tmpl::pin<ParentTag>>>,
      ParentTag>;
};

template <typename Tag>
struct append_subitem_tags<Tag, Requires<not has_subitems_v<Tag>>> {
  using type = tmpl::list<Tag>;
};

template <typename TagsList>
using expand_subitems =
    tmpl::flatten<tmpl::transform<TagsList, append_subitem_tags<tmpl::_1>>>;

template <typename ComputeTag, typename ArgumentTag,
          typename FoundArgumentTagInBox>
struct report_missing_compute_item_argument {
  static_assert(std::is_same_v<ComputeTag, void>,
                "A compute item's argument could not be found in the DataBox "
                "or was found multiple times.  See the first template argument "
                "of report_missing_compute_item_argument for the compute item "
                "and the second for the missing (or duplicated) argument.");
};

template <typename ComputeTag, typename ArgumentTag>
struct report_missing_compute_item_argument<ComputeTag, ArgumentTag,
                                            std::true_type> {
  using type = void;
};

template <typename TagsList, typename ComputeTag>
struct create_compute_tag_argument_edges {
#ifdef SPECTRE_DEBUG
  using argument_check_assertion = tmpl::transform<
      typename ComputeTag::argument_tags,
      report_missing_compute_item_argument<
          tmpl::pin<ComputeTag>, tmpl::_1,
          has_unique_matching_tag<tmpl::pin<TagsList>, tmpl::_1>>>;
#endif  // SPECTRE_DEBUG
  // These edges record that a compute item's value depends on the
  // values of it's arguments.
  using type = tmpl::transform<
      typename ComputeTag::argument_tags,
      tmpl::bind<tmpl::edge,
                 tmpl::bind<first_matching_tag, tmpl::pin<TagsList>, tmpl::_1>,
                 tmpl::pin<ComputeTag>>>;
};

template <typename TagsList, typename ImmutableItemTagsList>
struct create_dependency_graph {
  using immutable_item_argument_edges =
      tmpl::join<tmpl::transform<ImmutableItemTagsList,
                                 detail::create_compute_tag_argument_edges<
                                     tmpl::pin<TagsList>, tmpl::_1>>>;
  using type = immutable_item_argument_edges;
};
}  // namespace detail

/*!
 * \ingroup DataBoxGroup
 * \brief A DataBox stores objects that can be retrieved by using Tags
 * \warning
 * The order of the tags in DataBoxes returned by db::create depends on
 * implementation-defined behavior, and therefore should not be specified in
 * source files. If explicitly naming a DataBox type is necessary they should be
 * generated using db::compute_databox_type.
 *
 * \see db::create
 *
 * @tparam Tags list of DataBoxTag's
 */
template <typename... Tags>
class DataBox<tmpl::list<Tags...>> : private detail::Item<Tags>... {

 public:
  /*!
   * \brief A typelist (`tmpl::list`) of Tags that the DataBox holds
   */
  using tags_list = tmpl::list<Tags...>;

  /// A list of all the immutable item tags, including their subitems.
  /// Immutable items are compute tags and reference tags.
  using immutable_item_tags =
      tmpl::filter<tags_list, db::is_immutable_item_tag<tmpl::_1>>;

  /// A list of all the immutable item tags used to create the DataBox
  ///
  /// \note This does not include subitems of immutable items
  using immutable_item_creation_tags =
      tmpl::remove_if<immutable_item_tags, tt::is_a<::Tags::Subitem, tmpl::_1>>;

  /// A list of all the mutable item tags, including their subitems
  using mutable_item_tags =
      tmpl::filter<tags_list, db::is_mutable_item_tag<tmpl::_1>>;

  /// A list of the expanded simple subitems, not including the main Subitem
  /// tags themselves.
  ///
  /// Specifically, if there is a `Variables<Tag0, Tag1>`, then this list would
  /// contain `Tag0, Tag1`.
  using mutable_subitem_tags = tmpl::flatten<
      tmpl::transform<mutable_item_tags, db::Subitems<tmpl::_1>>>;

  /// A list of all the mutable item tags used to create the DataBox
  ///
  /// \note This does not include subitems of mutable items
  using mutable_item_creation_tags =
      tmpl::list_difference<mutable_item_tags, mutable_subitem_tags>;

  /// A list of all the compute item tags
  using compute_item_tags =
      tmpl::filter<immutable_item_tags, db::is_compute_tag<tmpl::_1>>;

  /// A list of all the reference tags
  using reference_item_tags =
      tmpl::filter<immutable_item_tags, db::is_reference_tag<tmpl::_1>>;

  /// \cond HIDDEN_SYMBOLS
  /*!
   * \note the default constructor is only used for serialization
   */
  DataBox() = default;
  constexpr DataBox(DataBox&& rhs) : detail::Item<Tags>(std::move(rhs))... {
    reset_all_subitems();
  }
  constexpr DataBox& operator=(DataBox&& rhs) {
    if (&rhs != this) {
#if defined(__GNUC__) && !defined(__clang__) && \
    (__GNUC__ < 8 || (__GNUC__ > 10 && __GNUC__ < 12))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif  // defined(__GNUC__) && !defined(__clang__) && (__GNUC__ < 8 ||
        // (__GNUC__ > 10 && __GNUC__ < 12))
      ::expand_pack(
          (get_item<Tags>() = std::move(rhs.template get_item<Tags>()))...);
#if defined(__GNUC__) && !defined(__clang__) && \
    (__GNUC__ < 8 || (__GNUC__ > 10 && __GNUC__ < 12))
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) && !defined(__clang__) && (__GNUC__ < 8 ||
        // (__GNUC__ > 10 && __GNUC__ < 12))
      reset_all_subitems();
    }
    return *this;
  }
  DataBox(const DataBox& rhs) = delete;
  DataBox& operator=(const DataBox& rhs) = delete;
  ~DataBox() = default;

  /// \endcond

  /// Print the expanded type aliases
  std::string print_types() const;

  /// Print the items
  template <bool PrintImmutableItems = true>
  std::string print_items() const;

  /// Retrieve the tag `Tag`, should be called by the free function db::get
  template <typename Tag>
  const auto& get() const;

  /// \brief Copy the items with tags `TagsOfItemsToCopy` from the DataBox
  /// into a TaggedTuple, should be called by the free function db::copy_items
  template <typename... TagsOfItemsToCopy>
  tuples::TaggedTuple<TagsOfItemsToCopy...> copy_items() const;

  /// Retrieve a mutable reference to the tag `Tag`, should be called
  /// by the free function db::get_mutable_reference
  template <typename Tag>
  auto& get_mutable_reference();

  /// Check whether a tags depends on another tag.  Should be called
  /// through the metafunction db::tag_depends_on.
  template <typename Consumer, typename Provider>
  constexpr static bool tag_depends_on();

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) {
    // We do not send subitems for both simple items and compute items since
    // they can be reconstructed very cheaply.
    pup_impl(p, mutable_item_creation_tags{}, immutable_item_creation_tags{});
  }

  template <typename... AddMutableItemTags, typename AddImmutableItemTagsList,
            typename... Args>
  constexpr DataBox(tmpl::list<AddMutableItemTags...> /*meta*/,
                    AddImmutableItemTagsList /*meta*/, Args&&... args);

 private:
  template <typename CopiedItemsTagList, typename DbTagList>
  // clang-tidy: redundant declaration
  friend auto copy_items(const DataBox<DbTagList>& box);

  template <typename... MutateTags, typename TagList, typename Invokable,
            typename... Args>
  // clang-tidy: redundant declaration
  friend decltype(auto) mutate(Invokable&& invokable,
                               gsl::not_null<DataBox<TagList>*> box,  // NOLINT
                               Args&&... args);                       // NOLINT

  // evaluates the compute item corresponding to ComputeTag passing along
  // items fetched via ArgumentTags
  template <typename ComputeTag, typename... ArgumentTags>
  void evaluate_compute_item(tmpl::list<ArgumentTags...> /*meta*/) const;

  // retrieves the reference of the ReferenceTag passing along items fetched via
  // ArgumentTags
  template <typename ReferenceTag, typename... ArgumentTags>
  const auto& get_reference_item(tmpl::list<ArgumentTags...> /*meta*/) const;

  // get a constant reference to the item corresponding to Tag
  template <typename Tag>
  const auto& get_item() const {
    return static_cast<const detail::Item<Tag>&>(*this);
  }

  // get a mutable reference to the item corresponding to Tag
  template <typename Tag>
  auto& get_item() {
    return static_cast<detail::Item<Tag>&>(*this);
  }

  // copy item correspond to Tag
  template <typename Tag>
  auto copy_item() const;

  // copy items corresponding to CopiedItemsTags
  // from the DataBox to a TaggedTuple
  template <typename... CopiedItemsTags>
  tuples::TaggedTuple<CopiedItemsTags...> copy_items(
      tmpl::list<CopiedItemsTags...> /*meta*/) const;

  template <typename ParentTag>
  constexpr void add_mutable_subitems_to_box(tmpl::list<> /*meta*/) {}

  // add mutable items for the subitems of the item corresponding to ParentTag
  template <typename ParentTag, typename... SubitemTags>
  constexpr void add_mutable_subitems_to_box(
      tmpl::list<SubitemTags...> /*meta*/);

  // sets the mutable item corresponding to Tag with the ArgsIndex object in
  // items
  template <size_t ArgsIndex, typename Tag, typename... Ts>
  constexpr char add_mutable_item_to_box(std::tuple<Ts...>& items);

  // set the mutable items corresponding to AddMutableItemTags to the
  // appropriate objects from items, and checks the dependencies of the
  // immutable items corresponding to AddImmutableItemTags
  template <typename... Ts, typename... AddMutableItemTags,
            typename... AddImmutableItemTags, size_t... Is>
  void add_items_to_box(std::tuple<Ts...>& items,
                        tmpl::list<AddMutableItemTags...> /*meta*/,
                        std::index_sequence<Is...> /*meta*/,
                        tmpl::list<AddImmutableItemTags...> /*meta*/);

  // clang-tidy: no non-const references
  template <typename... MutableItemCreationTags,
            typename... ImmutableItemCreationTags>
  void pup_impl(PUP::er& p,  // NOLINT
                tmpl::list<MutableItemCreationTags...> /*meta*/,
                tmpl::list<ImmutableItemCreationTags...> /*meta*/);

  // Mutating items in the DataBox
  template <typename ParentTag>
  constexpr void mutate_mutable_subitems(tmpl::list<> /*meta*/) {}

  template <typename ParentTag, typename... Subtags>
  constexpr void mutate_mutable_subitems(tmpl::list<Subtags...> /*meta*/);

  constexpr void reset_all_subitems();

  template <typename ImmutableItemTag>
  constexpr void reset_compute_item();

  template <typename... TagsOfImmutableItemsToReset>
  SPECTRE_ALWAYS_INLINE constexpr void reset_compute_items_after_mutate(
      tmpl::list<TagsOfImmutableItemsToReset...> /*meta*/);

  SPECTRE_ALWAYS_INLINE constexpr void reset_compute_items_after_mutate(
      tmpl::list<> /*meta*/) {}
  // End mutating items in the DataBox

  using edge_list =
      typename detail::create_dependency_graph<tags_list,
                                               immutable_item_tags>::type;

  bool mutate_locked_box_{false};
};

template <typename... Tags>
std::string DataBox<tmpl::list<Tags...>>::print_types() const {
  std::ostringstream os;
  os << "DataBox type aliases:\n";
  os << "using tags_list = " << pretty_type::get_name<tags_list>() << ";\n";
  os << "using immutable_item_tags = "
     << pretty_type::get_name<immutable_item_tags>() << ";\n";
  os << "using immutable_item_creation_tags = "
     << pretty_type::get_name<immutable_item_creation_tags>() << ";\n";
  os << "using mutable_item_tags = "
     << pretty_type::get_name<mutable_item_tags>() << ";\n";
  os << "using mutable_subitem_tags = "
     << pretty_type::get_name<mutable_subitem_tags>() << ";\n";
  os << "using compute_item_tags = "
     << pretty_type::get_name<compute_item_tags>() << ";\n";
  os << "using reference_item_tags = "
     << pretty_type::get_name<reference_item_tags>() << ";\n";
  os << "using edge_list = " << pretty_type::get_name<edge_list>() << ";\n";
  return os.str();
}

template <typename... Tags>
template <bool PrintImmutableItems>
std::string DataBox<tmpl::list<Tags...>>::print_items() const {
  std::ostringstream os;
  os << "Items:\n";
  const auto print_item = [this, &os](auto tag_v) {
    (void)this;
    using tag = tmpl::type_from<decltype(tag_v)>;
    using type = typename tag::type;
    os << "----------\n";
    os << "Name:  " << pretty_type::get_name<tag>() << "\n";
    os << "Type:  " << pretty_type::get_name<type>() << "\n";
    if constexpr (tt::is_streamable_v<std::ostringstream, type>) {
      os << "Value: " << this->get<tag>() << "\n";
    } else {
      os << "Value: UNSTREAMABLE\n";
    }
  };
  tmpl::for_each<mutable_item_creation_tags>(print_item);
  if constexpr (PrintImmutableItems) {
    tmpl::for_each<immutable_item_creation_tags>(print_item);
  }
  return os.str();
}

namespace detail {
// This function exists so that the user can look at the template
// arguments to find out what triggered the static_assert.
template <typename ImmutableItemTag, typename ArgumentTag, typename TagsList>
constexpr char check_immutable_item_tag_dependency() {
  using immutable_item_tag_index = tmpl::index_of<TagsList, ImmutableItemTag>;
  static_assert(
      tmpl::less<tmpl::index_if<TagsList,
                                std::is_same<tmpl::pin<ArgumentTag>, tmpl::_1>,
                                immutable_item_tag_index>,
                 immutable_item_tag_index>::value,
      "The argument_tags of an immutable item tag must be added before itself. "
      "This is done to ensure no cyclic dependencies arise.  See the first and "
      "second template arguments of check_immutable_item_tag_dependency for "
      "the immutable item tag and its missing (or incorrectly added) argument "
      "tag.  The third template argument is the TagsList of the DataBox (in "
      "which the argument tag should precede the immutable item tag)");
  return '0';
}

template <typename ImmutableItemTag, typename TagsList,
          typename... ArgumentsTags>
SPECTRE_ALWAYS_INLINE constexpr void check_immutable_item_tag_dependencies_impl(
    tmpl::list<ArgumentsTags...> /*meta*/) {
  DEBUG_STATIC_ASSERT(
      tmpl2::flat_all_v<is_tag_v<ArgumentsTags>...>,
      "Cannot have non-DataBoxTag arguments to a ComputeItem or ReferenceItem. "
      "Please make sure all the specified argument_tags derive from "
      "db::SimpleTag or db::BaseTag.");
  DEBUG_STATIC_ASSERT(
      not tmpl2::flat_any_v<std::is_same_v<ArgumentsTags, ImmutableItemTag>...>,
      "A ComputeItem cannot take its own Tag as an argument.");
  expand_pack(detail::check_immutable_item_tag_dependency<
              ImmutableItemTag, ArgumentsTags, TagsList>()...);
}

template <typename ImmutableItemTag, typename TagsList>
SPECTRE_ALWAYS_INLINE constexpr void check_immutable_item_tag_dependencies() {
  check_immutable_item_tag_dependencies_impl<ImmutableItemTag, TagsList>(
      tmpl::transform<typename ImmutableItemTag::argument_tags,
                      tmpl::bind<detail::first_matching_tag,
                                 tmpl::pin<TagsList>, tmpl::_1>>{});
}
}  // namespace detail

template <typename... Tags>
template <typename ParentTag, typename... SubitemTags>
SPECTRE_ALWAYS_INLINE constexpr void
db::DataBox<tmpl::list<Tags...>>::add_mutable_subitems_to_box(
    tmpl::list<SubitemTags...> /*meta*/) {
  const auto add_mutable_subitem_to_box = [this](auto tag_v) {
    (void)this;  // Compiler bug warns this is unused
    using subitem_tag = decltype(tag_v);
    get_item<subitem_tag>() =
        detail::Item<subitem_tag>(typename subitem_tag::type{});
    Subitems<ParentTag>::template create_item<subitem_tag>(
        make_not_null(&get_item<ParentTag>().mutate()),
        make_not_null(&get_item<subitem_tag>().mutate()));
  };

  EXPAND_PACK_LEFT_TO_RIGHT(add_mutable_subitem_to_box(SubitemTags{}));
}

template <typename... Tags>
template <size_t ArgsIndex, typename MutableItemTag, typename... Ts>
SPECTRE_ALWAYS_INLINE constexpr char
db::DataBox<tmpl::list<Tags...>>::add_mutable_item_to_box(
    std::tuple<Ts...>& items) {
  if constexpr (sizeof...(Ts) > 0) {
    using ArgType = std::tuple_element_t<ArgsIndex, std::tuple<Ts...>>;
    get_item<MutableItemTag>() = detail::Item<MutableItemTag>(
        std::forward<ArgType>(std::get<ArgsIndex>(items)));
  }
  add_mutable_subitems_to_box<MutableItemTag>(
      typename Subitems<MutableItemTag>::type{});
  return '0';  // must return in constexpr function
}

// Add items or compute items to the TaggedDeferredTuple `data`. If
// `AddItemTags...` is an empty pack then only compute items are added, while if
// `AddComputeTags...` is an empty pack only items are added. Items are
// always added before compute items.
template <typename... Tags>
template <typename... Ts, typename... AddMutableItemTags,
          typename... AddImmutableItemTags, size_t... Is>
SPECTRE_ALWAYS_INLINE void DataBox<tmpl::list<Tags...>>::add_items_to_box(
    std::tuple<Ts...>& items, tmpl::list<AddMutableItemTags...> /*meta*/,
    std::index_sequence<Is...> /*meta*/,
    tmpl::list<AddImmutableItemTags...> /*meta*/) {
  expand_pack(add_mutable_item_to_box<Is, AddMutableItemTags>(items)...);
  EXPAND_PACK_LEFT_TO_RIGHT(
      detail::check_immutable_item_tag_dependencies<AddImmutableItemTags,
                                                    tags_list>());
}

namespace detail {
// This function (and its unused template argument) exist so that
// users can see what tag has the wrong type when the static_assert
// fails.
template <typename Tag, typename TagType, typename SuppliedType>
constexpr int check_initialization_argument_type() {
  static_assert(std::is_same_v<TagType, SuppliedType>,
                "The type of each Tag must be the same as the type being "
                "passed into the function creating the new DataBox.  See the "
                "template parameters of check_initialization_argument_type for "
                "the tag, expected type, and supplied type.");
  return 0;
}
}  // namespace detail

/// \cond
template <typename... Tags>
template <typename... AddMutableItemTags, typename AddImmutableItemTagsList,
          typename... Args>
constexpr DataBox<tmpl::list<Tags...>>::DataBox(
    tmpl::list<AddMutableItemTags...> /*meta*/,
    AddImmutableItemTagsList /*meta*/, Args&&... args) {
  DEBUG_STATIC_ASSERT(
      sizeof...(Args) == 0 or sizeof...(Args) == sizeof...(AddMutableItemTags),
      "Must pass in as many arguments as AddTags, or none to "
      "default-construct them.");
#ifdef SPECTRE_DEBUG
  if constexpr (sizeof...(Args) > 0) {
    // The check_argument_type call is very expensive compared to the majority
    // of DataBox
    expand_pack(detail::check_initialization_argument_type<
                AddMutableItemTags, typename AddMutableItemTags::type,
                std::decay_t<Args>>()...);
  }
#endif  // SPECTRE_DEBUG

  std::tuple<Args&&...> args_tuple(std::forward<Args>(args)...);
  add_items_to_box(args_tuple, tmpl::list<AddMutableItemTags...>{},
                   std::make_index_sequence<sizeof...(AddMutableItemTags)>{},
                   AddImmutableItemTagsList{});
}
/// \endcond

////////////////////////////////////////////////////////////////
// Serialization of DataBox

template <typename... Tags>
template <typename... MutableItemCreationTags,
          typename... ImmutableItemCreationTags>
void DataBox<tmpl::list<Tags...>>::pup_impl(
    PUP::er& p, tmpl::list<MutableItemCreationTags...> /*meta*/,
    tmpl::list<ImmutableItemCreationTags...> /*meta*/) {
  const auto pup_simple_item = [&p, this](auto current_tag) {
    (void)this;  // Compiler bug warning this capture is not used
    using tag = decltype(current_tag);
    get_item<tag>().pup(p);
    if (p.isUnpacking()) {
      add_mutable_subitems_to_box<tag>(typename Subitems<tag>::type{});
    }
  };
  (void)pup_simple_item;  // Silence GCC warning about unused variable
  EXPAND_PACK_LEFT_TO_RIGHT(pup_simple_item(MutableItemCreationTags{}));

  EXPAND_PACK_LEFT_TO_RIGHT(get_item<ImmutableItemCreationTags>().pup(p));
}

////////////////////////////////////////////////////////////////
// Mutating items in the DataBox
// Classes and functions necessary for db::mutate to work

template <typename... Tags>
template <typename ImmutableItemTag>
SPECTRE_ALWAYS_INLINE constexpr void
DataBox<tmpl::list<Tags...>>::reset_compute_item() {
  // reference items do not need to be reset
  if constexpr (db::is_compute_tag_v<ImmutableItemTag>) {
    get_item<ImmutableItemTag>().reset();
  }
}

// This function recursively calls itself to reset all compute items
// that depended upon the mutated items.  It starts with
// TagsOfImmutableItemsToReset as the tags of the immutable items that directly
// depended upon the mutated items.  This function resets the compute items in
// TagsOfImmutableItemsToReset and then constructs the list of immutable items
// that directly depend on TagsOfImmutableItemsToReset which will become the
// TagsOfImmutableItemsToReset on the next invocation of the function.  The
// recursion terminates when TagsOfImmutableItemsToReset becomes an empty list
// (using the function overload inlined above in the definition of DataBox).
template <typename... Tags>
template <typename... TagsOfImmutableItemsToReset>
SPECTRE_ALWAYS_INLINE constexpr void
db::DataBox<tmpl::list<Tags...>>::reset_compute_items_after_mutate(
    tmpl::list<TagsOfImmutableItemsToReset...> /*meta*/) {
  EXPAND_PACK_LEFT_TO_RIGHT(reset_compute_item<TagsOfImmutableItemsToReset>());
  using current_tags_to_reset = tmpl::list<TagsOfImmutableItemsToReset...>;
  using next_compute_tags_to_reset = tmpl::list_difference<
      tmpl::remove_duplicates<tmpl::transform<
          tmpl::filter<
              typename DataBox<tmpl::list<Tags...>>::edge_list,
              tmpl::lazy::list_contains<
                  tmpl::pin<tmpl::list<TagsOfImmutableItemsToReset...>>,
                  tmpl::get_source<tmpl::_1>>>,
          tmpl::get_destination<tmpl::_1>>>,
      current_tags_to_reset>;
  reset_compute_items_after_mutate(next_compute_tags_to_reset{});
}

template <typename... Tags>
template <typename ParentTag, typename... Subtags>
SPECTRE_ALWAYS_INLINE constexpr void
db::DataBox<tmpl::list<Tags...>>::mutate_mutable_subitems(
    tmpl::list<Subtags...> /*meta*/) {
  const auto helper = [this](auto tag_v) {
    (void)this;  // Compiler bug warns about unused this capture
    using tag = decltype(tag_v);
    Subitems<ParentTag>::template create_item<tag>(
        make_not_null(&get_item<ParentTag>().mutate()),
        make_not_null(&get_item<tag>().mutate()));
  };

  EXPAND_PACK_LEFT_TO_RIGHT(helper(Subtags{}));
}

template <typename... Tags>
SPECTRE_ALWAYS_INLINE constexpr void
db::DataBox<tmpl::list<Tags...>>::reset_all_subitems() {
  tmpl::for_each<mutable_item_tags>([this](auto tag) {
    using Tag = tmpl::type_from<decltype(tag)>;
    this->mutate_mutable_subitems<Tag>(typename Subitems<Tag>::type{});
  });
}

/*!
 * \ingroup DataBoxGroup
 * \brief Allows changing the state of one or more non-computed elements in
 * the DataBox
 *
 * `mutate()`'s second argument is the DataBox from which to retrieve the tags
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
 *
 * The `invokable` may have function return values, and any returns are
 * forwarded as returns to the `db::mutate` call.
 *
 * For convenience in generic programming, if `::Tags::DataBox` is
 * passed as the sole `MutateTags` parameter, this function will
 * simply call its first argument with its remaining arguments.
 * Except for this case, `::Tags::DataBox` is not usable with this
 * function.
 *
 * \warning Using `db::mutate` returns to obtain non-const references or
 * pointers to box items is potentially very dangerous. The \ref DataBoxGroup
 * "DataBox" cannot track any subsequent changes to quantities that have been
 * "unsafely" extracted in this manner, so we consider it undefined behavior to
 * return pointers or references to \ref DataBoxGroup "DataBox" contents.
 */
template <typename... MutateTags, typename TagList, typename Invokable,
          typename... Args>
decltype(auto) mutate(Invokable&& invokable,
                      const gsl::not_null<DataBox<TagList>*> box,
                      Args&&... args) {
  if constexpr ((... or std::is_same_v<MutateTags, Tags::DataBox>)) {
    // This branch doesn't directly access the box, so no locking or
    // resetting is necessary.
    static_assert(
        std::is_same_v<tmpl::list<MutateTags...>, tmpl::list<Tags::DataBox>>,
        "Cannot mutate other tags when obtaining the mutable DataBox.");
    return invokable(box, std::forward<Args>(args)...);
  } else {
    static_assert(
        tmpl2::flat_all_v<
            detail::has_unique_matching_tag_v<TagList, MutateTags>...>,
        "One of the tags being mutated could not be found in the DataBox or "
        "is a base tag identifying more than one tag.");
    static_assert(tmpl2::flat_all_v<tmpl::list_contains_v<
                      typename DataBox<TagList>::mutable_item_tags,
                      detail::first_matching_tag<TagList, MutateTags>>...>,
                  "Can only mutate mutable items");
    if (UNLIKELY(box->mutate_locked_box_)) {
      ERROR(
          "Unable to mutate a DataBox that is already being mutated. This "
          "error occurs when mutating a DataBox from inside the invokable "
          "passed to the mutate function.");
    }
    using mutate_tags_list =
        tmpl::list<detail::first_matching_tag<TagList, MutateTags>...>;
    // For all the tags in the DataBox, check if one of their subtags is
    // being mutated and if so add the parent to the list of tags
    // being mutated. Then, remove any tags that would be passed
    // multiple times.
    using extra_mutated_tags = tmpl::list_difference<
        tmpl::filter<TagList,
                     tmpl::bind<tmpl::found, Subitems<tmpl::_1>,
                                tmpl::pin<tmpl::lazy::list_contains<
                                    tmpl::pin<mutate_tags_list>, tmpl::_1>>>>,
        mutate_tags_list>;
    // Extract the subtags inside the MutateTags and reset compute items
    // depending on those too.
    using full_mutated_items =
        tmpl::append<detail::expand_subitems<mutate_tags_list>,
                     extra_mutated_tags>;

    using first_compute_items_to_reset = tmpl::remove_duplicates<
        tmpl::transform<tmpl::filter<typename DataBox<TagList>::edge_list,
                                     tmpl::lazy::list_contains<
                                         tmpl::pin<full_mutated_items>,
                                         tmpl::get_source<tmpl::_1>>>,
                        tmpl::get_destination<tmpl::_1>>>;

    const CleanupRoutine unlock_box = [&box]() {
      box->mutate_locked_box_ = false;
      EXPAND_PACK_LEFT_TO_RIGHT(
          box->template mutate_mutable_subitems<MutateTags>(
              typename Subitems<MutateTags>::type{}));
      box->template reset_compute_items_after_mutate(
          first_compute_items_to_reset{});
    };
    box->mutate_locked_box_ = true;
    return invokable(
        make_not_null(&box->template get_item<
                              detail::first_matching_tag<TagList, MutateTags>>()
                           .mutate())...,
        std::forward<Args>(args)...);
  }
}

////////////////////////////////////////////////////////////////
// Retrieving items from the DataBox

/// \cond
template <typename... Tags>
template <typename ComputeTag, typename... ArgumentTags>
void DataBox<tmpl::list<Tags...>>::evaluate_compute_item(
    tmpl::list<ArgumentTags...> /*meta*/) const {
  get_item<ComputeTag>().evaluate(get<ArgumentTags>()...);
}

template <typename... Tags>
template <typename ReferenceTag, typename... ArgumentTags>
const auto& DataBox<tmpl::list<Tags...>>::get_reference_item(
    tmpl::list<ArgumentTags...> /*meta*/) const {
  return ReferenceTag::get(get<ArgumentTags>()...);
}

template <typename... Tags>
template <typename Tag>
const auto& DataBox<tmpl::list<Tags...>>::get() const {
  if constexpr (std::is_same_v<Tag, ::Tags::DataBox>) {
    if (UNLIKELY(mutate_locked_box_)) {
      ERROR(
          "Unable to retrieve a (compute) item 'DataBox' from the DataBox from "
          "within a call to mutate. You must pass these either through the "
          "capture list of the lambda or the constructor of a class, this "
          "restriction exists to avoid complexity.");
    }
    return *this;
  } else {
    DEBUG_STATIC_ASSERT(
        not detail::has_no_matching_tag_v<tags_list, Tag>,
        "Found no tags in the DataBox that match the tag being retrieved.");
    DEBUG_STATIC_ASSERT(
        detail::has_unique_matching_tag_v<tags_list, Tag>,
        "Found more than one tag in the DataBox that matches the tag "
        "being retrieved. This happens because more than one tag with the same "
        "base (class) tag was added to the DataBox.");
    using item_tag = detail::first_matching_tag<tags_list, Tag>;
    if (UNLIKELY(mutate_locked_box_)) {
      ERROR("Unable to retrieve a (compute) item '"
            << db::tag_name<item_tag>()
            << "' from the DataBox from within a "
               "call to mutate. You must pass these either through the capture "
               "list of the lambda or the constructor of a class, this "
               "restriction exists to avoid complexity.");
    }
    if constexpr (detail::Item<item_tag>::item_type ==
                  detail::ItemType::Reference) {
      return get_reference_item<item_tag>(typename item_tag::argument_tags{});
    } else {
      if constexpr (detail::Item<item_tag>::item_type ==
                    detail::ItemType::Compute) {
        if (not get_item<item_tag>().evaluated()) {
          evaluate_compute_item<item_tag>(typename item_tag::argument_tags{});
        }
      }
      if constexpr (tt::is_a_v<std::unique_ptr, typename item_tag::type>) {
        return *(get_item<item_tag>().get());
      } else {
        return get_item<item_tag>().get();
      }
    }
  }
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
SPECTRE_ALWAYS_INLINE const auto& get(const DataBox<TagList>& box) {
  return box.template get<Tag>();
}

////////////////////////////////////////////////////////////////
// Copy mutable creation items from the DataBox

/// \cond
template <typename... Tags>
template <typename Tag>
SPECTRE_ALWAYS_INLINE auto DataBox<tmpl::list<Tags...>>::copy_item() const {
  using item_tag = detail::first_matching_tag<tags_list, Tag>;
  using item_type = typename item_tag::type;
  static_assert(tmpl::list_contains_v<mutable_item_creation_tags, item_tag>,
                "Can only copy mutable creation items");
  return deserialize<item_type>(
      serialize<item_type>(get_item<item_tag>().get()).data());
}

template <typename... DbTags>
template <typename... CopiedItemsTags>
tuples::TaggedTuple<CopiedItemsTags...>
DataBox<tmpl::list<DbTags...>>::copy_items(
    tmpl::list<CopiedItemsTags...> /*meta*/) const {
  return tuples::TaggedTuple<CopiedItemsTags...>{
      copy_item<CopiedItemsTags>()...};
}
/// \endcond

/*!
 * \ingroup DataBoxGroup
 * \brief Copy the items from the DataBox into a TaggedTuple
 *
 * \return The objects corresponding to CopiedItemsTagList
 *
 * \note The tags in CopiedItemsTagList must be a subset of
 * the mutable_item_creation_tags of the DataBox
 */
template <typename CopiedItemsTagList, typename DbTagList>
SPECTRE_ALWAYS_INLINE auto copy_items(const DataBox<DbTagList>& box) {
  return box.copy_items(CopiedItemsTagList{});
}

////////////////////////////////////////////////////////////////
// Get mutable reference from the DataBox
template <typename... Tags>
template <typename Tag>
auto& DataBox<tmpl::list<Tags...>>::get_mutable_reference() {
  DEBUG_STATIC_ASSERT(
      not detail::has_no_matching_tag_v<tmpl::list<Tags...>, Tag>,
      "Found no tags in the DataBox that match the tag being retrieved.");
  DEBUG_STATIC_ASSERT(
      detail::has_unique_matching_tag_v<tmpl::list<Tags...>, Tag>,
      "Found more than one tag in the DataBox that matches the tag "
      "being retrieved. This happens because more than one tag with the same "
      "base (class) tag was added to the DataBox.");

  using item_tag = detail::first_matching_tag<tmpl::list<Tags...>, Tag>;

  DEBUG_STATIC_ASSERT(tmpl::list_contains_v<mutable_item_tags, item_tag>,
                      "Can only mutate mutable items");

  DEBUG_STATIC_ASSERT(
      not (... or
           tmpl::list_contains_v<typename Subitems<Tags>::type, item_tag>),
      "Cannot extract references to subitems");
  DEBUG_STATIC_ASSERT(not detail::has_subitems_v<item_tag>,
                      "Cannot extract references to items with subitems.");

  DEBUG_STATIC_ASSERT(
      tmpl::none<edge_list, std::is_same<tmpl::pin<item_tag>,
                                         tmpl::get_source<tmpl::_1>>>::value,
      "Cannot extract references to items used by compute items.");

  return get_item<item_tag>().mutate();
}

template <typename... Tags>
template <typename Consumer, typename Provider>
constexpr bool DataBox<tmpl::list<Tags...>>::tag_depends_on() {
  // We need to check for things depending on the passed tag, any
  // subitems, and the parent item if we were passed a subitem.  These
  // dependencies are handled internally by the mutation functions and
  // not encoded in the graph.
  using provider_aliases =
      tmpl::push_front<typename Subitems<Provider>::type,
                       creation_tag<Provider, DataBox<tmpl::list<Tags...>>>>;

  // We have to replace subitems with their parents here because
  // subitems of compute tags sometimes get graph edges from their
  // parents and sometimes do not, depending on if they have
  // dependencies.
  using consumer_tag_to_check =
      creation_tag<Consumer, DataBox<tmpl::list<Tags...>>>;

  // We cannot recursively call the function we are in, because we
  // need to avoid the normalization done above.  Otherwise we could
  // end up in a loop when destination of an edge normalizes to its
  // source.
  //
  // Lambdas cannot capture themselves, but they can take themselves
  // as an argument.
  auto check_dependents = [](auto&& recurse,
                             auto node_depending_on_provider_v) {
    using node_depending_on_provider = decltype(node_depending_on_provider_v);
    if (std::is_same_v<node_depending_on_provider, consumer_tag_to_check>) {
      return true;
    }

    using next_nodes_to_check = tmpl::transform<
        tmpl::filter<
            edge_list,
            tmpl::has_source<tmpl::_1, tmpl::pin<node_depending_on_provider>>>,
        tmpl::get_destination<tmpl::_1>>;

    return tmpl::as_pack<next_nodes_to_check>([&](auto... nodes) {
      return (... or recurse(recurse, tmpl::type_from<decltype(nodes)>{}));
    });
  };

  return tmpl::as_pack<provider_aliases>([&](auto... aliases) {
    return (... or check_dependents(check_dependents,
                                    tmpl::type_from<decltype(aliases)>{}));
  });
}

/*!
 * \ingroup DataBoxGroup
 * \brief Retrieve a mutable reference to the item with tag `Tag` from the
 * DataBox.
 *
 * The tag retrieved cannot be used by any compute tags, cannot have
 * subitems, and cannot itself be a subitem.  These requirements
 * prevent changes to the retrieved item from affecting any other tags
 * in the DataBox, so it can safely be modified without causing
 * internal inconsistencies.
 */
template <typename Tag, typename TagList>
SPECTRE_ALWAYS_INLINE auto& get_mutable_reference(
    const gsl::not_null<DataBox<TagList>*> box) {
  return box->template get_mutable_reference<Tag>();
}

/// @{
/*!
 * \ingroup DataBoxGroup
 * \brief Check whether the tag \p Consumer depends on the tag \p
 * Provider in a given \p Box.
 *
 * This is equivalent to the question of whether changing \p Provider
 * (through `db::mutate`, updating one of its dependencies, etc.)
 * could change the value of `db::get<Consumer>`.  Tags depend on
 * themselves, and an item and its subitems all depend on one another.
 */
template <typename Consumer, typename Provider, typename Box>
using tag_depends_on =
    std::bool_constant<Box::template tag_depends_on<Consumer, Provider>()>;

template <typename Consumer, typename Provider, typename Box>
constexpr bool tag_depends_on_v =
    tag_depends_on<Consumer, Provider, Box>::value;
/// @}

/*!
 * \ingroup DataBoxGroup
 * \brief List of Tags to add to the DataBox
 */
template <typename... Tags>
using AddSimpleTags = tmpl::flatten<tmpl::list<Tags...>>;

/*!
 * \ingroup DataBoxGroup
 * \brief List of Compute Item Tags to add to the DataBox
 */
template <typename... Tags>
using AddComputeTags = tmpl::flatten<tmpl::list<Tags...>>;

namespace detail {
template <class TagList>
struct compute_dbox_type {
  using immutable_item_tags = detail::expand_subitems<
      tmpl::filter<TagList, db::is_immutable_item_tag<tmpl::_1>>>;
  using mutable_item_tags = detail::expand_subitems<
      tmpl::filter<TagList, db::is_mutable_item_tag<tmpl::_1>>>;
  using type = DataBox<tmpl::append<mutable_item_tags, immutable_item_tags>>;
};
}  // namespace detail

/*!
 * \ingroup DataBoxGroup
 * \brief Returns the type of the DataBox that would be constructed from the
 * `TagList` of tags.
 */
template <typename TagList>
using compute_databox_type = typename detail::compute_dbox_type<TagList>::type;

/*!
 * \ingroup DataBoxGroup
 * \brief Create a new DataBox
 *
 * \details
 * Creates a new DataBox holding types Tags::type filled with the arguments
 * passed to the function. Compute and reference items must be added so that
 * their dependencies are added before themselves. For example, say you have
 * compute items `A` and `B` where `B` depends on `A`, then you must add them
 * using `db::AddImmutableItemTags<A, B>`.
 *
 * \example
 * \snippet Test_DataBox.cpp create_databox
 *
 * \see create_from
 *
 * \tparam AddMutableItemTags the tags of the mutable items that are being added
 * \tparam AddImmutableItemTags list of \ref ComputeTag "compute item tags" and
 *         \ref ReferenceTag "refernce item tags" to add to the DataBox
 * \param args the initial values for the mutable items to add to the DataBox
 */
template <typename AddMutableItemTags,
          typename AddImmutableItemTags = tmpl::list<>, typename... Args>
SPECTRE_ALWAYS_INLINE constexpr auto create(Args&&... args) {
  static_assert(tt::is_a_v<tmpl::list, AddImmutableItemTags>,
                "AddImmutableItemTags must be a tmpl::list");
  static_assert(tt::is_a_v<tmpl::list, AddMutableItemTags>,
                "AddMutableItemTags must be a tmpl::list");
  static_assert(
      tmpl::all<AddMutableItemTags, is_non_base_tag<tmpl::_1>>::value and
          tmpl::all<AddImmutableItemTags, is_non_base_tag<tmpl::_1>>::value,
      "Can only add tags derived from db::SimpleTag.");
  static_assert(
      tmpl::all<AddMutableItemTags, is_mutable_item_tag<tmpl::_1>>::value,
      "Cannot add any ComputeTags or ReferenceTags in the AddMutableTags list, "
      "must use the AddImmutableItemTags list.");
  static_assert(
      tmpl::all<AddImmutableItemTags, is_immutable_item_tag<tmpl::_1>>::value,
      "Cannot add any SimpleTags in the AddImmutableItemTags list, must use "
      "the "
      "AddMutableItemTags list.");

  using mutable_item_tags = detail::expand_subitems<AddMutableItemTags>;
  using immutable_item_tags = detail::expand_subitems<AddImmutableItemTags>;

  return db::DataBox<tmpl::append<mutable_item_tags, immutable_item_tags>>(
      AddMutableItemTags{}, AddImmutableItemTags{},
      std::forward<Args>(args)...);
}

namespace detail {
template <typename DataBoxTags, typename... TagsToRetrieve>
constexpr bool check_tags_are_in_databox(
    DataBoxTags /*meta*/, tmpl::list<TagsToRetrieve...> /*meta*/) {
  static_assert(
      (tag_is_retrievable_v<TagsToRetrieve, DataBox<DataBoxTags>> and ...),
      "A desired tag is not in the DataBox.  See the first template "
      "argument of tag_is_retrievable_v for the missing tag, and the "
      "second for the available tags.");
  return true;
}

template <typename... ArgumentTags, typename F, typename BoxTags,
          typename... Args>
static constexpr auto apply(F&& f, const DataBox<BoxTags>& box,
                            tmpl::list<ArgumentTags...> /*meta*/,
                            Args&&... args) {
  if constexpr (is_apply_callable_v<
                    F,
                    tmpl::conditional_t<
                        std::is_same_v<ArgumentTags, ::Tags::DataBox>,
                        const DataBox<BoxTags>&,
                        const_item_type<ArgumentTags, BoxTags>>...,
                    Args...>) {
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

/// @{
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
                                           Args&&... args) {
  detail::check_tags_are_in_databox(
      BoxTags{}, tmpl::remove<ArgumentTags, ::Tags::DataBox>{});
  return detail::apply(std::forward<F>(f), box, ArgumentTags{},
                       std::forward<Args>(args)...);
}

template <typename F, typename BoxTags, typename... Args>
SPECTRE_ALWAYS_INLINE constexpr auto apply(F&& f, const DataBox<BoxTags>& box,
                                           Args&&... args) {
  return apply<typename std::decay_t<F>::argument_tags>(
      std::forward<F>(f), box, std::forward<Args>(args)...);
}

template <typename F, typename BoxTags, typename... Args>
SPECTRE_ALWAYS_INLINE constexpr auto apply(const DataBox<BoxTags>& box,
                                           Args&&... args) {
  return apply(F{}, box, std::forward<Args>(args)...);
}
/// @}

namespace detail {
template <typename Tag, typename BoxTags>
using tag_return_type =
    tmpl::conditional_t<std::is_same_v<Tag, ::Tags::DataBox>,
                        db::DataBox<BoxTags>, typename Tag::type>;

template <typename... ReturnTags, typename... ArgumentTags, typename F,
          typename BoxTags, typename... Args>
SPECTRE_ALWAYS_INLINE constexpr decltype(auto) mutate_apply(
    F&& f, const gsl::not_null<db::DataBox<BoxTags>*> box,
    tmpl::list<ReturnTags...> /*meta*/, tmpl::list<ArgumentTags...> /*meta*/,
    Args&&... args) {
  if constexpr (sizeof...(ReturnTags) == 0) {
    return apply<tmpl::list<ArgumentTags...>>(std::forward<F>(f), *box,
                                              std::forward<Args>(args)...);
  } else {
    detail::check_tags_are_in_databox(BoxTags{}, tmpl::list<ReturnTags...>{});
    detail::check_tags_are_in_databox(BoxTags{}, tmpl::list<ArgumentTags...>{});
    static_assert(not(... or std::is_same_v<ArgumentTags, Tags::DataBox>),
                  "Cannot pass Tags::DataBox to mutate_apply when mutating "
                  "since the db::get won't work inside mutate_apply.");
    if constexpr (is_apply_callable_v<
                      F,
                      const gsl::not_null<
                          tag_return_type<ReturnTags, BoxTags>*>...,
                      const_item_type<ArgumentTags, BoxTags>..., Args...>) {
      return ::db::mutate<ReturnTags...>(
          [](const gsl::not_null<
                 tag_return_type<ReturnTags, BoxTags>*>... mutated_items,
             const_item_type<ArgumentTags, BoxTags>... args_items,
             decltype(std::forward<Args>(args))... l_args) {
            return std::decay_t<F>::apply(mutated_items..., args_items...,
                                          std::forward<Args>(l_args)...);
          },
          box, db::get<ArgumentTags>(*box)..., std::forward<Args>(args)...);
    } else if constexpr (
        ::tt::is_callable_v<
            F, const gsl::not_null<tag_return_type<ReturnTags, BoxTags>*>...,
            const_item_type<ArgumentTags, BoxTags>..., Args...>) {
      return ::db::mutate<ReturnTags...>(f, box, db::get<ArgumentTags>(*box)...,
                                         std::forward<Args>(args)...);
    } else {
      error_function_not_callable<
          F, gsl::not_null<tag_return_type<ReturnTags, BoxTags>*>...,
          const_item_type<ArgumentTags, BoxTags>..., Args...>();
    }
  }
}
}  // namespace detail

/// @{
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
 * Any return values of the invokable `f` are forwarded as returns to the
 * `mutate_apply` call.
 *
 * \note If `MutateTags` is empty this will forward to `db::apply`, so
 * `::Tags::DataBox` will be available.  Otherwise it will call
 * `db::mutate`.  See those functions for more details on retrieving
 * the entire box.
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
SPECTRE_ALWAYS_INLINE constexpr decltype(auto) mutate_apply(
    F&& f, const gsl::not_null<DataBox<BoxTags>*> box, Args&&... args) {
  return detail::mutate_apply(std::forward<F>(f), box, MutateTags{},
                              ArgumentTags{}, std::forward<Args>(args)...);
}

template <typename F, typename BoxTags, typename... Args>
SPECTRE_ALWAYS_INLINE constexpr decltype(auto) mutate_apply(
    F&& f, const gsl::not_null<DataBox<BoxTags>*> box, Args&&... args) {
  return mutate_apply<typename std::decay_t<F>::return_tags,
                      typename std::decay_t<F>::argument_tags>(
      std::forward<F>(f), box, std::forward<Args>(args)...);
}

template <typename F, typename BoxTags, typename... Args>
SPECTRE_ALWAYS_INLINE constexpr decltype(auto) mutate_apply(
    const gsl::not_null<DataBox<BoxTags>*> box, Args&&... args) {
  return mutate_apply(F{}, box, std::forward<Args>(args)...);
}
/// @}
}  // namespace db

template <typename TagsList>
std::ostream& operator<<(std::ostream& os, const db::DataBox<TagsList>& box) {
  os << box.print_types() << "\n";
  os << box.print_items() << "\n";
  return os;
}
