// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines classes and functions used for manipulating DataBox's

#pragma once

#include <algorithm>
#include <cassert>
#include <functional>
#include <type_traits>
#include <unordered_map>

#include "DataStructures/DataBoxTag.hpp"
#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/Deferred.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/*!
 * \ingroup DataBoxGroup
 * \brief Namespace for DataBox related things
 */
namespace db {

/*!
 * \ingroup DataBoxGroup
 * \brief Compute the canonical typelist used in the DataBox
 * \note The result is architecture-dependent.
 *
 * \requires `tt::is_a<typelist, Ls>::``value` is true
 * \metareturns `typelist` of all elements in `Ls` but ordered for the
 * db::DataBox
 */
template <typename Ls>
using get_databox_list =
    tmpl::sort<Ls, db::detail::databox_tag_less<tmpl::_1, tmpl::_2>>;

// Forward declarations
/// \cond
template <typename TagsLs>
class DataBox;
/// \endcond

namespace detail {
template <typename PoppedTagLs, typename FullTagLs>
struct DataBoxAddHelper;

template <typename DependencyGraph, typename Vertices>
struct ResetComputeItems;
}  // namespace detail

// @{
/*!
 * \ingroup TypeTraits DataBox
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

namespace detail {
/*!
 * \ingroup DataBoxGroup
 * \brief TaggedTuple of objects or lazily evaluated functions
 * \tparam Tags the tags of the objects to be placed in the tuple
 */
template <typename... Tags>
struct TaggedDeferredTuple {
  static_assert(tmpl::is_set<Tags...>::value,
                "Cannot have repeated Tags in a DataBox.");
  template <typename... Args>
  constexpr explicit TaggedDeferredTuple(Args&&... args)
      : data_(std::forward<Args>(args)...) {}
  static constexpr size_t size = sizeof...(Tags);

  // @{
  /*!
   * \ingroup DataBoxGroup
   * \brief Retrieve on object from a TaggedDeferredTuple
   *
   * \tparam T the tag of the object to retrieve
   * \return reference to the object held in the tuple
   */
  template <typename T>
  constexpr Deferred<item_type<T>>& get() noexcept {
    static_assert(not std::is_same<tmpl::index_of<tmpl::list<Tags...>, T>,
                                   tmpl::no_such_type_>::value,
                  "Could not retrieve Tag from DataBox. See the first template "
                  "parameter of the instantiation for what Tag is being "
                  "retrieved and the remaining template parameters for what "
                  "takes are available.");
    return std::get<tmpl::index_of<tmpl::list<Tags...>, T>::value>(data_);
  }

  template <typename T>
  constexpr const Deferred<item_type<T>>& get() const noexcept {
    static_assert(not std::is_same<tmpl::index_of<tmpl::list<Tags...>, T>,
                                   tmpl::no_such_type_>::value,
                  "Could not retrieve Tag from DataBox. See the first template "
                  "parameter of the instantiation for what Tag is being "
                  "retrieved and the remaining template parameters for what "
                  "takes are available.");
    return std::get<tmpl::index_of<tmpl::list<Tags...>, T>::value>(data_);
  }
  // @}

 private:
  std::tuple<Deferred<item_type<Tags>>...> data_;
};

template <typename Element, typename = void>
struct extract_dependent_items {
  using type = typelist<Element>;
};

template <typename Element>
struct extract_dependent_items<
    Element, std::enable_if_t<tt::is_a<Variables, item_type<Element>>::value>> {
  using type =
      tmpl::append<typelist<Element>, typename item_type<Element>::tags_list>;
};

template <typename Caller, typename Callee, typename List, typename = void>
struct create_dependency_graph {
  using new_edge = tmpl::edge<Callee, Caller>;
  using type = tmpl::conditional_t<
      tmpl::found<List, std::is_same<tmpl::_1, tmpl::pin<new_edge>>>::value,
      List, tmpl::push_back<List, new_edge>>;
};

template <typename Caller, typename Callee, typename List>
struct create_dependency_graph<
    Caller, Callee, List,
    std::enable_if_t<is_simple_compute_item<Callee>::value>> {
  using sub_tree =
      tmpl::fold<typename Callee::argument_tags, List,
                 create_dependency_graph<Callee, tmpl::_element, tmpl::_state>>;
  using type = tmpl::conditional_t<
      std::is_same<void, Caller>::value, sub_tree,
      tmpl::push_back<sub_tree, tmpl::edge<Callee, Caller>>>;
};

template <typename Caller, typename Callee, typename List>
struct create_dependency_graph<
    Caller, Callee, List,
    std::enable_if_t<is_variables_compute_item<Callee>::value>> {
  using partial_sub_tree = tmpl::fold<
      typename Callee::argument_tags, List,
      create_dependency_graph<tmpl::pin<Callee>, tmpl::_element, tmpl::_state>>;
  using variables_tags_dependency = tmpl::fold<
      typename item_type<Callee>::tags_list, tmpl::list<>,
      tmpl::bind<tmpl::push_back, tmpl::_state,
                 tmpl::bind<tmpl::edge, tmpl::pin<Callee>, tmpl::_element>>>;
  using sub_tree = tmpl::append<partial_sub_tree, variables_tags_dependency>;
  using type = tmpl::conditional_t<
      std::is_same<void, Caller>::value, sub_tree,
      tmpl::push_back<sub_tree, tmpl::edge<Callee, Caller>>>;
};
}  // namespace detail

/*!
 * \ingroup DataBoxGroup
 * \brief A DataBox stores objects that can be retrieved by using Tags
 * \warning
 * The order of the tags in DataBoxes returned by create and create_from depends
 * on implementation-defined behavior, and therefore should not be
 * specified in source files. If explicitly naming a DataBox type is
 * necessary they should be generated using get_databox_list.
 *
 * @tparam TagsLs a metasequence
 * @tparam Tags list of DataBoxTag's
 */
template <template <typename...> class TagsLs, typename... Tags>
class DataBox<TagsLs<Tags...>> {
  static_assert(
      cpp17::conjunction<std::is_base_of<db::DataBoxTag, Tags>...>::value,
      "All structs used to Tag (compute) items in a DataBox must derive off of "
      "db::DataBoxTag");
  static_assert(cpp17::conjunction<detail::tag_has_label<Tags>...>::value,
                "Missing a label on a Tag");
  static_assert(
      cpp17::conjunction<detail::tag_label_correct_type<Tags>...>::value,
      "One of the labels of the Tags in a DataBox has the incorrect "
      "type. It should be a DataBoxString_t.");

 public:
  /*!
   * \brief A ::typelist of Tags that the DataBox holds
   */
  using tags_list = typelist<Tags...>;

  /// \cond HIDDEN_SYMBOLS
  /*!
   * \note the default constructor is only used for serialization
   */
  DataBox() = default;
  DataBox(DataBox&& rhs) noexcept = default;
  DataBox& operator=(DataBox&& rhs) noexcept = default;
  DataBox(const DataBox& rhs) = default;
  DataBox& operator=(const DataBox& rhs) = default;
  ~DataBox() = default;
  /// \endcond

  /// @cond HIDDEN_SYMBOLS
  /*!
   * \brief Helper function called by db::create to call the constructor
   *
   * \requires `tt::is_a<::typelist, AddTags>::value` is true,
   * `tt::is_a<::typelist, AddComputeItems>::value` is true,
   * `tmpl::all<AddComputeItems, is_compute_item<tmpl::_1>>::value` is true, and
   * `std::conjunction<std::is_same<item_type<AddTags>, Args>...>::value` is
   * true
   *
   * \return A DataBox with items described by Tags in `AddTags` and
   * values `args...`, and compute items described by Tags in `AddComputeItems`
   */
  template <typename AddTags, typename AddComputeItems, typename... Args>
  static constexpr auto create(Args&&... args);

  /*!
   * \brief Helper function called by db::create_from to call the constructor
   *
   * \requires `tt::is_a<::typelist, AddTags>::value` is true,
   * `tt::is_a<::typelist, AddComputeItems>::value` is true,
   * `tmpl::all<AddComputeItems, is_compute_item<tmpl::_1>>::value` is true,
   * `std::conjunction<std::is_same<item_type<AddTags>, Args>...>::value` is
   * true and `tt::is_a<DataBox, Box>::value` is true
   *
   * \return A DataBox with items described by Tags in `AddTags` and
   * values `args...`, (compute) items in `Box` that are not in the
   * `RemoveTags`, and compute items described by Tags in `AddComputeItems`
   */
  template <typename RemoveTags, typename AddTags, typename AddComputeItems,
            typename Box, typename... Args>
  static constexpr auto create_from(const Box& box, Args&&... args);
  /// @endcond

  /*!
   * \requires Type `T` is one of the Tags corresponding to an object stored in
   * the DataBox
   *
   * \return The object corresponding to the Tag `T`
   */
  template <typename T>
  constexpr const item_type<T>& get() const noexcept {
    return data_.template get<T>().get();
  }

  /// \cond HIDDEN_SYMBOLS
  /*!
   * \requires Type `T` is one of the Tags corresponding to an object stored in
   * the DataBox
   *
   * \note This should not be used outside of implementation details
   *
   * @return The lazy object corresponding to the Tag `T`
   */
  template <typename T>
  constexpr const Deferred<item_type<T>>& get_lazy() const noexcept {
    return data_.template get<T>();
  }
  /// \endcond

  /*!
   * \requires Type `T` is one of the Tags corresponding to an object stored in
   * the DataBox
   *
   * `mutate()` is similar to get, however it allows altering the value of the
   * item in the DataBox.
   * \return The object corresponding to the Tag `T`
   */
  template <typename T>
  constexpr item_type<T>& mutate() noexcept {
    static_assert(not db::is_compute_item_v<T>, "Cannot mutate a compute item");
    return data_.template get<T>().mutate();
  }

 private:
  template <typename... TagsInArgsOrder, typename... FullItems,
            typename... ComputeItemTags, typename... FullComputeItems,
            typename... Args,
            std::enable_if_t<not cpp17::disjunction<
                db::is_databox<std::decay_t<Args>>...>::value>* = nullptr>
  constexpr DataBox(typelist<TagsInArgsOrder...> /*meta*/,
                    typelist<FullItems...> /*meta*/,
                    typelist<ComputeItemTags...> /*meta*/,
                    typelist<FullComputeItems...> /*meta*/, Args&&... args);

  template <typename OldTags, typename... KeepTags, typename... NewTags,
            typename... NewComputeItems, typename ComputeItemsToKeep,
            typename... Args>
  constexpr DataBox(const DataBox<OldTags>& old_box,
                    typelist<KeepTags...> /*meta*/,
                    typelist<NewTags...> /*meta*/,
                    typelist<NewComputeItems...> /*meta*/,
                    ComputeItemsToKeep /*meta*/, Args&&... args);

  SPECTRE_ALWAYS_INLINE void check_tags() const {
#ifdef SPECTRE_DEBUG
    ASSERT(tmpl::size<tags_list>::value == 0 or
               tmpl::for_each<tags_list>(detail::check_tag_labels{}).value,
           "Could not match one of the Tag labels with the Tag type. That is, "
           "the label of a Tag must be the same as the Tag.");
#endif
  }

  detail::TaggedDeferredTuple<Tags...> data_;
};

/// \cond HIDDEN_SYMBOLS
template <template <typename...> class TagsLs, typename... Tags>
template <typename... TagsInArgsOrder, typename... FullItems,
          typename... ComputeItemTags, typename... FullComputeItems,
          typename... Args, std::enable_if_t<not cpp17::disjunction<
                                is_databox<std::decay_t<Args>>...>::value>*>
constexpr DataBox<TagsLs<Tags...>>::DataBox(
    typelist<TagsInArgsOrder...> /*meta*/, typelist<FullItems...> /*meta*/,
    typelist<ComputeItemTags...> /*meta*/,
    typelist<FullComputeItems...> /*meta*/, Args&&... args) {
  check_tags();
  static_assert(
      sizeof...(Tags) == sizeof...(FullItems) + sizeof...(FullComputeItems),
      "Must pass in as many (compute) items as there are Tags.");
  static_assert(sizeof...(TagsInArgsOrder) == sizeof...(Args),
                "Must pass in as many arguments as AddTags");
  static_assert(
      cpp17::conjunction<std::is_same<typename TagsInArgsOrder::type,
                                      std::decay_t<Args>>...>::value,
      "The type of each Tag must be the same as the type being passed into "
      "the function creating the new DataBox.");
  std::tuple<Args...> args_tuple(std::forward<Args>(args)...);
  using full_tags_list =
      tmpl::append<typelist<TagsInArgsOrder...>, typelist<ComputeItemTags...>>;
  using complete_tags_list =
      tmpl::append<typelist<FullItems...>, typelist<FullComputeItems...>>;
  detail::DataBoxAddHelper<full_tags_list, complete_tags_list>::
      template add_items_to_box<0>(args_tuple, data_);
}

template <template <typename...> class TagsLs, typename... Tags>
template <typename OldTags, typename... KeepTags, typename... NewTags,
          typename... NewComputeItems, typename ComputeItemsToKeep,
          typename... Args>
constexpr DataBox<TagsLs<Tags...>>::DataBox(
    const DataBox<OldTags>& old_box, typelist<KeepTags...> /*meta*/,
    typelist<NewTags...> /*meta*/, typelist<NewComputeItems...> /*meta*/,
    ComputeItemsToKeep /*meta*/, Args&&... args) {
  static_assert(sizeof...(NewTags) == sizeof...(Args),
                "Must pass in as many arguments as AddTags");
  static_assert(
      cpp17::conjunction<
          std::is_same<typename NewTags::type, std::decay_t<Args>>...>::value,
      "The type of each Tag must be the same as the type being passed into "
      "the function creating the new DataBox.");
  // Create dependency graph between compute items and items being reset
  using edge_list = tmpl::fold<
      ComputeItemsToKeep, typelist<>,
      detail::create_dependency_graph<void, tmpl::_element, tmpl::_state>>;
  using DependencyGraph = tmpl::digraph<edge_list>;

  check_tags();
  // Merge old tags, including all ComputeItems even though they might be
  // reset.
  detail::DataBoxAddHelper<typelist<KeepTags...>, typelist<KeepTags...>>::
      template merge_old_box(old_box, data_);

  std::tuple<Args...> args_tuple(std::forward<Args>(args)...);
  detail::DataBoxAddHelper<typelist<NewTags...>, typelist<NewTags...>>::
      template add_items_to_box<0>(args_tuple, data_);

  detail::ResetComputeItems<DependencyGraph, ComputeItemsToKeep>::apply(data_);

  // Add new compute items
  detail::DataBoxAddHelper<
      typelist<NewComputeItems...>,
      typelist<KeepTags..., NewTags..., NewComputeItems...>>::
      template add_items_to_box<0>(args_tuple, data_);
}

template <template <typename...> class TagsLs, typename... Tags>
template <typename AddTags, typename AddComputeItems, typename... Args>
constexpr auto DataBox<TagsLs<Tags...>>::create(Args&&... args) {
  static_assert(tt::is_a<tmpl::list, AddComputeItems>::value,
                "AddComputeItems must by a typelist");
  static_assert(tt::is_a<tmpl::list, AddTags>::value,
                "AddTags must by a typelist");
  static_assert(
      not tmpl::any<AddTags, is_compute_item<tmpl::_1>>::value,
      "Cannot add any ComputeItemTag in the AddTags list, must use the "
      "AddComputeItems list.");
  static_assert(tmpl::all<AddComputeItems, is_compute_item<tmpl::_1>>::value,
                "Cannot add any Tags in the AddComputeItems list, must use the "
                "AddTags list.");
  using full_items =
      tmpl::fold<AddTags, tmpl::list<>,
                 tmpl::bind<tmpl::append, tmpl::_state,
                            detail::extract_dependent_items<tmpl::_element>>>;
  using full_compute_items =
      tmpl::fold<AddComputeItems, tmpl::list<>,
                 tmpl::bind<tmpl::append, tmpl::_state,
                            detail::extract_dependent_items<tmpl::_element>>>;
  using sorted_tags =
      ::db::get_databox_list<tmpl::append<full_items, full_compute_items>>;
  return DataBox<sorted_tags>(AddTags{}, full_items{}, AddComputeItems{},
                              full_compute_items{},
                              std::forward<Args>(args)...);
}

template <template <typename...> class TagsLs, typename... Tags>
template <typename RemoveTags, typename AddTags, typename AddComputeItems,
          typename Box, typename... Args>
constexpr auto DataBox<TagsLs<Tags...>>::create_from(const Box& box,
                                                     Args&&... args) {
  static_assert(tt::is_a_v<::db::DataBox, Box>,
                "create_from must receive a DataBox as its first argument");
  static_assert(tt::is_a_v<tmpl::list, RemoveTags>,
                "RemoveTags must by a typelist");
  static_assert(tt::is_a_v<tmpl::list, AddTags>, "AddTags must by a typelist");
  static_assert(tt::is_a_v<tmpl::list, AddComputeItems>,
                "AddComputeItems must by a typelist");
  static_assert(
      tmpl::all<AddComputeItems, is_compute_item<tmpl::_1>>::value,
      "Cannot add any ComputeItemTag in the AddTags list, must use the "
      "AddComputeItems list.");
  using old_tags_list = typename Box::tags_list;

  // Build list of compute items in Box::tags_list that are not in RemoveTags
  using compute_items_to_keep = tmpl::filter<
      old_tags_list,
      tmpl::and_<
          db::is_compute_item<tmpl::_1>,
          tmpl::not_<tmpl::bind<tmpl::found, tmpl::pin<RemoveTags>,
                                tmpl::bind<std::is_same, tmpl::parent<tmpl::_1>,
                                           tmpl::bind<tmpl::pin, tmpl::_1>>>>>>;

  // Build list of tags where we expand the tags inside Variables<Tags...>
  // objects. This is needed since we actually want those tags to be part of the
  // DataBox type as well
  using full_remove_tags =
      tmpl::fold<RemoveTags, tmpl::list<>,
                 tmpl::bind<tmpl::append, tmpl::_state,
                            detail::extract_dependent_items<tmpl::_element>>>;
  using full_items =
      tmpl::fold<AddTags, tmpl::list<>,
                 tmpl::bind<tmpl::append, tmpl::_state,
                            detail::extract_dependent_items<tmpl::_element>>>;
  using full_compute_items =
      tmpl::fold<AddComputeItems, tmpl::list<>,
                 tmpl::bind<tmpl::append, tmpl::_state,
                            detail::extract_dependent_items<tmpl::_element>>>;
  using remaining_tags =
      tmpl::fold<full_remove_tags, old_tags_list,
                 tmpl::bind<tmpl::remove, tmpl::_state, tmpl::_element>>;
  using new_tags = tmpl::append<remaining_tags, full_items, full_compute_items>;
  using sorted_tags = ::db::get_databox_list<new_tags>;
  return DataBox<sorted_tags>(box, remaining_tags{}, AddTags{},
                              AddComputeItems{}, compute_items_to_keep{},
                              std::forward<Args>(args)...);
}
/// \endcond

namespace detail {
template <typename PoppedTagLs, typename FullTagLs>
struct DataBoxAddHelper {
  template <typename ComputeItem, typename... Tags, typename... ComputeItemTags>
  SPECTRE_ALWAYS_INLINE static constexpr void add_compute_item_to_box(
      detail::TaggedDeferredTuple<Tags...>& data,
      tmpl::list<ComputeItemTags...> /*typelist*/) {
    static_assert(
        cpp17::conjunction<
            std::is_base_of<db::DataBoxTag, ComputeItemTags>...>::value,
        "Cannot have non-DataBoxTag arguments to a ComputeItem. Please make "
        "sure all the specified argument_tags in the ComputeItem derive from "
        "db::DataBoxTag.");
    using index = tmpl::index_of<FullTagLs, ComputeItem>;
    static_assert(not cpp17::disjunction<
                      std::is_same<ComputeItemTags, ComputeItem>...>::value,
                  "A ComputeItem cannot take its own Tag as an argument.");
    static_assert(
        cpp17::conjunction<tmpl::less<
            tmpl::index_of<FullTagLs, ComputeItemTags>, index>...>::value,
        "The dependencies of a ComputeItem must be added before the "
        "ComputeItem itself. This is done to ensure no cyclic "
        "dependencies arise.");

    data.template get<ComputeItem>() = make_deferred(
        ComputeItem::function, data.template get<ComputeItemTags>()...);
  }

  template <int ArgsIndex, typename... T, typename... Tags,
            typename ItemsLs = PoppedTagLs,
            std::enable_if_t<
                is_simple_compute_item<tmpl::front<ItemsLs>>::value>* = nullptr>
  SPECTRE_ALWAYS_INLINE static constexpr void add_items_to_box(
      std::tuple<T...>& tuple, detail::TaggedDeferredTuple<Tags...>& data) {
    using ls_pop = tmpl::pop_front<PoppedTagLs>;
    using compute_item = tmpl::front<PoppedTagLs>;
    add_compute_item_to_box<compute_item>(
        data, typename compute_item::argument_tags{});
    DataBoxAddHelper<ls_pop,
                     FullTagLs>::template add_items_to_box<ArgsIndex + 1>(tuple,
                                                                          data);
  }

  template <typename VariablesTag, typename... Tags>
  SPECTRE_ALWAYS_INLINE static constexpr void add_variables_compute_tags_to_box(
      detail::TaggedDeferredTuple<Tags...>& /*data*/, typelist<> /*unused*/) {}

  template <typename VariablesTag, typename T, typename... Ts, typename... Tags>
  SPECTRE_ALWAYS_INLINE static void add_variables_compute_tags_to_box(
      detail::TaggedDeferredTuple<Tags...>& data,
      typelist<T, Ts...> /*unused*/) {
    data.template get<T>() =
        make_deferred([lazy_function = data.template get<VariablesTag>()]() {
          return lazy_function.get().template get<T>();
        });
    add_variables_compute_tags_to_box<VariablesTag>(data, typelist<Ts...>{});
  }

  // add_variables_tags_to_box is used to add the Tensor's inside a Variables
  // object into the DataBox. If the Tag does not hold a Variables then we
  // have nothing to add, otherwise we recurse through the Tensors we need
  // added. Currently this uses the same infrastructure as the ComputeItems that
  // return variables, but it would be more efficient to not deal with it that
  // way and just add them directly. However, then there could be issues with
  // dependencies that will need to be checked carefully.
  template <typename Tag, typename... Tags,
            typename std::enable_if_t<
                not tt::is_a<Variables, typename Tag::type>::value>* = nullptr>
  SPECTRE_ALWAYS_INLINE static void add_variables_tags_to_box_helper(
      detail::TaggedDeferredTuple<Tags...>& /*data*/) {}

  template <typename Tag, typename... Tags,
            typename std::enable_if_t<
                tt::is_a<Variables, item_type<Tag>>::value>* = nullptr>
  SPECTRE_ALWAYS_INLINE static constexpr void add_variables_tags_to_box_helper(
      detail::TaggedDeferredTuple<Tags...>& data) {
    add_variables_compute_tags_to_box<Tag>(
        data, typename item_type<Tag>::tags_list{});
  }

  template <int ArgsIndex, typename... T, typename... Tags,
            typename ItemsLs = PoppedTagLs,
            std::enable_if_t<is_variables_compute_item<
                tmpl::front<ItemsLs>>::value>* = nullptr>
  SPECTRE_ALWAYS_INLINE static constexpr void add_items_to_box(
      std::tuple<T...>& tuple, detail::TaggedDeferredTuple<Tags...>& data) {
    using ls_pop = tmpl::pop_front<PoppedTagLs>;
    using compute_item = tmpl::front<PoppedTagLs>;
    add_compute_item_to_box<compute_item>(
        data, typename compute_item::argument_tags{});
    add_variables_compute_tags_to_box<compute_item>(
        data, typename item_type<compute_item>::tags_list{});
    DataBoxAddHelper<ls_pop,
                     FullTagLs>::template add_items_to_box<ArgsIndex + 1>(tuple,
                                                                          data);
  }

  // Add a tag that isn't a compute item
  template <int ArgsIndex, typename... T, typename... Tags,
            typename ItemsLs = PoppedTagLs,
            std::enable_if_t<
                not is_compute_item<tmpl::front<ItemsLs>>::value>* = nullptr>
  SPECTRE_ALWAYS_INLINE static constexpr void add_items_to_box(
      std::tuple<T...>& tuple, detail::TaggedDeferredTuple<Tags...>& data) {
    using ls_pop = tmpl::pop_front<PoppedTagLs>;
    using tag = tmpl::front<PoppedTagLs>;
    static_assert(not tt::is_a<Deferred, tmpl::front<PoppedTagLs>>::value,
                  "Cannot pass a Deferred into the DataBox as an Item. This "
                  "functionally can trivially be added, however it is "
                  "intentionally omitted because users of DataBox are not "
                  "supposed to deal with Deferred.");
    data.template get<tag>() =
        Deferred<item_type<tag>>(std::move(std::get<ArgsIndex>(tuple)));
    // If `tag` holds a Variables then add the contained Tensor's
    add_variables_tags_to_box_helper<tag>(data);
    DataBoxAddHelper<ls_pop,
                     FullTagLs>::template add_items_to_box<ArgsIndex + 1>(tuple,
                                                                          data);
  }

  template <typename OldTags, typename... Tags>
  SPECTRE_ALWAYS_INLINE static constexpr void merge_old_box(
      const DataBox<OldTags>& old_box,
      detail::TaggedDeferredTuple<Tags...>& data) {
    using tag = tmpl::front<PoppedTagLs>;
    data.template get<tag>() = old_box.template get_lazy<tag>();
    DataBoxAddHelper<tmpl::pop_front<PoppedTagLs>, FullTagLs>::merge_old_box(
        old_box, data);
  }
};

// Base case used to end recursion
template <template <typename...> class TagLs, typename FullTagLs>
struct DataBoxAddHelper<TagLs<>, FullTagLs> {
  template <typename OldTags, typename... Tags>
  SPECTRE_ALWAYS_INLINE static constexpr void merge_old_box(
      const DataBox<OldTags>& /*old_box*/,
      detail::TaggedDeferredTuple<Tags...>& /*data*/) {}

  template <int ArgsIndex, typename... T, typename... Tags>
  SPECTRE_ALWAYS_INLINE static constexpr void add_items_to_box(
      std::tuple<T...>& /*tuple*/,
      detail::TaggedDeferredTuple<Tags...>& /*data*/) {}
};

template <typename DependencyGraph, typename Vertices>
struct ResetComputeItems {
  template <typename ComputeItem, typename... Tags, typename... ComputeItemTags>
  SPECTRE_ALWAYS_INLINE static constexpr void add_compute_item_to_box(
      detail::TaggedDeferredTuple<Tags...>& data,
      tmpl::list<ComputeItemTags...> /*meta*/) {
    data.template get<ComputeItem>() = make_deferred(
        ComputeItem::function, data.template get<ComputeItemTags>()...);
  }

  template <
      typename Tag, typename... Tags,
      typename std::enable_if_t<not is_compute_item<Tag>::value>* = nullptr>
  SPECTRE_ALWAYS_INLINE static constexpr void reset_compute_item(
      detail::TaggedDeferredTuple<Tags...>& /*data*/) {}

  template <
      typename Tag, typename... Tags,
      typename std::enable_if_t<is_simple_compute_item<Tag>::value>* = nullptr>
  SPECTRE_ALWAYS_INLINE static constexpr void reset_compute_item(
      detail::TaggedDeferredTuple<Tags...>& data) {
    add_compute_item_to_box<Tag>(data, typename Tag::argument_tags{});
  }
  template <typename Tag, typename... Tags,
            typename std::enable_if_t<is_variables_compute_item<Tag>::value>* =
                nullptr>
  SPECTRE_ALWAYS_INLINE static constexpr void reset_compute_item(
      detail::TaggedDeferredTuple<Tags...>& data) {
    add_compute_item_to_box<Tag>(data, typename Tag::argument_tags{});
    DataBoxAddHelper<typelist<int>, typelist<int>>::
        add_variables_compute_tags_to_box<Tag>(
            data, typename item_type<Tag>::tags_list{});
  }

  template <typename... Tags>
  static constexpr void apply(detail::TaggedDeferredTuple<Tags...>& data) {
    using current_vertex = tmpl::front<Vertices>;
    // Reset me first
    reset_compute_item<current_vertex>(data);
    // Reset what depends on me next
    using outgoing_edges = tmpl::branch_if_t<
        tmpl::size<typename current_vertex::argument_tags>::value == 0,
        tmpl::list<>,
        tmpl::bind<tmpl::outgoing_edges, DependencyGraph, current_vertex>>;
    using next_vertices =
        tmpl::transform<outgoing_edges, tmpl::get_destination<tmpl::_1>>;
    ResetComputeItems<DependencyGraph, next_vertices>::apply(data);
    // Reset next item in current list (level)
    ResetComputeItems<DependencyGraph, tmpl::pop_front<Vertices>>::apply(data);
  }
};

template <typename DependencyGraph, template <typename...> class Vertices>
struct ResetComputeItems<DependencyGraph, Vertices<>> {
  template <typename... Tags>
  SPECTRE_ALWAYS_INLINE static constexpr void apply(
      detail::TaggedDeferredTuple<Tags...>& /*data*/) {}
};
}  // namespace detail

/*!
 * \ingroup DataBoxGroup
 * \brief List of Tags to remove from the DataBox
 */
template <typename... Tags>
using RemoveTags = tmpl::flatten<typelist<Tags...>>;

/*!
 * \ingroup DataBoxGroup
 * \brief List of Tags to add to the DataBox
 */
template <typename... Tags>
using AddTags = tmpl::flatten<typelist<Tags...>>;

/*!
 * \ingroup DataBoxGroup
 * \brief List of Compute Item Tags to add to the DataBox
 */
template <typename... Tags>
using AddComputeItemsTags = tmpl::flatten<typelist<Tags...>>;

/*!
 * \ingroup DataBoxGroup
 * \brief Create a new DataBox
 *
 * \details
 * Creates a new DataBox holding types Tags::type filled with the arguments
 * passed to the function. Compute items must be added so that the dependencies
 * of a compute item are added before the compute item. For example, say you
 * have compute items `A` and `B` where `B` depends on `A`, then you must
 * add them using `db::AddComputeItemsTags<A, B>`.
 *
 * \example
 * \snippet Test_DataBox.cpp create_databox
 *
 * \see create_from get_tags_from_box
 *
 * \tparam AddTags the tags of the args being added
 * \tparam AddComputeItems list of \ref ComputeItemTag "compute item tags" to
 * add to the DataBox
 *  \param args the data to be added to the DataBox
 */
template <typename AddTags, typename AddComputeItems = typelist<>,
          typename... Args>
SPECTRE_ALWAYS_INLINE constexpr auto create(Args&&... args) {
  return DataBox<::db::get_databox_list<tmpl::append<
      tmpl::fold<AddTags, tmpl::list<>,
                 tmpl::bind<tmpl::append, tmpl::_state,
                            detail::extract_dependent_items<tmpl::_element>>>,
      tmpl::fold<
          AddComputeItems, tmpl::list<>,
          tmpl::bind<tmpl::append, tmpl::_state,
                     detail::extract_dependent_items<tmpl::_element>>>>>>::
      template create<AddTags, AddComputeItems>(std::forward<Args>(args)...);
}

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
 * \snippet Test_DataBox.cpp create_from_add_item
 *
 * \see create DataBox get_tags_from_box
 *
 * \tparam RemoveTags typelist of Tags to remove
 * \tparam AddTags typelist of Tags corresponding to the arguments to be added
 * \tparam AddComputeItems list of \ref ComputeItemTag "compute item tags" to
 * add to the DataBox
 * \param box the DataBox the new box should be based off
 * \param args the values for the items to add to the DataBox
 * \return DataBox like `box` but altered by RemoveTags and AddTags
 */
template <typename RemoveTags, typename AddTags = typelist<>,
          typename AddComputeItems = typelist<>, typename Box, typename... Args>
SPECTRE_ALWAYS_INLINE constexpr auto create_from(const Box& box,
                                                 Args&&... args) {
  return DataBox<::db::get_databox_list<tmpl::append<
      tmpl::fold<tmpl::fold<RemoveTags, tmpl::list<>,
                            tmpl::lazy::append<tmpl::_state,
                                               detail::extract_dependent_items<
                                                   tmpl::_element>>>,
                 typename Box::tags_list,
                 tmpl::bind<tmpl::remove, tmpl::_state, tmpl::_element>>,
      tmpl::fold<
          tmpl::append<AddTags, AddComputeItems>, tmpl::list<>,
          tmpl::bind<tmpl::append, tmpl::_state,
                     detail::extract_dependent_items<tmpl::_element>>>>>>::
      template create_from<RemoveTags, AddTags, AddComputeItems>(
          box, std::forward<Args>(args)...);
}

namespace detail {
template <typename Type, typename Tags, typename TagLs,
          std::enable_if_t<(tmpl::size<Tags>::value == 0)>* = nullptr>
[[noreturn]] const Type& get_item_from_box(const DataBox<TagLs>& /*box*/,
                                           const std::string& /*tag_name*/) {
  static_assert(tmpl::size<Tags>::value != 0,
                "No items with the requested type were found in the DataBox");
  ERROR("Cannot ever reach this function.");
}

template <typename Type, typename Tags, typename TagLs,
          std::enable_if_t<(tmpl::size<Tags>::value == 1)>* = nullptr>
const Type& get_item_from_box(const DataBox<TagLs>& box,
                              const std::string& tag_name) {
  using Tag = tmpl::front<Tags>;
  if (get_tag_name<Tag>() != tag_name) {
    std::stringstream tags_in_box;
    tmpl::for_each<TagLs>([&tags_in_box](auto t) {
      tags_in_box << "  " << decltype(t)::type::label << "\n";
    });
    ERROR("Could not find the tag named \""
          << tag_name << "\" in the DataBox. Available tags are:\n"
          << tags_in_box.str());
  }
  return box.template get<Tag>();
}

template <typename Type, typename Tags, typename TagLs,
          std::enable_if_t<(tmpl::size<Tags>::value > 1)>* = nullptr>
constexpr const Type& get_item_from_box(const DataBox<TagLs>& box,
                                        const std::string& tag_name) {
  using Tag = tmpl::front<Tags>;
  return get_tag_name<Tag>() == tag_name
             ? box.template get<Tag>()
             : get_item_from_box<Type, tmpl::pop_front<Tags>>(box, tag_name);
}
}  // namespace detail

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
template <typename Type, typename TagLs>
constexpr const Type& get_item_from_box(const DataBox<TagLs>& box,
                                        const std::string& tag_name) {
  using tags = tmpl::filter<
      TagLs, std::is_same<tmpl::bind<item_type, tmpl::_1>, tmpl::pin<Type>>>;
  return detail::get_item_from_box<Type, tags>(box, tag_name);
}

namespace detail {
template <typename TagsLs>
struct Apply;

template <template <typename...> class TagsLs, typename... Tags>
struct Apply<TagsLs<Tags...>> {
  template <typename F, typename... BoxTags, typename... Args>
  static constexpr auto apply(F f, const DataBox<BoxTags...>& box,
                              Args&&... args) {
    static_assert(tt::is_callable<std::remove_pointer_t<F>, item_type<Tags>...,
                                  Args...>::value,
                  "Cannot call the function f with the list of tags and "
                  "arguments specified. Check that the Tags::type and the "
                  "types of the Args match the function f.");
    return f(box.template get<Tags>()..., std::forward<Args>(args)...);
  }

  template <typename F, typename... BoxTags, typename... Args>
  static constexpr auto apply_with_box(F f, const DataBox<BoxTags...>& box,
                                       Args&&... args) {
    static_assert(tt::is_callable<F, DataBox<BoxTags...>, item_type<Tags>...,
                                  Args...>::value,
                  "Cannot call the function f with the list of tags and "
                  "arguments specified. Check that the Tags::type and the "
                  "types of the Args match the function f and that f is "
                  "receiving the correct type of DataBox.");
    return f(box, box.template get<Tags>()..., std::forward<Args>(args)...);
  }
};
}  // namespace detail

/*!
 *  \ingroup DataBoxGroup
 *  \brief Apply the function `f` with argument Tags `TagLs` from DataBox `box`
 *
 *  \details
 *  Apply the function `f` with arguments that are of type `Tags::type` where
 *  `Tags` is defined as `TagLs<Tags...>`. The arguments to `f` are retrieved
 *  from the DataBox `box`.
 *
 *  \usage
 *  Given a function `func` that takes arguments of types
 *  `T1`, `T2`, `A1` and `A2`. Let the Tags for the quantities of types `T1` and
 *  `T2` in the DataBox `box` be `Tag1` and `Tag2`, and objects `a1` of type
 *  `A1` and `a2` of type `A2`, then
 *  \code
 *  auto result = apply<typelist<Tag1, Tag2>>(func, box, a1, a2);
 *  \endcode
 *  \return `decltype(func(box.get<Tag1>(), box.get<Tag2>(), a1, a2))`
 *
 *  \semantics
 *  For tags `Tags...` in a DataBox `box`, and a function `func` that takes
 *  `sizeof...(Tags)` arguments of types `typename Tags::type...`,  and
 *  `sizeof...(Args)` arguments of types `Args...`,
 *  \code
 *  result = func(box, box.get<Tags>()..., args...);
 *  \endcode
 *
 *  \example
 *  \snippet Test_DataBox.cpp apply_example
 *
 *  \see apply_with_box DataBox
 *  \tparam TagsLs typelist of Tags in the order that they are to be passed to
 *  `f`
 *  \param f the function to apply
 *  \param box the DataBox out of which to retrieve the Tags and to pass to `f`
 *  \param args the arguments to pass to the function that are not in the
 *  DataBox, `box`
 */
template <typename TagsLs, typename F, typename... BoxTags, typename... Args>
inline constexpr auto apply(F f, const DataBox<BoxTags...>& box,
                            Args&&... args) {
  return detail::Apply<TagsLs>::apply(f, box, std::forward<Args>(args)...);
}

/*!
 * \ingroup DataBoxGroup
 * \brief Apply the function `f` with argument Tags `TagLs` from DataBox `box`
 *  and `box` as the first argument
 *
 *  \details
 *  Apply the function `f` with arguments that are of type `Tags::type` where
 *  `Tags` is defined as `TagLs<Tags...>`. The arguments to `f` are retrieved
 *  from the DataBox `box` and the first argument passed to `f` is the DataBox.
 *
 *  \usage
 *  Given a function `func` that takes arguments of types `DataBox<Tags...>`,
 *  `T1`, `T2`, `A1` and `A2`. Let the Tags for the quantities of types `T1` and
 *  `T2` in the DataBox `box` be `Tag1` and `Tag2`, and objects `a1` of type
 *  `A1` and `a2` of type `A2`, then
 *  \code
 *  auto result = apply_with_box<typelist<Tag1, Tag2>>(func, box, a1, a2);
 *  \endcode
 *  \return `decltype(func(box, box.get<Tag1>(), box.get<Tag2>(), a1, a2))`
 *
 *  \semantics
 *  For tags `Tags...` in a DataBox `box`, and a function `func` that takes as
 *  its first argument a value of type`decltype(box)`,
 *  `sizeof...(Tags)` arguments of types `typename Tags::type...`, and
 *  `sizeof...(Args)` arguments of types `Args...`,
 *  \code
 *  result = func(box, box.get<Tags>()..., args...);
 *  \endcode
 *
 *  \example
 *  \snippet Test_DataBox.cpp apply_with_box_example
 *
 *  \see apply DataBox
 *  \tparam TagsLs typelist of Tags in the order that they are to be passed to
 *  `f`
 *  \param f the function to apply
 *  \param box the DataBox out of which to retrieve the Tags and to pass to `f`
 *  \param args the arguments to pass to the function that are not in the
 *  DataBox, `box`
 */
template <typename TagsLs, typename F, typename... BoxTags, typename... Args>
inline constexpr auto apply_with_box(F f, const DataBox<BoxTags...>& box,
                                     Args&&... args) {
  return detail::Apply<TagsLs>::apply_with_box(f, box,
                                               std::forward<Args>(args)...);
}

namespace detail {
template <typename Seq, typename Element>
struct filter_helper {
  using type =
      tmpl::not_<tmpl::found<Seq, std::is_same<tmpl::_1, tmpl::pin<Element>>>>;
};
}  // namespace detail

/*!
 * \ingroup DataBoxGroup
 * \brief Get typelist of tags to remove from a DataBox so as to keep only
 * desired tags
 *
 * \metareturns a typelist of tags that need to be removed from the DataBox with
 * tags `DataBoxTagsLs` in order to keep the tags in `KeepTagsLs`
 *
 * \example
 * \snippet Test_DataBox.cpp remove_tags_from_keep_tags
 */
template <typename DataBoxTagsLs, typename KeepTagsLs>
using remove_tags_from_keep_tags =
    tmpl::filter<DataBoxTagsLs,
                 detail::filter_helper<tmpl::pin<KeepTagsLs>, tmpl::_1>>;

}  // namespace db
