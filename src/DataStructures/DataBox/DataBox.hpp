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

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/Deferred.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
template <typename TagList>
class Variables;
// clang-tidy: redundant declaration: DataBox doesn't need to include
// Variables.hpp, but just these forward declarations. However, clang-tidy
// sometimes complains about this.
template <typename Tag, typename TagList>
constexpr typename Tag::type& get(Variables<TagList>& v) noexcept;  // NOLINT
// clang-tidy: redundant declaration: DataBox doesn't need to include
// Variables.hpp, but just these forward declarations. However, clang-tidy
// sometimes complains about this.
template <typename Tag, typename TagList>
constexpr const typename Tag::type& get(  // NOLINT
    const Variables<TagList>& v) noexcept;
/// \endcond

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
 * \requires `tt::is_a<typelist, List>::``value` is true
 * \metareturns `typelist` of all elements in `List` but ordered for the
 * db::DataBox
 */
template <typename List>
using get_databox_list =
    tmpl::sort<List, db::detail::databox_tag_less<tmpl::_1, tmpl::_2>>;

// Forward declarations
/// \cond
template <typename TagsList>
class DataBox;
/// \endcond

// @{
/*!
 * \ingroup TypeTraitsGroup DataBox
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

namespace databox_detail {
namespace tuples_detail {
struct no_such_type {
  no_such_type() = delete;
  no_such_type(no_such_type const& /*unused*/) = delete;
  no_such_type(no_such_type&& /*unused*/) = delete;
  ~no_such_type() = delete;
  no_such_type& operator=(no_such_type const& /*unused*/) = delete;
  no_such_type operator=(no_such_type&& /*unused*/) = delete;
};

template <class Tag>
class TaggedDeferredTupleLeaf;

template <class Tag>
class TaggedDeferredTupleLeaf {
  using value_type = Deferred<item_type<Tag>>;
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
  // Tested in constexpr context in Unit.TaggedDeferredTuple.Ebo
  constexpr TaggedDeferredTupleLeaf() noexcept(
      cpp17::is_nothrow_default_constructible_v<value_type>)
      : value_() {
    static_assert(!cpp17::is_reference_v<value_type>,
                  "Cannot default construct a reference element in a "
                  "TaggedDeferredTuple");
  }

  // clang-tidy: forwarding references are hard
  template <
      class T,
      Requires<!cpp17::is_same_v<std::decay_t<T>, TaggedDeferredTupleLeaf> &&
               cpp17::is_constructible_v<value_type, T&&>> = nullptr>
  constexpr explicit TaggedDeferredTupleLeaf(T&& t) noexcept(  // NOLINT
      cpp17::is_nothrow_constructible_v<value_type, T&&>)
      : value_(std::forward<T>(t)) {  // NOLINT
    static_assert(can_bind_reference<T>(),
                  "Cannot construct an lvalue reference with an rvalue");
  }

  constexpr TaggedDeferredTupleLeaf(TaggedDeferredTupleLeaf const& /*rhs*/) =
      default;
  constexpr TaggedDeferredTupleLeaf(TaggedDeferredTupleLeaf&& /*rhs*/) =
      default;
  constexpr TaggedDeferredTupleLeaf& operator=(
      TaggedDeferredTupleLeaf const& /*rhs*/) = default;
  constexpr TaggedDeferredTupleLeaf& operator=(
      TaggedDeferredTupleLeaf&& /*rhs*/) = default;

  ~TaggedDeferredTupleLeaf() = default;

#if __cplusplus < 201402L
  value_type& get() noexcept { return value_; }
#else
  constexpr value_type& get() noexcept { return value_; }
#endif
  constexpr const value_type& get() const noexcept { return value_; }

  // clang-tidy: runtime-references
  void pup(PUP::er& p) { p | value_; }  // NOLINT
};

struct disable_constructors {
  static constexpr bool enable_default() noexcept { return false; }
  static constexpr bool enable_explicit() noexcept { return false; }
  static constexpr bool enable_implicit() noexcept { return false; }
};
}  // namespace tuples_detail

template <class... Tags>
class TaggedDeferredTuple;

template <class Tag, class... Tags>
constexpr const Deferred<::db::item_type<Tag>>& get(
    const TaggedDeferredTuple<Tags...>& t) noexcept;
template <class Tag, class... Tags>
constexpr Deferred<::db::item_type<Tag>>& get(
    TaggedDeferredTuple<Tags...>& t) noexcept;
template <class Tag, class... Tags>
constexpr const Deferred<::db::item_type<Tag>>&& get(
    const TaggedDeferredTuple<Tags...>&& t) noexcept;
template <class Tag, class... Tags>
constexpr Deferred<::db::item_type<Tag>>&& get(
    TaggedDeferredTuple<Tags...>&& t) noexcept;

// clang-tidy: does not define copy or move assignment operator, false positive
template <class... Tags>
class TaggedDeferredTuple  // NOLINT
    : private tuples_detail::TaggedDeferredTupleLeaf<Tags>... {
  template <class... Args>
  struct pack_is_TaggedDeferredTuple : std::false_type {};
  template <class Arg>
  struct pack_is_TaggedDeferredTuple<Arg>
      : std::is_same<std::decay_t<Arg>, TaggedDeferredTuple> {};

  template <bool EnableConstructor, class Dummy = void>
  struct args_constructor : tuples_detail::disable_constructors {};

  template <class Dummy>
  struct args_constructor<true, Dummy> {
    static constexpr bool enable_default() {
      return tmpl2::flat_all_v<
          cpp17::is_default_constructible_v<Deferred<item_type<Tags>>>...>;
    }

    template <class... Ts>
    static constexpr bool enable_explicit() noexcept {
      return tmpl2::flat_all_v<cpp17::is_constructible_v<
                 tuples_detail::TaggedDeferredTupleLeaf<Tags>, Ts>...> and
             not tmpl2::flat_all_v<
                 cpp17::is_convertible_v<Ts, Deferred<item_type<Tags>>>...>;
    }
    template <class... Ts>
    static constexpr bool enable_implicit() noexcept {
      return sizeof...(Ts) == sizeof...(Tags) and
             tmpl2::flat_all_v<cpp17::is_constructible_v<
                 tuples_detail::TaggedDeferredTupleLeaf<Tags>, Ts>...> and
             tmpl2::flat_all_v<
                 cpp17::is_convertible_v<Ts, Deferred<item_type<Tags>>>...>;
    }
  };

  template <bool EnableConstructor, bool = sizeof...(Tags) == 1,
            class Dummy = void>
  struct tuple_like_constructor : tuples_detail::disable_constructors {};

  template <class Dummy>
  struct tuple_like_constructor<true, false, Dummy> {
    template <class Tuple, class... Ts>
    static constexpr bool enable_explicit() noexcept {
      return not tmpl2::flat_all_v<
          cpp17::is_convertible_v<Ts, Deferred<item_type<Tags>>>...>;
    }

    template <class Tuple, class... Ts>
    static constexpr bool enable_implicit() noexcept {
      return tmpl2::flat_all_v<
          cpp17::is_convertible_v<Ts, Deferred<item_type<Tags>>>...>;
    }
  };

  template <class Dummy>
  struct tuple_like_constructor<true, true, Dummy> {
    template <class Tuple, class... Ts>
    static constexpr bool enable_explicit() noexcept {
      return not tmpl2::flat_all_v<
                 cpp17::is_convertible_v<Ts, Deferred<item_type<Tags>>>...> and
             (not tmpl2::flat_all_v<cpp17::is_convertible_v<
                  Tuple, Deferred<item_type<Tags>>>...> and
              not tmpl2::flat_all_v<cpp17::is_constructible_v<
                  Deferred<item_type<Tags>>, Tuple>...> and
              not tmpl2::flat_all_v<
                  cpp17::is_same_v<Deferred<item_type<Tags>>, Ts>...>);
    }

    template <class Tuple, class... Ts>
    static constexpr bool enable_implicit() noexcept {
      return tmpl2::flat_all_v<
                 cpp17::is_convertible_v<Ts, Deferred<item_type<Tags>>>...> and
             (not tmpl2::flat_all_v<cpp17::is_convertible_v<
                  Tuple, Deferred<item_type<Tags>>>...> and
              not tmpl2::flat_all_v<cpp17::is_constructible_v<
                  Deferred<item_type<Tags>>, Tuple>...> and
              not tmpl2::flat_all_v<
                  cpp17::is_same_v<Deferred<item_type<Tags>>, Ts>...>);
    }
  };

  // C++17 Draft 23.5.3.2 Assignment - helper aliases
  using is_copy_assignable = tmpl2::flat_all<
      cpp17::is_copy_assignable_v<Deferred<item_type<Tags>>>...>;
  using is_nothrow_copy_assignable = tmpl2::flat_all<
      cpp17::is_nothrow_copy_assignable_v<Deferred<item_type<Tags>>>...>;
  using is_move_assignable = tmpl2::flat_all<
      cpp17::is_move_assignable_v<Deferred<item_type<Tags>>>...>;
  using is_nothrow_move_assignable = tmpl2::flat_all<
      cpp17::is_nothrow_move_assignable_v<Deferred<item_type<Tags>>>...>;

  // clang-tidy: redundant declaration
  template <class Tag, class... LTags>
  friend constexpr const Deferred<::db::item_type<Tag>>& get(  // NOLINT
      const TaggedDeferredTuple<LTags...>& t) noexcept;
  template <class Tag, class... LTags>
  friend constexpr Deferred<::db::item_type<Tag>>& get(  // NOLINT
      TaggedDeferredTuple<LTags...>& t) noexcept;
  template <class Tag, class... LTags>
  friend constexpr const Deferred<::db::item_type<Tag>>&& get(  // NOLINT
      const TaggedDeferredTuple<LTags...>&& t) noexcept;
  template <class Tag, class... LTags>
  friend constexpr Deferred<::db::item_type<Tag>>&& get(  // NOLINT
      TaggedDeferredTuple<LTags...>&& t) noexcept;

 public:
  static constexpr size_t size() noexcept { return sizeof...(Tags); }

  // clang-tidy: runtime-references
  void pup(PUP::er& p) {  // NOLINT
    static_cast<void>(std::initializer_list<char>{
        (tuples_detail::TaggedDeferredTupleLeaf<Tags>::pup(p), '0')...});
  }

  // C++17 Draft 23.5.3.1 Construction
  template <bool Dummy = true,
            Requires<args_constructor<Dummy>::enable_default()> = nullptr>
  // clang-tidy: use = default, can't because won't compile
  constexpr TaggedDeferredTuple() noexcept(  // NOLINT
      tmpl2::flat_all_v<cpp17::is_nothrow_default_constructible_v<
          Deferred<item_type<Tags>>>...>) {}

  constexpr TaggedDeferredTuple(TaggedDeferredTuple const& /*rhs*/) = default;
  constexpr TaggedDeferredTuple(TaggedDeferredTuple&& /*rhs*/) = default;

  template <bool Dummy = true,
            Requires<args_constructor<Dummy>::template enable_explicit<
                Deferred<item_type<Tags>> const&...>()> = nullptr>
  constexpr explicit TaggedDeferredTuple(item_type<Tags> const&... ts) noexcept(
      tmpl2::flat_all_v<cpp17::is_nothrow_copy_constructible_v<
          tuples_detail::TaggedDeferredTupleLeaf<Tags>>...>)
      : tuples_detail::TaggedDeferredTupleLeaf<Tags>(ts)... {}

  template <bool Dummy = true,
            Requires<args_constructor<Dummy>::template enable_implicit<
                Deferred<item_type<Tags>> const&...>()> = nullptr>
  // clang-tidy: mark explicit
  // clang-format off
  constexpr TaggedDeferredTuple(  // NOLINT
      item_type<
      Tags> const&... ts)
      noexcept(tmpl2::flat_all_v<cpp17::is_nothrow_copy_constructible_v<
               tuples_detail::TaggedDeferredTupleLeaf<Tags>>...>)
      // clang-format on
      : tuples_detail::TaggedDeferredTupleLeaf<Tags>(ts)... {}

  template <class... Us,
            Requires<args_constructor<
                not pack_is_TaggedDeferredTuple<Us...>::value and
                sizeof...(Us) == sizeof...(Tags)>::
                         template enable_explicit<Us&&...>()> = nullptr>
  constexpr explicit TaggedDeferredTuple(Us&&... us) noexcept(
      tmpl2::flat_all_v<cpp17::is_nothrow_constructible_v<
          tuples_detail::TaggedDeferredTupleLeaf<Tags>, Us&&>...>)
      : tuples_detail::TaggedDeferredTupleLeaf<Tags>(std::forward<Us>(us))... {}

  template <class... Us,
            Requires<args_constructor<
                not pack_is_TaggedDeferredTuple<Us...>::value and
                sizeof...(Us) == sizeof...(Tags)>::
                         template enable_implicit<Us&&...>()> = nullptr>
  // clang-tidy: mark explicit
  constexpr TaggedDeferredTuple(Us&&... us) noexcept(  // NOLINT
      tmpl2::flat_all_v<cpp17::is_nothrow_constructible_v<
          tuples_detail::TaggedDeferredTupleLeaf<Tags>, Us&&>...>)
      : tuples_detail::TaggedDeferredTupleLeaf<Tags>(std::forward<Us>(us))... {}

  template <class... UTags,
            Requires<tuple_like_constructor<
                sizeof...(Tags) == sizeof...(UTags) and
                tmpl2::flat_all_v<cpp17::is_constructible_v<
                    Deferred<item_type<Tags>>, item_type<UTags> const&>...>>::
                         template enable_explicit<
                             TaggedDeferredTuple<UTags...> const&,
                             item_type<UTags>...>()> = nullptr>
  constexpr explicit TaggedDeferredTuple(
      TaggedDeferredTuple<UTags...> const&
          t) noexcept(tmpl2::
                          flat_all_v<cpp17::is_nothrow_constructible_v<
                              Deferred<item_type<Tags>>,
                              item_type<UTags> const&>...>)
      : tuples_detail::TaggedDeferredTupleLeaf<Tags>(get<UTags>(t))... {}

  template <class... UTags,
            Requires<tuple_like_constructor<
                sizeof...(Tags) == sizeof...(UTags) and
                tmpl2::flat_all_v<cpp17::is_constructible_v<
                    Deferred<item_type<Tags>>, item_type<UTags> const&>...>>::
                         template enable_implicit<
                             TaggedDeferredTuple<UTags...> const&,
                             item_type<UTags>...>()> = nullptr>
  // clang-tidy: mark explicit
  constexpr TaggedDeferredTuple(  // NOLINT
      TaggedDeferredTuple<UTags...> const&
          t) noexcept(tmpl2::
                          flat_all_v<cpp17::is_nothrow_constructible_v<
                              Deferred<item_type<Tags>>,
                              item_type<UTags> const&>...>)
      : tuples_detail::TaggedDeferredTupleLeaf<Tags>(get<UTags>(t))... {}

  template <
      class... UTags,
      Requires<tuple_like_constructor<
          sizeof...(Tags) == sizeof...(UTags) and
          tmpl2::flat_all_v<cpp17::is_constructible_v<Deferred<item_type<Tags>>,
                                                      item_type<UTags>&&>...>>::
                   template enable_explicit<TaggedDeferredTuple<UTags...>&&,
                                            Deferred<item_type<UTags>>...>()> =
          nullptr>
  constexpr explicit TaggedDeferredTuple(
      TaggedDeferredTuple<UTags...>&&
          t) noexcept(tmpl2::
                          flat_all_v<cpp17::is_nothrow_constructible_v<
                              Deferred<item_type<Tags>>,
                              item_type<UTags>&&>...>)
      : tuples_detail::TaggedDeferredTupleLeaf<Tags>(
            std::forward<item_type<UTags>>(get<UTags>(t)))... {}

  template <
      class... UTags,
      Requires<tuple_like_constructor<
          sizeof...(Tags) == sizeof...(UTags) and
          tmpl2::flat_all_v<cpp17::is_constructible_v<
              Deferred<item_type<Tags>>, Deferred<item_type<UTags>>&&>...>>::
                   template enable_implicit<TaggedDeferredTuple<UTags...>&&,
                                            Deferred<item_type<UTags>>...>()> =
          nullptr>
  // clang-tidy: mark explicit
  constexpr TaggedDeferredTuple(  // NOLINT
      TaggedDeferredTuple<UTags...>&&
          t) noexcept(tmpl2::
                          flat_all_v<cpp17::is_nothrow_constructible_v<
                              Deferred<item_type<Tags>>,
                              Deferred<item_type<UTags>>&&>...>)
      : tuples_detail::TaggedDeferredTupleLeaf<Tags>(
            std::forward<item_type<UTags>>(get<UTags>(t)))... {}

  ~TaggedDeferredTuple() = default;

  // C++17 Draft 23.5.3.2 Assignment
  constexpr TaggedDeferredTuple& operator=(
      typename std::conditional<is_copy_assignable::value, TaggedDeferredTuple,
                                tuples_detail::no_such_type>::type const&
          t) noexcept(is_nothrow_copy_assignable::value) {
    if (&t == this) {
      return *this;
    }
    swallow((::db::databox_detail::get<Tags>(*this) =
                 ::db::databox_detail::get<Tags>(t))...);
    return *this;
  }

  constexpr TaggedDeferredTuple& operator=(
      typename std::conditional<is_move_assignable::value, TaggedDeferredTuple,
                                tuples_detail::no_such_type>::type&&
          t) noexcept(is_nothrow_move_assignable::value) {
    if (&t == this) {
      return *this;
    }
    swallow((::db::databox_detail::get<Tags>(*this) =
                 std::forward<Deferred<item_type<Tags>>>(
                     ::db::databox_detail::get<Tags>(t)))...);
    return *this;
  }

  template <class... UTags,
            Requires<sizeof...(Tags) == sizeof...(UTags) and
                     tmpl2::flat_all_v<cpp17::is_assignable_v<
                         Deferred<item_type<Tags>>&,
                         Deferred<item_type<UTags>> const&>...>> = nullptr>
  constexpr TaggedDeferredTuple&
  operator=(TaggedDeferredTuple<UTags...> const& t) noexcept(
      tmpl2::flat_all_v<cpp17::is_nothrow_assignable_v<
          Deferred<item_type<Tags>>&, Deferred<item_type<UTags>> const&>...>) {
    swallow((get<Tags>(*this) = get<UTags>(t))...);
    return *this;
  }

  template <class... UTags,
            Requires<sizeof...(Tags) == sizeof...(UTags) and
                     tmpl2::flat_all_v<cpp17::is_assignable_v<
                         Deferred<item_type<Tags>>&,
                         Deferred<item_type<UTags>>&&>...>> = nullptr>
  constexpr TaggedDeferredTuple&
  operator=(TaggedDeferredTuple<UTags...>&& t) noexcept(
      tmpl2::flat_all_v<cpp17::is_nothrow_assignable_v<
          Deferred<item_type<Tags>>&, Deferred<item_type<UTags>>&&>...>) {
    swallow((get<Tags>(*this) =
                 std::forward<Deferred<item_type<UTags>>>(get<UTags>(t)))...);
    return *this;
  }
};

template <>
class TaggedDeferredTuple<> {
 public:
  static constexpr size_t size() noexcept { return 0; }
  TaggedDeferredTuple() noexcept = default;
};

template <class Tag, class... Tags>
inline constexpr const Deferred<::db::item_type<Tag>>& get(
    const TaggedDeferredTuple<Tags...>& t) noexcept {
  static_assert(
      cpp17::is_base_of_v<tuples_detail::TaggedDeferredTupleLeaf<Tag>,
                          TaggedDeferredTuple<Tags...>>,
      "Could not retrieve Tag from DataBox. See the instantiation for "
      "what Tag is being retrieved and what Tags are available.");
  return static_cast<const tuples_detail::TaggedDeferredTupleLeaf<Tag>&>(t)
      .get();
}
template <class Tag, class... Tags>
inline constexpr Deferred<::db::item_type<Tag>>& get(
    TaggedDeferredTuple<Tags...>& t) noexcept {
  static_assert(
      cpp17::is_base_of_v<tuples_detail::TaggedDeferredTupleLeaf<Tag>,
                          TaggedDeferredTuple<Tags...>>,
      "Could not retrieve Tag from DataBox. See the instantiation for "
      "what Tag is being retrieved and what Tags are available.");
  return static_cast<tuples_detail::TaggedDeferredTupleLeaf<Tag>&>(t).get();
}
template <class Tag, class... Tags>
inline constexpr const Deferred<::db::item_type<Tag>>&& get(
    const TaggedDeferredTuple<Tags...>&& t) noexcept {
  static_assert(
      cpp17::is_base_of_v<tuples_detail::TaggedDeferredTupleLeaf<Tag>,
                          TaggedDeferredTuple<Tags...>>,
      "Could not retrieve Tag from DataBox. See the instantiation for "
      "what Tag is being retrieved and what Tags are available.");
  return static_cast<const ::db::item_type<Tag>&&>(
      static_cast<const tuples_detail::TaggedDeferredTupleLeaf<Tag>&&>(t)
          .get());
}
template <class Tag, class... Tags>
inline constexpr Deferred<::db::item_type<Tag>>&& get(
    TaggedDeferredTuple<Tags...>&& t) noexcept {
  static_assert(
      cpp17::is_base_of_v<tuples_detail::TaggedDeferredTupleLeaf<Tag>,
                          TaggedDeferredTuple<Tags...>>,
      "Could not retrieve Tag from DataBox. See the instantiation for "
      "what Tag is being retrieved and what Tags are available.");
  return static_cast<::db::item_type<Tag>&&>(
      static_cast<tuples_detail::TaggedDeferredTupleLeaf<Tag>&&>(t).get());
}

template <typename Element, typename = std::nullptr_t>
struct extract_dependent_items {
  using type = typelist<Element>;
};

template <typename Element>
struct extract_dependent_items<
    Element, Requires<tt::is_a_v<Variables, item_type<Element>>>> {
  using type =
      tmpl::append<typelist<Element>, typename item_type<Element>::tags_list>;
};

template <bool IsVariables>
struct closure_extract_dependent_items {
  template <typename Element>
  using f = tmpl::list<Element>;
};

template <>
struct closure_extract_dependent_items<true> {
  template <typename Element>
  using f = tmpl::append<typelist<Element>,
                         typename ::db::item_type<Element>::tags_list>;
};

// Given a typelist List, returns a new typelist with all the Variables Tags
template <typename List>
using extracted_items = tmpl::flatten<
    tmpl::transform<List, databox_detail::extract_dependent_items<tmpl::_1>>>;

template <typename Caller, typename Callee, typename List,
          typename = std::nullptr_t>
struct create_dependency_graph {
  using new_edge = tmpl::edge<Callee, Caller>;
  using type = tmpl::conditional_t<
      tmpl::found<List, std::is_same<tmpl::_1, tmpl::pin<new_edge>>>::value,
      List, tmpl::push_back<List, new_edge>>;
};

template <typename Caller, typename Callee, typename List>
struct create_dependency_graph<
    Caller, Callee, List, Requires<is_simple_compute_item<Callee>::value>> {
  using sub_tree = tmpl::fold<
      typename Callee::argument_tags, List,
      create_dependency_graph<tmpl::pin<Callee>, tmpl::_element, tmpl::_state>>;
  using type = tmpl::conditional_t<
      cpp17::is_same_v<void, Caller>, sub_tree,
      tmpl::push_back<sub_tree, tmpl::edge<Callee, Caller>>>;
};

template <typename Caller, typename Callee, typename List>
struct create_dependency_graph<
    Caller, Callee, List, Requires<is_variables_compute_item<Callee>::value>> {
  using partial_sub_tree = tmpl::fold<
      typename Callee::argument_tags, List,
      create_dependency_graph<tmpl::pin<Callee>, tmpl::_element, tmpl::_state>>;
  using variables_tags_dependency = tmpl::fold<
      typename item_type<Callee>::tags_list, tmpl::list<>,
      tmpl::bind<tmpl::push_back, tmpl::_state,
                 tmpl::bind<tmpl::edge, tmpl::pin<Callee>, tmpl::_element>>>;
  using sub_tree = tmpl::append<partial_sub_tree, variables_tags_dependency>;
  using type = tmpl::conditional_t<
      cpp17::is_same_v<void, Caller>, sub_tree,
      tmpl::push_back<sub_tree, tmpl::edge<Callee, Caller>>>;
};

template <class T>
using is_variables = tt::is_a<Variables, T>;
}  // namespace databox_detail

/*!
 * \ingroup DataBoxGroup
 * \brief A DataBox stores objects that can be retrieved by using Tags
 * \warning
 * The order of the tags in DataBoxes returned by db::create and
 * db::create_from depends on implementation-defined behavior, and
 * therefore should not be specified in source files. If explicitly
 * naming a DataBox type is necessary they should be generated using
 * db::get_databox_list.
 *
 * \see db::create db::create_from
 *
 * @tparam TagsList a metasequence
 * @tparam Tags list of DataBoxTag's
 */
template <template <typename...> class TagsList, typename... Tags>
class DataBox<TagsList<Tags...>> {
  static_assert(
      tmpl2::flat_all_v<cpp17::is_base_of_v<db::DataBoxTag, Tags>...>,
      "All structs used to Tag (compute) items in a DataBox must derive off of "
      "db::DataBoxTag");
  static_assert(
      tmpl2::flat_all_v<detail::tag_has_label<Tags>::value...>,
      "Missing a label on a Tag. All Tags must have a static "
      "constexpr db::DataBoxString member variable named 'label' with "
      "the name of the Tag.");
  static_assert(
      tmpl2::flat_all_v<detail::tag_label_correct_type<Tags>::value...>,
      "One of the labels of the Tags in a DataBox has the incorrect "
      "type. It should be a DataBoxString.");

 public:
  /*!
   * \brief A ::typelist of Tags that the DataBox holds
   */
  using tags_list = typelist<Tags...>;

  using compute_item_tags =
      tmpl::filter<tags_list, std::is_base_of<db::ComputeItemTag, tmpl::_1>>;

  using item_tags =
      tmpl::filter<tags_list,
                   tmpl::not_<std::is_base_of<db::ComputeItemTag, tmpl::_1>>>;

  using variables_item_tags =
      tmpl::filter<item_tags, tmpl::bind<databox_detail::is_variables,
                                         tmpl::bind<db::item_type, tmpl::_1>>>;

  using extracted_from_variables =
      tmpl::filter<databox_detail::extracted_items<variables_item_tags>,
                   tmpl::not_<tmpl::bind<databox_detail::is_variables,
                                         tmpl::bind<db::item_type, tmpl::_1>>>>;

  using edge_list =
      tmpl::fold<compute_item_tags, typelist<>,
                 databox_detail::create_dependency_graph<void, tmpl::_element,
                                                         tmpl::_state>>;

  /// \cond HIDDEN_SYMBOLS
  /*!
   * \note the default constructor is only used for serialization
   */
  DataBox() = default;
  DataBox(DataBox&& rhs) noexcept(
      cpp17::is_nothrow_move_constructible_v<
          databox_detail::TaggedDeferredTuple<Tags...>>) = default;
  DataBox& operator=(DataBox&& rhs) noexcept(
      cpp17::is_nothrow_move_assignable_v<
          databox_detail::TaggedDeferredTuple<Tags...>>) {
    if (&rhs == this) {
      return *this;
    }
    data_ = std::move(rhs.data_);
    return *this;
  }
  DataBox(const DataBox& rhs) = default;
  DataBox& operator=(const DataBox& rhs) {
    if (&rhs == this) {
      return *this;
    }
    data_ = rhs.data_;
    return *this;
  }
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

  template <typename T, typename TagList>
  // clang-tidy: redundant declaration
  friend auto get(  // NOLINT
      const DataBox<TagList>& t) noexcept
      -> const tmpl::conditional_t<cpp17::is_same_v<T, ::Tags::DataBox>,
                                   DataBox<TagList>, item_type<T>>&;

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
  const Deferred<item_type<T>>& get_lazy() const noexcept {
    if (UNLIKELY(mutate_locked_box_)) {
      ERROR("Unable to retrieve a (compute) item '"
            << T::label
            << "' from the DataBox from within a "
               "call to mutate. You must pass these either through the capture "
               "list of the lambda or the constructor of a class, this "
               "restriction exists to avoid complexity.");
    }
    return ::db::databox_detail::get<T>(data_);
  }
  /// \endcond

  template <typename... MutateTags, typename TagList, typename Invokable,
            typename... Args>
  // clang-tidy: redundant declaration
  friend void mutate(DataBox<TagList>& box,                            // NOLINT
                     Invokable&& invokable, Args&&... args) noexcept;  // NOLINT

 private:
  template <typename Tag,
            Requires<not cpp17::is_same_v<Tag, ::Tags::DataBox>> = nullptr>
  const db::item_type<Tag>& get_impl() const noexcept;

  template <typename Tag,
            Requires<cpp17::is_same_v<Tag, ::Tags::DataBox>> = nullptr>
  const DataBox& get_impl() const noexcept;

  template <typename... TagsInArgsOrder, typename... FullItems,
            typename... ComputeItemTags, typename... FullComputeItems,
            typename... Args,
            Requires<not tmpl2::flat_any_v<
                db::is_databox<std::decay_t<Args>>::value...>> = nullptr>
  constexpr DataBox(typelist<TagsInArgsOrder...> /*meta*/,
                    typelist<FullItems...> /*meta*/,
                    typelist<ComputeItemTags...> /*meta*/,
                    typelist<FullComputeItems...> /*meta*/, Args&&... args);

  template <typename OldTags, typename... KeepTags, typename... NewTags,
            typename... NewComputeItems, typename ComputeItemsToKeep,
            typename... MutatedTags, typename... Args>
  constexpr DataBox(const DataBox<OldTags>& old_box,
                    typelist<KeepTags...> /*meta*/,
                    typelist<NewTags...> /*meta*/,
                    typelist<NewComputeItems...> /*meta*/,
                    ComputeItemsToKeep /*meta*/,
                    tmpl::list<MutatedTags...> /*meta*/, Args&&... args);

  SPECTRE_ALWAYS_INLINE void check_tags() const {
#ifdef SPECTRE_DEBUG
    ASSERT(tmpl::size<tags_list>::value == 0 or
               tmpl::for_each<tags_list>(detail::check_tag_labels{}).value,
           "Could not match one of the Tag labels with the Tag type. That is, "
           "the label of a Tag must be the same as the Tag.");
#endif
  }

  bool mutate_locked_box_{false};
  databox_detail::TaggedDeferredTuple<Tags...> data_;
};

/// \cond HIDDEN_SYMBOLS
namespace databox_detail {
template <typename T, typename = std::nullptr_t>
struct select_if_variables {
  using type = tmpl::list<>;
};
template <typename T>
struct select_if_variables<T, Requires<tt::is_a_v<Variables, item_type<T>>>> {
  using type = typename item_type<T>::tags_list;
};

template <
    typename VariablesTag, typename... Ts, typename... Tags,
    Requires<not tt::is_a_v<Variables, item_type<VariablesTag>>> = nullptr>
SPECTRE_ALWAYS_INLINE constexpr void add_variables_compute_item_tags_to_box(
    databox_detail::TaggedDeferredTuple<Tags...>& /*data*/,
    typelist<Ts...> /*meta*/) {}

template <typename VariablesTag, typename... VarsTags, typename... Tags,
          Requires<tt::is_a_v<Variables, item_type<VariablesTag>>> = nullptr>
SPECTRE_ALWAYS_INLINE constexpr void add_variables_compute_item_tags_to_box(
    databox_detail::TaggedDeferredTuple<Tags...>& data,
    typelist<VarsTags...> /*meta*/) {
  const auto helper = [lazy_function = ::db::databox_detail::get<VariablesTag>(
                           data)](auto tag) {
    return get<decltype(tag)>(lazy_function.get());
  };
  static_cast<void>(std::initializer_list<char>{
      (static_cast<void>(
           ::db::databox_detail::get<VarsTags>(data) =
               make_deferred<db::item_type<VarsTags>>(helper, VarsTags{})),
       '0')...});
}

template <
    typename VariablesTag, typename... Ts, typename... Tags,
    Requires<not tt::is_a_v<Variables, item_type<VariablesTag>>> = nullptr>
SPECTRE_ALWAYS_INLINE constexpr void add_variables_item_tags_to_box(
    databox_detail::TaggedDeferredTuple<Tags...>& /*data*/,
    typelist<Ts...> /*meta*/) {}

template <typename VariablesTag, typename... Ts, typename... Tags,
          Requires<tt::is_a_v<Variables, item_type<VariablesTag>>> = nullptr>
SPECTRE_ALWAYS_INLINE constexpr void add_variables_item_tags_to_box(
    databox_detail::TaggedDeferredTuple<Tags...>& data,
    typelist<Ts...> /*meta*/) {
  const auto helper = [&data](auto tag_v) {
    using tag = decltype(tag_v);
    auto& vars =
        get<tag>(::db::databox_detail::get<VariablesTag>(data).mutate());
    ::db::databox_detail::get<tag>(data) =
        Deferred<db::item_type<tag>>(db::item_type<tag>{});
    auto& var = ::db::databox_detail::get<tag>(data).mutate();
    for (auto vars_it = vars.begin(), var_it = var.begin();
         vars_it != vars.end(); ++vars_it, ++var_it) {
      var_it->set_data_ref(
          gsl::not_null<std::add_pointer_t<decltype(*vars_it)>>(&*vars_it));
    }
  };

  static_cast<void>(
      std::initializer_list<char>{(static_cast<void>(helper(Ts{})), '0')...});
}

template <size_t ArgsIndex, typename Tag, typename... Tags, typename... Ts>
SPECTRE_ALWAYS_INLINE constexpr cpp17::void_type add_item_to_box(
    std::tuple<Ts...>& tupull,
    databox_detail::TaggedDeferredTuple<Tags...>&
        data) noexcept(noexcept(::db::databox_detail::get<Tag>(data) =
                                    Deferred<item_type<Tag>>(std::move(
                                        std::get<ArgsIndex>(tupull)))) and
                       noexcept(add_variables_item_tags_to_box<Tag>(
                           data, typename select_if_variables<Tag>::type{}))) {
  static_assert(
      not tt::is_a<Deferred,
                   std::decay_t<decltype(std::get<ArgsIndex>(tupull))>>::value,
      "Cannot pass a Deferred into the DataBox as an Item. This "
      "functionally can trivially be added, however it is "
      "intentionally omitted because users of DataBox are not "
      "supposed to deal with Deferred.");
  ::db::databox_detail::get<Tag>(data) =
      Deferred<item_type<Tag>>(std::move(std::get<ArgsIndex>(tupull)));
  // If `tag` holds a Variables then add the contained Tensor's
  add_variables_item_tags_to_box<Tag>(
      data, typename select_if_variables<Tag>::type{});
  return cpp17::void_type{};  // must return in constexpr function
}

template <typename ComputeItem, typename FullTagList, typename... Tags,
          typename... ComputeItemArgumentsTags>
// clang-format off
SPECTRE_ALWAYS_INLINE constexpr void
add_compute_item_to_box_impl(
    databox_detail::TaggedDeferredTuple<Tags...>& data,
    tmpl::list<ComputeItemArgumentsTags...> /*meta*/) noexcept(
        noexcept(::db::databox_detail::get<ComputeItem>(data) =
                 make_deferred<db::item_type<ComputeItem>>(
                     ComputeItem::function,
                     ::db::databox_detail::get<ComputeItemArgumentsTags>
                     (data)...))) {
  // clang-format on
  static_assert(
      tmpl2::flat_all_v<
          cpp17::is_base_of_v<db::DataBoxTag, ComputeItemArgumentsTags>...>,
      "Cannot have non-DataBoxTag arguments to a ComputeItem. Please make "
      "sure all the specified argument_tags in the ComputeItem derive from "
      "db::DataBoxTag.");
  using index = tmpl::index_of<FullTagList, ComputeItem>;
  static_assert(not tmpl2::flat_any_v<
                    cpp17::is_same_v<ComputeItemArgumentsTags, ComputeItem>...>,
                "A ComputeItem cannot take its own Tag as an argument.");
  static_assert(
      tmpl2::flat_all_v<
          tmpl::less<tmpl::index_of<FullTagList, ComputeItemArgumentsTags>,
                     index>::value...>,
      "The dependencies of a ComputeItem must be added before the "
      "ComputeItem itself. This is done to ensure no cyclic "
      "dependencies arise.");

  ::db::databox_detail::get<ComputeItem>(data) =
      make_deferred<db::item_type<ComputeItem>>(
          ComputeItem::function,
          ::db::databox_detail::get<ComputeItemArgumentsTags>(data)...);
}

template <typename Tag, typename FullTagList, typename... Tags>
SPECTRE_ALWAYS_INLINE constexpr void add_compute_item_to_box(
    databox_detail::TaggedDeferredTuple<Tags...>&
        data) noexcept(noexcept(add_compute_item_to_box_impl<Tag,
                                                             FullTagList>(
    data, typename Tag::argument_tags{}))) {
  add_compute_item_to_box_impl<Tag, FullTagList>(data,
                                                 typename Tag::argument_tags{});
  // If `tag` holds a Variables then add the contained Tensor's
  add_variables_compute_item_tags_to_box<Tag>(
      data, typename select_if_variables<Tag>::type{});
}

// Add items or compute items to the TaggedDeferredTuple `data`. If
// `AddItemTags...` is an empty pack then only compute items are added, while if
// `AddComputeItemTags...` is an empty pack only items are added. Items are
// always added before compute items.
template <typename FullTagList, typename... Ts, typename... Tags,
          typename... AddItemTags, typename... AddComputeItemTags, size_t... Is,
          bool... DependenciesAddedBefore>
SPECTRE_ALWAYS_INLINE void add_items_to_box(
    std::tuple<Ts...>& tupull,
    databox_detail::TaggedDeferredTuple<Tags...>& data,
    tmpl::list<AddItemTags...> /*meta*/, std::index_sequence<Is...> /*meta*/,
    tmpl::list<AddComputeItemTags...> /*meta*/) {
  swallow(add_item_to_box<Is, AddItemTags>(tupull, data)...);
  static_cast<void>(std::initializer_list<char>{(
      add_compute_item_to_box<AddComputeItemTags, FullTagList>(data), '0')...});
}

template <typename OldTagsList, typename... Tags, typename... OldTags>
SPECTRE_ALWAYS_INLINE constexpr void merge_old_box(
    const DataBox<OldTagsList>& old_box,
    databox_detail::TaggedDeferredTuple<Tags...>& data,
    tmpl::list<OldTags...> /*meta*/) {
  static_cast<void>(std::initializer_list<char>{
      (static_cast<void>(::db::databox_detail::get<OldTags>(data) =
                             old_box.template get_lazy<OldTags>()),
       '0')...});
}

template <bool IsComputeItemTag>
struct get_argument_list_impl {
  template <class Tag>
  using f = tmpl::list<>;
};

template <>
struct get_argument_list_impl<true> {
  template <class Tag>
  using f = typename Tag::argument_tags;
};

template <class Tag>
using get_argument_list = typename get_argument_list_impl<
    ::db::is_compute_item_v<Tag>>::template f<Tag>;

template <typename ComputeItem, typename... Tags, typename... ComputeItemTags,
          Requires<db::is_compute_item_v<ComputeItem>> = nullptr>
SPECTRE_ALWAYS_INLINE static constexpr void add_reset_compute_item_to_box(
    databox_detail::TaggedDeferredTuple<Tags...>& data,
    tmpl::list<ComputeItemTags...> /*meta*/) {
  update_deferred_args(
      make_not_null(&::db::databox_detail::get<ComputeItem>(data)),
      ComputeItem::function,
      ::db::databox_detail::get<ComputeItemTags>(data)...);

  // If `tag` holds a Variables then add the contained Tensor's
  add_variables_compute_item_tags_to_box<ComputeItem>(
      data, typename select_if_variables<ComputeItem>::type{});
}

template <typename ComputeItem, typename... Tags,
          Requires<not db::is_compute_item_v<ComputeItem>> = nullptr>
SPECTRE_ALWAYS_INLINE static constexpr void add_reset_compute_item_to_box(
    databox_detail::TaggedDeferredTuple<Tags...>& /*data*/,
    tmpl::list<> /*meta*/) {}

template <bool empty, class full_edge_list, class mutate_tags_list>
struct reset_compute_items_after_mutate;

template <class full_edge_list, class... tags, template <class...> class F>
struct reset_compute_items_after_mutate<false, full_edge_list, F<tags...>> {
  using tags_dependent_on_current_tags = tmpl::transform<
      tmpl::append<tmpl::filter<
          full_edge_list,
          std::is_same<tmpl::pin<tags>, tmpl::get_source<tmpl::_1>>>...>,
      tmpl::get_destination<tmpl::_1>>;

  template <typename... Tags>
  SPECTRE_ALWAYS_INLINE static constexpr void apply(
      databox_detail::TaggedDeferredTuple<Tags...>& data) noexcept {
    (void)std::initializer_list<bool>{
        (add_reset_compute_item_to_box<tags>(data, get_argument_list<tags>{}),
         false)...};
    reset_compute_items_after_mutate<
        tmpl::size<tags_dependent_on_current_tags>::value == 0, full_edge_list,
        tags_dependent_on_current_tags>::apply(data);
  }
};

template <class full_edge_list, template <class...> class F>
struct reset_compute_items_after_mutate<true, full_edge_list, F<>> {
  template <typename... Tags>
  SPECTRE_ALWAYS_INLINE static constexpr void apply(
      databox_detail::TaggedDeferredTuple<Tags...>& /*data*/) noexcept {}
};

template <class VariablesTag>
struct get_tags_list {
  using type = typename db::item_type<VariablesTag>::tags_list;
};
}  // namespace databox_detail

/*!
 * \ingroup DataBoxGroup
 * \brief Allows changing the state of one or more non-computed elements in
 * the DataBox
 *
 * `mutate()`'s first argument is the DataBox from which to retrieve the tags
 * `MutateTags`. The objects corresponding to the `MutateTags` are then passed
 * to `invokable`, which is a lambda or a function object taking as many
 * arguments as there are `MutateTags` and with the arguments being of types
 * `db::item_type<MutateTags>...`. Inside the `invokable` no items can be
 * retrieved from the DataBox `box`. This is to avoid confusing subtleties with
 * order of evaluation of compute items, as well as dangling references. If an
 * `invokable` needs read access to items in `box` they should be passed as
 * additional arguments to `mutate`. Capturing them by reference in a lambda
 * does not work because of a bug in GCC 6.3 and earlier. For a function object
 * the read-only items can also be stored as const references inside the object
 * by passing `db::get<TAG>(t)` to the constructor.
 *
 * \example
 * \snippet Test_DataBox.cpp databox_mutate_example
 */
template <typename... MutateTags, typename TagList, typename Invokable,
          typename... Args>
void mutate(DataBox<TagList>& box, Invokable&& invokable,
            Args&&... args) noexcept {
  using mutate_tags_list = tmpl::list<MutateTags...>;
  static_assert(not tmpl2::flat_any_v<db::is_compute_item_v<MutateTags>...>,
                "Cannot mutate a compute item");
  if (UNLIKELY(box.mutate_locked_box_)) {
    ERROR(
        "Unable to mutate a DataBox that is already being mutated. This "
        "error occurs when mutating a DataBox from inside the invokable "
        "passed to the mutate function.");
  }
  box.mutate_locked_box_ = true;
  invokable(::db::databox_detail::get<MutateTags>(box.data_).mutate()...,
            std::forward<Args>(args)...);
  // For all the variable tags in the DataBox, check if one of their Tags is
  // being mutated and if so add it to the list of tags being mutated. Then,
  // remove any Variables tags that would be passed multiple times.
  using variables_tags = tmpl::filter<
      tmpl::filter<
          typename DataBox<TagList>::variables_item_tags,
          tmpl::bind<
              tmpl::found, databox_detail::get_tags_list<tmpl::_1>,
              tmpl::bind<tmpl::found, tmpl::pin<tmpl::list<MutateTags...>>,
                         tmpl::bind<std::is_same, tmpl::pin<tmpl::_1>,
                                    tmpl::parent<tmpl::_1>>>>>,
      tmpl::bind<tmpl::not_found, tmpl::pin<mutate_tags_list>,
                 tmpl::bind<std::is_same, tmpl::bind<tmpl::pin, tmpl::_1>,
                            tmpl::parent<tmpl::_1>>>>;

  // Since MutateTags could contain Variables Tags we need to extract the Tags
  // inside the Variables class and reset compute items depending on those too.
  using mutated_items_including_variables = tmpl::append<
      typename databox_detail::closure_extract_dependent_items<tt::is_a_v<
          Variables, item_type<MutateTags>>>::template f<MutateTags>...,
      variables_tags>;
  using first_compute_items_to_reset = tmpl::transform<
      tmpl::filter<
          typename DataBox<TagList>::edge_list,
          tmpl::bind<
              tmpl::found, tmpl::pin<mutated_items_including_variables>,
              tmpl::bind<std::is_same,
                         tmpl::bind<tmpl::pin, tmpl::get_source<tmpl::_1>>,
                         tmpl::parent<tmpl::_1>>>>,
      tmpl::get_destination<tmpl::_1>>;
  databox_detail::reset_compute_items_after_mutate<
      tmpl::size<first_compute_items_to_reset>::value == 0,
      typename DataBox<TagList>::edge_list,
      first_compute_items_to_reset>::apply(box.data_);

  box.mutate_locked_box_ = false;
}

/*!
 * \ingroup DataBoxGroup
 * \requires Type `T` is one of the Tags corresponding to an object stored in
 * the DataBox
 *
 * \return The object corresponding to the Tag `T`
 */
template <typename T, typename TagList>
SPECTRE_ALWAYS_INLINE auto get(const DataBox<TagList>& t) noexcept
    -> const tmpl::conditional_t<cpp17::is_same_v<T, ::Tags::DataBox>,
                                 DataBox<TagList>, item_type<T>>& {
  return t.template get_impl<T>();
}

template <template <typename...> class TagsList, typename... Tags>
template <typename Tag, Requires<not cpp17::is_same_v<Tag, ::Tags::DataBox>>>
inline const db::item_type<Tag>& DataBox<TagsList<Tags...>>::get_impl() const
    noexcept {
  if (UNLIKELY(mutate_locked_box_)) {
    ERROR("Unable to retrieve a (compute) item '"
          << Tag::label
          << "' from the DataBox from within a "
             "call to mutate. You must pass these either through the capture "
             "list of the lambda or the constructor of a class, this "
             "restriction exists to avoid complexity.");
  }
  return ::db::databox_detail::get<Tag>(data_).get();
}

template <template <typename...> class TagsList, typename... Tags>
template <typename Tag, Requires<cpp17::is_same_v<Tag, ::Tags::DataBox>>>
inline const DataBox<TagsList<Tags...>>& DataBox<TagsList<Tags...>>::get_impl()
    const noexcept {
  if (UNLIKELY(mutate_locked_box_)) {
    ERROR(
        "Unable to retrieve a (compute) item 'DataBox' from the DataBox from "
        "within a call to mutate. You must pass these either through the "
        "capture list of the lambda or the constructor of a class, this "
        "restriction exists to avoid complexity.");
  }
  return *this;
}

template <template <typename...> class TagsList, typename... Tags>
template <
    typename... TagsInArgsOrder, typename... FullItems,
    typename... ComputeItemTags, typename... FullComputeItems, typename... Args,
    Requires<not tmpl2::flat_any_v<is_databox<std::decay_t<Args>>::value...>>>
constexpr DataBox<TagsList<Tags...>>::DataBox(
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
      tmpl2::flat_all_v<cpp17::is_same_v<typename TagsInArgsOrder::type,
                                         std::decay_t<Args>>...>,
      "The type of each Tag must be the same as the type being passed into "
      "the function creating the new DataBox.");

  std::tuple<Args...> args_tuple(std::forward<Args>(args)...);
  databox_detail::add_items_to_box<typelist<FullItems..., FullComputeItems...>>(
      args_tuple, data_, typelist<TagsInArgsOrder...>{},
      std::make_index_sequence<sizeof...(TagsInArgsOrder)>{},
      typelist<ComputeItemTags...>{});
}

template <template <typename...> class TagsList, typename... Tags>
template <typename OldTags, typename... KeepTags, typename... NewTags,
          typename... NewComputeItems, typename ComputeItemsToKeep,
          typename... MutatedTags, typename... Args>
constexpr DataBox<TagsList<Tags...>>::DataBox(
    const DataBox<OldTags>& old_box, typelist<KeepTags...> /*meta*/,
    typelist<NewTags...> /*meta*/, typelist<NewComputeItems...> /*meta*/,
    ComputeItemsToKeep /*meta*/, tmpl::list<MutatedTags...> /*meta*/,
    Args&&... args) {
  static_assert(sizeof...(NewTags) == sizeof...(Args),
                "Must pass in as many arguments as AddTags");
  static_assert(
      tmpl2::flat_all_v<
          cpp17::is_same_v<typename NewTags::type, std::decay_t<Args>>...>,
      "The type of each Tag must be the same as the type being passed into "
      "the function creating the new DataBox.");

  check_tags();
  // Merge old tags, including all ComputeItems even though they might be
  // reset.
  databox_detail::merge_old_box(old_box, data_, typelist<KeepTags...>{});

  std::tuple<Args...> args_tuple(std::forward<Args>(args)...);
  databox_detail::add_items_to_box<typelist<NewTags...>>(
      args_tuple, data_, typelist<NewTags...>{},
      std::make_index_sequence<sizeof...(NewTags)>{}, typelist<>{});

  // For all the variable tags in the DataBox, check if one of their Tags is
  // being mutated. We do not allow this to be done in create_from so it results
  // in an error
  using mutated_tags_list = tmpl::list<MutatedTags...>;
  using variables_tags_from_single_tags = tmpl::filter<
      extracted_from_variables,
      tmpl::bind<tmpl::found, tmpl::pin<mutated_tags_list>,
                 tmpl::bind<std::is_same, tmpl::_1, tmpl::parent<tmpl::_1>>>>;
  static_assert(cpp17::is_same_v<tmpl::list<>, variables_tags_from_single_tags>,
                "You are not allowed to mutate a single component of a "
                "Variables with db::create_from, you must mutate the entire "
                "Variables instead.");

  // Reset dependent compute items
  using mutated_items_including_variables = tmpl::append<
      typename databox_detail::closure_extract_dependent_items<tt::is_a_v<
          Variables, item_type<MutatedTags>>>::template f<MutatedTags>...>;
  using first_compute_items_to_reset = tmpl::transform<
      tmpl::filter<
          edge_list,
          tmpl::bind<
              tmpl::found, tmpl::pin<mutated_items_including_variables>,
              tmpl::bind<std::is_same,
                         tmpl::bind<tmpl::pin, tmpl::get_source<tmpl::_1>>,
                         tmpl::parent<tmpl::_1>>>>,
      tmpl::get_destination<tmpl::_1>>;

  databox_detail::reset_compute_items_after_mutate<
      tmpl::size<first_compute_items_to_reset>::value == 0, edge_list,
      first_compute_items_to_reset>::apply(data_);

  // Add new compute items
  databox_detail::add_items_to_box<
      typelist<KeepTags..., NewTags..., NewComputeItems...>>(
      args_tuple, data_, typelist<>{}, std::make_index_sequence<0>{},
      typelist<NewComputeItems...>{});
}

template <template <typename...> class TagsList, typename... Tags>
template <typename AddTags, typename AddComputeItems, typename... Args>
constexpr auto DataBox<TagsList<Tags...>>::create(Args&&... args) {
  static_assert(tt::is_a_v<tmpl::list, AddComputeItems>,
                "AddComputeItems must by a typelist");
  static_assert(tt::is_a_v<tmpl::list, AddTags>, "AddTags must by a typelist");
  static_assert(
      not tmpl::any<AddTags, is_compute_item<tmpl::_1>>::value,
      "Cannot add any ComputeItemTag in the AddTags list, must use the "
      "AddComputeItems list.");
  static_assert(tmpl::all<AddComputeItems, is_compute_item<tmpl::_1>>::value,
                "Cannot add any Tags in the AddComputeItems list, must use the "
                "AddTags list.");
  using full_items = tmpl::fold<
      AddTags, tmpl::list<>,
      tmpl::bind<tmpl::append, tmpl::_state,
                 databox_detail::extract_dependent_items<tmpl::_element>>>;
  using full_compute_items = tmpl::fold<
      AddComputeItems, tmpl::list<>,
      tmpl::bind<tmpl::append, tmpl::_state,
                 databox_detail::extract_dependent_items<tmpl::_element>>>;
  using sorted_tags =
      ::db::get_databox_list<tmpl::append<full_items, full_compute_items>>;
  return DataBox<sorted_tags>(AddTags{}, full_items{}, AddComputeItems{},
                              full_compute_items{},
                              std::forward<Args>(args)...);
}

template <template <typename...> class TagsList, typename... Tags>
template <typename RemoveTags, typename AddTags, typename AddComputeItems,
          typename Box, typename... Args>
constexpr auto DataBox<TagsList<Tags...>>::create_from(const Box& box,
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
  using full_remove_tags = databox_detail::extracted_items<RemoveTags>;
  using full_items = databox_detail::extracted_items<AddTags>;
  using full_compute_items = databox_detail::extracted_items<AddComputeItems>;
  using remaining_tags =
      tmpl::fold<full_remove_tags, old_tags_list,
                 tmpl::bind<tmpl::remove, tmpl::_state, tmpl::_element>>;
  using new_tags = tmpl::append<remaining_tags, full_items, full_compute_items>;
  using sorted_tags = ::db::get_databox_list<new_tags>;
  // List of tags that were both removed and added, and therefore were mutated
  using mutated_tags =
      tmpl::filter<AddTags,
                   tmpl::bind<tmpl::found, tmpl::pin<RemoveTags>,
                              tmpl::bind<std::is_same, tmpl::pin<tmpl::_1>,
                                         tmpl::parent<tmpl::_1>>>>;
  return DataBox<sorted_tags>(box, remaining_tags{}, AddTags{},
                              AddComputeItems{}, compute_items_to_keep{},
                              mutated_tags{}, std::forward<Args>(args)...);
}
/// \endcond

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
 * \see create_from
 *
 * \tparam AddTags the tags of the args being added
 * \tparam AddComputeItems list of \ref ComputeItemTag "compute item tags"
 * to add to the DataBox
 *  \param args the data to be added to the DataBox
 */
template <typename AddTags, typename AddComputeItems = typelist<>,
          typename... Args>
SPECTRE_ALWAYS_INLINE constexpr auto create(Args&&... args) {
  return DataBox<::db::get_databox_list<tmpl::append<
      tmpl::fold<
          AddTags, tmpl::list<>,
          tmpl::bind<tmpl::append, tmpl::_state,
                     databox_detail::extract_dependent_items<tmpl::_element>>>,
      tmpl::fold<AddComputeItems, tmpl::list<>,
                 tmpl::bind<tmpl::append, tmpl::_state,
                            databox_detail::extract_dependent_items<
                                tmpl::_element>>>>>>::
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
 * \snippet Test_DataBox.cpp create_from_add_compute_item
 *
 * \see create DataBox
 *
 * \tparam RemoveTags typelist of Tags to remove
 * \tparam AddTags typelist of Tags corresponding to the arguments to be
 * added
 * \tparam AddComputeItems list of \ref ComputeItemTag "compute item tags"
 * to add to the DataBox
 * \param box the DataBox the new box should be based off
 * \param args the values for the items to add to the DataBox
 * \return DataBox like `box` but altered by RemoveTags and AddTags
 */
template <typename RemoveTags, typename AddTags = typelist<>,
          typename AddComputeItems = typelist<>, typename Box, typename... Args>
SPECTRE_ALWAYS_INLINE constexpr auto create_from(const Box& box,
                                                 Args&&... args) {
  return DataBox<::db::get_databox_list<tmpl::append<
      tmpl::fold<
          tmpl::fold<RemoveTags, tmpl::list<>,
                     tmpl::lazy::append<tmpl::_state,
                                        databox_detail::extract_dependent_items<
                                            tmpl::_element>>>,
          typename Box::tags_list,
          tmpl::bind<tmpl::remove, tmpl::_state, tmpl::_element>>,
      tmpl::fold<tmpl::append<AddTags, AddComputeItems>, tmpl::list<>,
                 tmpl::bind<tmpl::append, tmpl::_state,
                            databox_detail::extract_dependent_items<
                                tmpl::_element>>>>>>::
      template create_from<RemoveTags, AddTags, AddComputeItems>(
          box, std::forward<Args>(args)...);
}

namespace detail {
template <typename Tag, typename TagList, typename T>
constexpr void get_item_from_box_helper(const DataBox<TagList>& box,
                                        const std::string& tag_name,
                                        T const** result) {
  if (get_tag_name<Tag>() == tag_name) {
    *result = &::db::get<Tag>(box);
  }
}

template <typename Type, typename... Tags, typename TagList>
const Type& get_item_from_box(const DataBox<TagList>& box,
                              const std::string& tag_name,
                              tmpl::list<Tags...> /*meta*/) {
  static_assert(sizeof...(Tags) != 0,
                "No items with the requested type were found in the DataBox");
  Type const* result = nullptr;
  static_cast<void>(std::initializer_list<char>{
      (get_item_from_box_helper<Tags>(box, tag_name, &result), '0')...});
  if (result == nullptr) {
    std::stringstream tags_in_box;
    tmpl::for_each<TagList>([&tags_in_box](auto temp) {
      tags_in_box << "  " << decltype(temp)::type::label << "\n";
    });
    ERROR("Could not find the tag named \""
          << tag_name << "\" in the DataBox. Available tags are:\n"
          << tags_in_box.str());
  }
  return *result;
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
template <typename Type, typename TagList>
constexpr const Type& get_item_from_box(const DataBox<TagList>& box,
                                        const std::string& tag_name) {
  using tags = tmpl::filter<
      TagList, std::is_same<tmpl::bind<item_type, tmpl::_1>, tmpl::pin<Type>>>;
  return detail::get_item_from_box<Type>(box, tag_name, tags{});
}

namespace detail {
template <typename TagsList>
struct Apply;

template <template <typename...> class TagsList, typename... Tags>
struct Apply<TagsList<Tags...>> {
  template <typename F, typename... BoxTags, typename... Args>
  static constexpr auto apply(F f, const DataBox<BoxTags...>& box,
                              Args&&... args) {
    static_assert(
        tt::is_callable_v<
            std::remove_pointer_t<F>,
            tmpl::conditional_t<cpp17::is_same_v<Tags, ::Tags::DataBox>,
                                const DataBox<BoxTags...>&, item_type<Tags>>...,
            Args...>,
        "Cannot call the function f with the list of tags and "
        "arguments specified. Check that the Tags::type and the "
        "types of the Args match the function f.");
    return f(::db::get<Tags>(box)..., std::forward<Args>(args)...);
  }

  template <typename F, typename... BoxTags, typename... Args>
  static constexpr auto apply_with_box(F f, const DataBox<BoxTags...>& box,
                                       Args&&... args) {
    static_assert(
        tt::is_callable_v<
            F, DataBox<BoxTags...>,
            tmpl::conditional_t<cpp17::is_same_v<Tags, ::Tags::DataBox>,
                                const DataBox<BoxTags...>&, item_type<Tags>>...,
            Args...>,
        "Cannot call the function f with the list of tags and "
        "arguments specified. Check that the Tags::type and the "
        "types of the Args match the function f and that f is "
        "receiving the correct type of DataBox.");
    return f(box, ::db::get<Tags>(box)..., std::forward<Args>(args)...);
  }
};
}  // namespace detail

/*!
 * \ingroup DataBoxGroup
 * \brief Apply the function `f` with argument Tags `TagList` from DataBox `box`
 *
 * \details
 * Apply the function `f` with arguments that are of type `Tags::type` where
 * `Tags` is defined as `TagList<Tags...>`. The arguments to `f` are retrieved
 * from the DataBox `box`.
 *
 * \usage
 * Given a function `func` that takes arguments of types
 * `T1`, `T2`, `A1` and `A2`. Let the Tags for the quantities of types `T1` and
 * `T2` in the DataBox `box` be `Tag1` and `Tag2`, and objects `a1` of type
 * `A1` and `a2` of type `A2`, then
 * \code
 * auto result = apply<typelist<Tag1, Tag2>>(func, box, a1, a2);
 * \endcode
 * \return `decltype(func(box.get<Tag1>(), box.get<Tag2>(), a1, a2))`
 *
 * \semantics
 * For tags `Tags...` in a DataBox `box`, and a function `func` that takes
 * `sizeof...(Tags)` arguments of types `typename Tags::type...`,  and
 * `sizeof...(Args)` arguments of types `Args...`,
 * \code
 * result = func(box, box.get<Tags>()..., args...);
 * \endcode
 *
 * \example
 * \snippet Test_DataBox.cpp apply_example
 *
 * \see apply_with_box DataBox
 * \tparam TagsList typelist of Tags in the order that they are to be passed
 * to `f`
 * \param f the function to apply
 * \param box the DataBox out of which to retrieve the Tags and to pass to `f`
 * \param args the arguments to pass to the function that are not in the
 * DataBox, `box`
 */
template <typename TagsList, typename F, typename... BoxTags, typename... Args>
inline constexpr auto apply(F f, const DataBox<BoxTags...>& box,
                            Args&&... args) {
  return detail::Apply<TagsList>::apply(f, box, std::forward<Args>(args)...);
}

namespace databox_detail {
CREATE_IS_CALLABLE(apply)

template <typename... ReturnTags, typename... ArgumentTags, typename F,
          typename... BoxTags, typename... Args,
          Requires<is_apply_callable_v<
              F, const gsl::not_null<db::item_type<ReturnTags>*>...,
              const std::add_lvalue_reference_t<db::item_type<ArgumentTags>>...,
              Args...>> = nullptr>
inline constexpr auto mutate_apply(F /*f*/, db::DataBox<BoxTags...>& box,
                                   tmpl::list<ReturnTags...> /*meta*/,
                                   tmpl::list<ArgumentTags...> /*meta*/,
                                   Args&&... args)
    // clang-format off
    noexcept(noexcept(F::apply(
        std::declval<gsl::not_null<db::item_type<ReturnTags>*>>()...,
        std::declval<const db::item_type<ArgumentTags>&>()...,
        std::forward<Args>(args)...))) {
  // clang-format on
  static_assert(
      not tmpl2::flat_any_v<
          cpp17::is_same_v<ArgumentTags, Tags::DataBox>...> and
          not tmpl2::flat_any_v<cpp17::is_same_v<ReturnTags, Tags::DataBox>...>,
      "Cannot pass a DataBox to mutate_apply since the db::get won't work "
      "inside mutate_apply.");
  ::db::mutate<ReturnTags...>(
      box,
      [](db::item_type<ReturnTags> & ... mutated_items,
         const db::item_type<ArgumentTags>&... args_items,
         decltype(std::forward<Args>(args))... l_args)
      // clang-format off
      noexcept(noexcept(F::apply(
          std::declval<gsl::not_null<db::item_type<ReturnTags>*>>()...,
          std::declval<const db::item_type<ArgumentTags>&>()...,
          std::forward<Args>(args)...))) {
        // clang-format on
        return F::apply(make_not_null(&mutated_items)..., args_items...,
                        std::forward<Args>(l_args)...);
      },
      db::get<ArgumentTags>(box)..., std::forward<Args>(args)...);
}

template <typename... ReturnTags, typename... ArgumentTags, typename F,
          typename... BoxTags, typename... Args,
          Requires<::tt::is_callable_v<
              F, const gsl::not_null<db::item_type<ReturnTags>*>...,
              const std::add_lvalue_reference_t<db::item_type<ArgumentTags>>...,
              Args...>> = nullptr>
inline constexpr auto mutate_apply(F f, db::DataBox<BoxTags...>& box,
                                   tmpl::list<ReturnTags...> /*meta*/,
                                   tmpl::list<ArgumentTags...> /*meta*/,
                                   Args&&... args)
    // clang-format off
    noexcept(noexcept(f(
        std::declval<gsl::not_null<db::item_type<ReturnTags>*>>()...,
        std::declval<const db::item_type<ArgumentTags>&>()...,
        std::forward<Args>(args)...))) {
  // clang-format on
  static_assert(
      not tmpl2::flat_any_v<
          cpp17::is_same_v<ArgumentTags, Tags::DataBox>...> and
          not tmpl2::flat_any_v<cpp17::is_same_v<ReturnTags, Tags::DataBox>...>,
      "Cannot pass a DataBox to mutate_apply since the db::get won't work "
      "inside mutate_apply.");
  ::db::mutate<ReturnTags...>(
      box,
      [&f](db::item_type<ReturnTags> & ... mutated_items,
           const db::item_type<ArgumentTags>&... args_items,
           decltype(std::forward<Args>(args))... l_args)
      // clang-format off
      noexcept(noexcept(f(make_not_null(&mutated_items)...,
                          args_items..., std::forward<Args>(
                              l_args)...)))
      // clang-format on
      {
        return f(make_not_null(&mutated_items)..., args_items...,
                 std::forward<Args>(l_args)...);
      },
      db::get<ArgumentTags>(box)..., std::forward<Args>(args)...);
}
}  // namespace databox_detail

/*!
 * \ingroup DataBoxGroup
 * \brief Apply the function `f` mutating items `MutateTags` and taking as
 * additional arguments `ArgumentTags` and `args`.
 *
 * \details
 * `f` must either by invokable with the arguments of type
 * `db::item_type<ReturnTags>..., db::item_type<ArgumentTags>..., Args...` where
 * the first two pack expansions are over the elements in the type lists
 * `MutateTags` and `ArgumentTags`, or have a static `apply` function  that is
 * callable with the same types.
 *
 * \example
 * An example of using `mutate_apply` with a lambda:
 * \snippet Test_DataBox.cpp mutate_apply_example
 *
 * An example of a class with a static `apply` function
 *
 * \snippet Test_DataBox.cpp mutate_apply_apply_struct_example
 * and how to use `mutate_apply` with the class
 * \snippet Test_DataBox.cpp mutate_apply_apply_example
 *
 * \see apply apply_with_box
 * \tparam MutateTags typelist of Tags to mutate
 * \tparam ArgumentTags typelist of additional items to retrieve from the
 * DataBox
 * \param f the function to apply
 * \param box the DataBox out of which to retrieve the Tags and to pass to `f`
 * \param args the arguments to pass to the function that are not in the
 * DataBox, `box`
 */
template <typename MutateTags, typename ArgumentTags, typename F,
          typename... BoxTags, typename... Args>
inline constexpr auto
mutate_apply(F f, DataBox<BoxTags...>& box, Args&&... args) noexcept(
    noexcept(databox_detail::mutate_apply(f, box, MutateTags{}, ArgumentTags{},
                                          std::forward<Args>(args)...))) {
  return databox_detail::mutate_apply(f, box, MutateTags{}, ArgumentTags{},
                                      std::forward<Args>(args)...);
}

/*!
 * \ingroup DataBoxGroup
 * \brief Apply the function `f` with argument Tags `TagList` from DataBox `box`
 * and `box` as the first argument
 *
 * \details
 * Apply the function `f` with arguments that are of type `Tags::type` where
 * `Tags` is defined as `TagList<Tags...>`. The arguments to `f` are retrieved
 * from the DataBox `box` and the first argument passed to `f` is the DataBox.
 *
 * \usage
 * Given a function `func` that takes arguments of types `DataBox<Tags...>`,
 * `T1`, `T2`, `A1` and `A2`. Let the Tags for the quantities of types `T1`
 * and `T2` in the DataBox `box` be `Tag1` and `Tag2`, and objects `a1` of type
 * `A1` and `a2` of type `A2`, then
 * \code
 * auto result = apply_with_box<typelist<Tag1, Tag2>>(func, box, a1, a2);
 * \endcode
 * \return `decltype(func(box, box.get<Tag1>(), box.get<Tag2>(), a1, a2))`
 *
 * \semantics
 * For tags `Tags...` in a DataBox `box`, and a function `func` that takes
 * as its first argument a value of type`decltype(box)`,
 * `sizeof...(Tags)` arguments of types `typename Tags::type...`, and
 * `sizeof...(Args)` arguments of types `Args...`,
 * \code
 * result = func(box, box.get<Tags>()..., args...);
 * \endcode
 *
 * \example
 * \snippet Test_DataBox.cpp apply_with_box_example
 *
 * \see apply DataBox
 * \tparam TagsList typelist of Tags in the order that they are to be passed
 * to `f`
 * \param f the function to apply
 * \param box the DataBox out of which to retrieve the Tags and to pass to
 * f`
 * \param args the arguments to pass to the function that are not in the
 * DataBox, `box`
 */
template <typename TagsList, typename F, typename... BoxTags, typename... Args>
inline constexpr auto apply_with_box(F f, const DataBox<BoxTags...>& box,
                                     Args&&... args) {
  return detail::Apply<TagsList>::apply_with_box(f, box,
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
 * \metareturns a typelist of tags that need to be removed from the DataBox
 * with tags `DataBoxTagsList` in order to keep the tags in `KeepTagsList`
 *
 * \example
 * \snippet Test_DataBox.cpp remove_tags_from_keep_tags
 */
template <typename DataBoxTagsList, typename KeepTagsList>
using remove_tags_from_keep_tags =
    tmpl::filter<DataBoxTagsList,
                 detail::filter_helper<tmpl::pin<KeepTagsList>, tmpl::_1>>;

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

namespace databox_detail {
template <class ItemsList, class ComputeItemsList>
struct compute_dbox_type;

template <class... ItemsPack, class ComputeItemsList>
struct compute_dbox_type<tmpl::list<ItemsPack...>, ComputeItemsList> {
  using type = decltype(db::create<tmpl::list<ItemsPack...>, ComputeItemsList>(
      std::declval<db::item_type<ItemsPack>>()...));
};
}  // namespace databox_detail

/*!
 * \ingroup DataBoxGroup
 * \brief Returns the type of the DataBox that would be constructed from the
 * `TagList` of tags.
 */
template <class TagList>
using compute_databox_type = typename databox_detail::compute_dbox_type<
    get_items<TagList>, get_compute_items<TagList>>::type;
}  // namespace db
