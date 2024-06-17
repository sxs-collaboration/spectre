// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>
#include <variant>

#include "Options/String.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Options {
template <typename... AlternativeLists>
struct Alternatives;
}  // namespace Options
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// \ingroup DataStructuresGroup
///
/// TaggedVariant and related functionality.
namespace variants {
namespace TaggedVariant_detail {
template <typename R, typename Variant, typename Tag, typename Visitor>
struct VisitAlternative;
}  // namespace TaggedVariant_detail

/// \cond
template <typename... Tags>
class TaggedVariant;
template <typename... Tags>
constexpr bool operator==(const TaggedVariant<Tags...>& a,
                          const TaggedVariant<Tags...>& b);
template <typename... Tags>
constexpr bool operator<(const TaggedVariant<Tags...>& a,
                         const TaggedVariant<Tags...>& b);
/// \endcond

/// \ingroup DataStructuresGroup
///
/// A class similar to `std::variant`, but indexed by tag structs.
///
/// \see variants::get, variants::get_if, variants::holds_alternative,
/// variants::visit
template <typename... Tags>
class TaggedVariant {
 private:
  static_assert(sizeof...(Tags) > 0);
  static_assert(
      std::is_same_v<tmpl::remove_duplicates<TaggedVariant>, TaggedVariant>,
      "TaggedVariant cannot have duplicate tags.");

  template <typename Tag>
  static constexpr size_t data_index =
      tmpl::index_of<TaggedVariant, Tag>::value;

 public:
  /// A default constructed instance has the first tag active.
  TaggedVariant() = default;
  TaggedVariant(const TaggedVariant&) = default;
  TaggedVariant(TaggedVariant&&) = default;
  TaggedVariant& operator=(const TaggedVariant&) = default;
  TaggedVariant& operator=(TaggedVariant&&) = default;
  ~TaggedVariant() = default;

  /// Construct with \p Tag active, using \p args to construct the
  /// contained object.
  ///
  /// \snippet DataStructures/Test_TaggedVariant.cpp construct in_place_type
  template <
      typename Tag, typename... Args,
      Requires<(... or std::is_same_v<Tag, Tags>) and
               std::is_constructible_v<typename Tag::type, Args...>> = nullptr>
  constexpr explicit TaggedVariant(std::in_place_type_t<Tag> /*meta*/,
                                   Args&&... args)
      : data_(std::in_place_index<data_index<Tag>>,
              std::forward<Args>(args)...) {}

  /// Construct the contained object from \p args.  Only available if
  /// the TaggedVariant only has one tag.
  ///
  /// \snippet DataStructures/Test_TaggedVariant.cpp construct single
  template <typename... Args,
            Requires<sizeof...(Tags) == 1 and
                     std::is_constructible_v<
                         typename tmpl::front<TaggedVariant>::type, Args...>> =
                nullptr>
  constexpr explicit TaggedVariant(Args&&... args)
      : TaggedVariant(std::in_place_type<tmpl::front<TaggedVariant>>,
                      std::forward<Args>(args)...) {}

  /// A TaggedVariant can be implicitly move-converted to another
  /// variant with a superset of the tags.
  ///
  /// \snippet DataStructures/Test_TaggedVariant.cpp convert
  /// @{
  template <typename... OtherTags,
            Requires<tmpl::size<tmpl::list_difference<
                         TaggedVariant<OtherTags...>, TaggedVariant>>::value ==
                     0> = nullptr>
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr TaggedVariant(TaggedVariant<OtherTags...>&& other);

  template <typename... OtherTags,
            Requires<tmpl::size<tmpl::list_difference<
                         TaggedVariant<OtherTags...>, TaggedVariant>>::value ==
                     0> = nullptr>
  constexpr TaggedVariant& operator=(TaggedVariant<OtherTags...>&& other);
  /// @}

  /// The index into the `Tags...` of the active object.
  constexpr size_t index() const { return data_.index(); }

  /// See `std::variant::valueless_by_exception`.
  constexpr bool valueless_by_exception() const {
    return data_.valueless_by_exception();
  }

  /// Destroys the contained object and actives \p Tag, constructing a
  /// new value from \p args.
  ///
  /// \snippet DataStructures/Test_TaggedVariant.cpp emplace
  template <
      typename Tag, typename... Args,
      Requires<(... or std::is_same_v<Tag, Tags>) and
               std::is_constructible_v<typename Tag::type, Args...>> = nullptr>
  constexpr typename Tag::type& emplace(Args&&... args) {
    return data_.template emplace<data_index<Tag>>(std::forward<Args>(args)...);
  }

  constexpr void swap(TaggedVariant& other) noexcept(noexcept(
      (... and (std::is_nothrow_move_constructible_v<typename Tags::type> and
                std::is_nothrow_swappable_v<typename Tags::type>)))) {
    data_.swap(other.data_);
  }

  void pup(PUP::er& p) { p | data_; }

  /// A TaggedVariant over option tags can be parsed as any of them.
  /// @{
  static constexpr Options::String help = "One of multiple options";
  using options = tmpl::list<Options::Alternatives<tmpl::list<Tags>...>>;
  template <typename Tag>
  explicit TaggedVariant(tmpl::list<Tag> /*meta*/, typename Tag::type value)
      : TaggedVariant(std::in_place_type<Tag>, std::move(value)) {}
  /// @}

 private:
  template <typename R, typename Variant, typename Tag, typename Visitor>
  friend struct TaggedVariant_detail::VisitAlternative;

  template <typename Tag, typename... Tags2>
  friend constexpr typename Tag::type& get(TaggedVariant<Tags2...>& variant);
  template <typename Tag, typename... Tags2>
  friend constexpr const typename Tag::type& get(
      const TaggedVariant<Tags2...>& variant);
  template <typename Tag, typename... Tags2>
  friend constexpr typename Tag::type&& get(TaggedVariant<Tags2...>&& variant);
  template <typename Tag, typename... Tags2>
  friend constexpr const typename Tag::type&& get(
      const TaggedVariant<Tags2...>&& variant);

  friend constexpr bool operator== <Tags...>(const TaggedVariant<Tags...>& a,
                                             const TaggedVariant<Tags...>& b);
  friend constexpr bool operator< <Tags...>(const TaggedVariant<Tags...>& a,
                                            const TaggedVariant<Tags...>& b);

  friend struct std::hash<TaggedVariant>;

  std::variant<typename Tags::type...> data_;
};

namespace TaggedVariant_detail {
template <typename... Tags>
constexpr bool is_variant_or_derived(const TaggedVariant<Tags...>* /*meta*/) {
  return true;
}
// NOLINTNEXTLINE(cert-dcl50-cpp) - variadic function
constexpr bool is_variant_or_derived(...) { return false; }

template <typename Tag, typename Value>
constexpr std::pair<tmpl::type_<Tag>, Value&&> make_visitor_pair(
    Value&& value) {
  return {tmpl::type_<Tag>{}, std::forward<Value>(value)};
}

struct DeduceReturn;

template <typename R, typename Variant, typename Tag, typename Visitor>
struct VisitAlternative {
  static constexpr R apply(Variant&& variant, const Visitor& visitor) {
    return std::forward<Visitor>(visitor)(make_visitor_pair<Tag>(
        get<std::decay_t<Variant>::template data_index<Tag>>(
            std::forward<Variant>(variant).data_)));
  }
};

template <typename Variant, typename Tag, typename Visitor>
struct VisitAlternative<void, Variant, Tag, Visitor> {
  static constexpr void apply(Variant&& variant, const Visitor& visitor) {
    std::forward<Visitor>(visitor)(make_visitor_pair<Tag>(
        get<std::decay_t<Variant>::template data_index<Tag>>(
            std::forward<Variant>(variant).data_)));
  }
};

template <typename Variant, typename Tag, typename Visitor>
struct VisitAlternative<DeduceReturn, Variant, Tag, Visitor> {
  static constexpr decltype(auto) apply(Variant&& variant,
                                        const Visitor& visitor) {
    return std::forward<Visitor>(visitor)(make_visitor_pair<Tag>(
        get<std::decay_t<Variant>::template data_index<Tag>>(
            std::forward<Variant>(variant).data_)));
  }
};

template <typename... Tags>
constexpr TaggedVariant<Tags...>& as_variant(TaggedVariant<Tags...>& variant) {
  return variant;
}
template <typename... Tags>
constexpr const TaggedVariant<Tags...>& as_variant(
    const TaggedVariant<Tags...>& variant) {
  return variant;
}
template <typename... Tags>
constexpr TaggedVariant<Tags...>&& as_variant(
    TaggedVariant<Tags...>&& variant) {
  return std::move(variant);
}
template <typename... Tags>
constexpr const TaggedVariant<Tags...>&& as_variant(
    const TaggedVariant<Tags...>&& variant) {
  return std::move(variant);
}

template <typename R, typename Visitor, typename... Variants>
constexpr decltype(auto) visit_impl(Visitor&& visitor) {
  return std::forward<Visitor>(visitor)();
}

template <typename R, typename Variant, typename Visitor,
          typename DecayedVariant = std::decay_t<Variant>>
struct VisitJumpTable;

template <typename R, typename Variant, typename Visitor, typename... Tags>
struct VisitJumpTable<R, Variant, Visitor, TaggedVariant<Tags...>> {
  static constexpr std::array value{
      VisitAlternative<R, Variant, Tags, Visitor>::apply...};
};

template <typename R, typename Visitor, typename FirstVariant,
          typename... Variants>
constexpr decltype(auto) visit_impl(Visitor&& visitor,
                                    FirstVariant&& first_variant,
                                    Variants&&... variants) {
  const auto recurse = [&]<typename Arg>(Arg&& first_arg) {
    return visit_impl<R>(
        [&]<typename... Rest>(Rest&&... rest) {
          return std::forward<Visitor>(visitor)(std::forward<Arg>(first_arg),
                                                std::forward<Rest>(rest)...);
        },
        std::forward<Variants>(variants)...);
  };
  if (UNLIKELY(first_variant.valueless_by_exception())) {
    throw std::bad_variant_access{};
  }
  return gsl::at(VisitJumpTable<R, FirstVariant, decltype(recurse)>::value,
                 first_variant.index())(
      std::forward<FirstVariant>(first_variant), recurse);
}
}  // namespace TaggedVariant_detail

/// Call \p visitor with the contents of one or more variants.
///
/// Calls \p visitor with the contents of each variant as arguments,
/// passed as `std::pair<tmpl::type_<Tag>, typename Tag::type ref>`,
/// where `Tag` is the active tag of the variant and `ref` is a
/// reference qualifier matching that of the passed variant.
///
/// If the template parameter \p R is supplied, the result is
/// implicitly converted to that type (which may be `void`).
/// Otherwise it is deduced from the return type of \p visitor, which
/// must be the same for all tags in the variant.
///
/// \warning Unlike `visit` for `std::variant`, the types of the
/// visitor arguments do not allow for implicit conversions between
/// reference types.  If the visitor expects, for example,
/// `std::pair<tmpl::type_<Tag>, const typename Tag::type&>`, the caller must
/// ensure that the passed variant is a const lvalue.
///
/// \snippet DataStructures/Test_TaggedVariant.cpp visit
/// @{
template <typename Visitor, typename... Variants,
          Requires<(... and TaggedVariant_detail::is_variant_or_derived(
                                std::add_pointer_t<std::remove_reference_t<
                                    Variants>>{}))> = nullptr>
constexpr decltype(auto) visit(Visitor&& visitor, Variants&&... variants) {
  return TaggedVariant_detail::visit_impl<TaggedVariant_detail::DeduceReturn>(
      visitor,
      TaggedVariant_detail::as_variant(std::forward<Variants>(variants))...);
}

template <typename R, typename Visitor, typename... Variants,
          Requires<(... and TaggedVariant_detail::is_variant_or_derived(
                                std::add_pointer_t<std::remove_reference_t<
                                    Variants>>{}))> = nullptr>
constexpr R visit(Visitor&& visitor, Variants&&... variants) {
  return TaggedVariant_detail::visit_impl<R>(
      visitor,
      TaggedVariant_detail::as_variant(std::forward<Variants>(variants))...);
}
/// @}

/// Check whether \p Tag is active.
template <typename Tag, typename... Tags>
constexpr bool holds_alternative(const TaggedVariant<Tags...>& variant) {
  return variant.index() == tmpl::index_of<TaggedVariant<Tags...>, Tag>::value;
}

/// Access the contained object.  Throws `std::bad_variant_access` if
/// \p Tag is not active.
/// @{
template <typename Tag, typename... Tags>
constexpr typename Tag::type& get(TaggedVariant<Tags...>& variant) {
  return get<TaggedVariant<Tags...>::template data_index<Tag>>(variant.data_);
}
template <typename Tag, typename... Tags>
constexpr const typename Tag::type& get(const TaggedVariant<Tags...>& variant) {
  return get<TaggedVariant<Tags...>::template data_index<Tag>>(variant.data_);
}
template <typename Tag, typename... Tags>
constexpr typename Tag::type&& get(TaggedVariant<Tags...>&& variant) {
  return get<TaggedVariant<Tags...>::template data_index<Tag>>(
      std::move(variant.data_));
}
template <typename Tag, typename... Tags>
constexpr const typename Tag::type&& get(
    const TaggedVariant<Tags...>&& variant) {
  return get<TaggedVariant<Tags...>::template data_index<Tag>>(
      std::move(variant.data_));
}
/// @}

/// Returns a pointer to the contained object if \p variant is a
/// non-null pointer and \p Tag is active.  Otherwise, returns
/// `nullptr`.
/// @{
template <typename Tag, typename... Tags>
constexpr const typename Tag::type* get_if(
    const TaggedVariant<Tags...>* variant) {
  if (variant != nullptr and holds_alternative<Tag>(*variant)) {
    return &get<Tag>(*variant);
  } else {
    return nullptr;
  }
}
template <typename Tag, typename... Tags>
constexpr typename Tag::type* get_if(TaggedVariant<Tags...>* variant) {
  if (variant != nullptr and holds_alternative<Tag>(*variant)) {
    return &get<Tag>(*variant);
  } else {
    return nullptr;
  }
}
/// @}

template <typename... Tags>
constexpr bool operator==(const TaggedVariant<Tags...>& a,
                          const TaggedVariant<Tags...>& b) {
  return a.data_ == b.data_;
}
template <typename... Tags>
constexpr bool operator!=(const TaggedVariant<Tags...>& a,
                          const TaggedVariant<Tags...>& b) {
  return not(a == b);
}
template <typename... Tags>
constexpr bool operator<(const TaggedVariant<Tags...>& a,
                         const TaggedVariant<Tags...>& b) {
  return a.data_ < b.data_;
}
template <typename... Tags>
constexpr bool operator>(const TaggedVariant<Tags...>& a,
                         const TaggedVariant<Tags...>& b) {
  return b < a;
}
template <typename... Tags>
constexpr bool operator<=(const TaggedVariant<Tags...>& a,
                          const TaggedVariant<Tags...>& b) {
  return not(b < a);
}
template <typename... Tags>
constexpr bool operator>=(const TaggedVariant<Tags...>& a,
                          const TaggedVariant<Tags...>& b) {
  return not(a < b);
}

template <
    typename... Tags,
    Requires<(... and (std::is_move_constructible_v<typename Tags::type> and
                       std::is_swappable_v<typename Tags::type>))> = nullptr>
constexpr void swap(TaggedVariant<Tags...>& a,
                    TaggedVariant<Tags...>& b) noexcept(noexcept(a.swap(b))) {
  a.swap(b);
}

template <typename... Tags>
template <
    typename... OtherTags,
    Requires<tmpl::size<tmpl::list_difference<TaggedVariant<OtherTags...>,
                                              TaggedVariant<Tags...>>>::value ==
             0>>
constexpr TaggedVariant<Tags...>::TaggedVariant(
    TaggedVariant<OtherTags...>&& other)
    : TaggedVariant(visit(
          []<typename Tag>(
              std::pair<tmpl::type_<Tag>, typename Tag::type&&> entry) {
            return TaggedVariant(std::in_place_type<Tag>,
                                 std::move(entry.second));
          },
          std::move(other))) {}

template <typename... Tags>
template <
    typename... OtherTags,
    Requires<tmpl::size<tmpl::list_difference<TaggedVariant<OtherTags...>,
                                              TaggedVariant<Tags...>>>::value ==
             0>>
constexpr TaggedVariant<Tags...>& TaggedVariant<Tags...>::operator=(
    TaggedVariant<OtherTags...>&& other) {
  visit(
      [&]<typename Tag>(
          std::pair<tmpl::type_<Tag>, typename Tag::type&&> entry) {
        emplace<Tag>(std::move(entry.second));
      },
      std::move(other));
  return *this;
}
}  // namespace variants

namespace std {
template <typename... Tags>
// https://github.com/llvm/llvm-project/issues/45454
// NOLINTNEXTLINE(cert-dcl58-cpp)
struct hash<::variants::TaggedVariant<Tags...>> {
  size_t operator()(const ::variants::TaggedVariant<Tags...>& variant) const {
    return std::hash<decltype(variant.data_)>{}(variant.data_);
  }
};
}  // namespace std
