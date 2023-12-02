// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/IsApplyCallable.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Utilities/CleanupRoutine.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsCallable.hpp"

namespace db {
/*!
 * \brief A class for retrieving items from a DataBox without needing to know
 * all the tags in the box.
 *
 * Retrieval is handled using a virtual function call but still uses Tags
 * rather than strings to ensure automatic handling of casting to the expected
 * type.
 */
class Access {
 public:
  virtual ~Access() = default;

  /// Print the expanded type aliases of the derived `db::DataBox`
  virtual std::string print_types() const = 0;

 private:
  template <typename Tag>
  friend const auto& get(const Access& box);
  template <typename... MutateTags, typename Invokable, typename... Args>
  friend decltype(auto) mutate(Invokable&& invokable,
                               gsl::not_null<Access*> box, Args&&... args);
  virtual void* mutate_item_by_name(const std::string& tag_name) = 0;
  virtual const void* get_item_by_name(const std::string& tag_name) const = 0;
  virtual bool lock_box_for_mutate() = 0;
  virtual void unlock_box_after_mutate() = 0;
  virtual void mutate_mutable_subitems(const std::string& tag_name) = 0;
  virtual void reset_compute_items_after_mutate(
      const std::string& tag_name) = 0;

  template <typename Tag>
  auto mutate() -> gsl::not_null<typename Tag::type*> {
    static const std::string tag_name = pretty_type::get_name<Tag>();
    return {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        reinterpret_cast<typename Tag::type*>(mutate_item_by_name(tag_name))};
  }
};

/// \brief Retrieve a tag from a `db::Access`
template <typename Tag>
SPECTRE_ALWAYS_INLINE const auto& get(const Access& box) {
  static const std::string tag_name = pretty_type::get_name<Tag>();
  if constexpr (tt::is_a_v<std::unique_ptr, typename Tag::type>) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return *reinterpret_cast<const typename Tag::type::element_type*>(
        box.get_item_by_name(tag_name));
  } else {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return *reinterpret_cast<const typename Tag::type*>(
        box.get_item_by_name(tag_name));
  }
}

/// \brief Mutate a tag in a `db::Access`.
template <typename... MutateTags, typename Invokable, typename... Args>
decltype(auto) mutate(Invokable&& invokable, const gsl::not_null<Access*> box,
                      Args&&... args) {
  static_assert((... and (not is_base_tag_v<MutateTags>)),
                "Cannot mutate base tags with only a db::Access");

  if (UNLIKELY(box->lock_box_for_mutate())) {
    ERROR(
        "Unable to mutate a DataBox that is already being mutated. This "
        "error occurs when mutating a DataBox from inside the invokable "
        "passed to the mutate function.");
  }

  const CleanupRoutine unlock_box = [&box]() {
    box->unlock_box_after_mutate();
    EXPAND_PACK_LEFT_TO_RIGHT([&box]() {
      static const std::string tag_name = pretty_type::get_name<MutateTags>();
      box->mutate_mutable_subitems(tag_name);
      box->reset_compute_items_after_mutate(tag_name);
    }());
  };
  return invokable(box->template mutate<MutateTags>()...,
                   std::forward<Args>(args)...);
}

namespace detail {
template <typename... ReturnTags, typename... ArgumentTags, typename F,
          typename... Args>
SPECTRE_ALWAYS_INLINE constexpr decltype(auto) mutate_apply(
    F&& f, const gsl::not_null<Access*> box, tmpl::list<ReturnTags...> /*meta*/,
    tmpl::list<ArgumentTags...> /*meta*/, Args&&... args) {
  static_assert(not(... or std::is_same_v<ArgumentTags, Tags::DataBox>),
                "Cannot pass Tags::DataBox to mutate_apply when mutating "
                "since the db::get won't work inside mutate_apply.");
  if constexpr (detail::is_apply_callable_v<
                    F, const gsl::not_null<typename ReturnTags::type*>...,
                    const_item_type<ArgumentTags, tmpl::list<>>..., Args...>) {
    return ::db::mutate<ReturnTags...>(
        [](const gsl::not_null<typename ReturnTags::type*>... mutated_items,
           const_item_type<ArgumentTags, tmpl::list<>>... args_items,
           decltype(std::forward<Args>(args))... l_args) {
          return std::decay_t<F>::apply(mutated_items..., args_items...,
                                        std::forward<Args>(l_args)...);
        },
        box, db::get<ArgumentTags>(*box)..., std::forward<Args>(args)...);
  } else if constexpr (::tt::is_callable_v<
                           F,
                           const gsl::not_null<typename ReturnTags::type*>...,
                           const_item_type<ArgumentTags, tmpl::list<>>...,
                           Args...>) {
    return ::db::mutate<ReturnTags...>(f, box, db::get<ArgumentTags>(*box)...,
                                       std::forward<Args>(args)...);
  } else {
    error_function_not_callable<F, gsl::not_null<typename ReturnTags::type*>...,
                                const_item_type<ArgumentTags, tmpl::list<>>...,
                                Args...>();
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
 * `Access`
 * \tparam F The invokable to apply
 */
template <typename MutateTags, typename ArgumentTags, typename F,
          typename... Args>
SPECTRE_ALWAYS_INLINE constexpr decltype(auto) mutate_apply(
    F&& f, const gsl::not_null<Access*> box, Args&&... args) {
  return detail::mutate_apply(std::forward<F>(f), box, MutateTags{},
                              ArgumentTags{}, std::forward<Args>(args)...);
}

template <typename F, typename... Args>
SPECTRE_ALWAYS_INLINE constexpr decltype(auto) mutate_apply(
    F&& f, const gsl::not_null<Access*> box, Args&&... args) {
  return mutate_apply<typename std::decay_t<F>::return_tags,
                      typename std::decay_t<F>::argument_tags>(
      std::forward<F>(f), box, std::forward<Args>(args)...);
}

template <typename F, typename... Args>
SPECTRE_ALWAYS_INLINE constexpr decltype(auto) mutate_apply(
    const gsl::not_null<Access*> box, Args&&... args) {
  return mutate_apply(F{}, box, std::forward<Args>(args)...);
}
/// @}
}  // namespace db
