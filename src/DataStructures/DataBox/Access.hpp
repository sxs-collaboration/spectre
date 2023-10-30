// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/TagName.hpp"
#include "Utilities/CleanupRoutine.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

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
}  // namespace db
