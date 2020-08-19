// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Utilities/TMPL.hpp"

namespace db {
/// \ingroup DataBoxGroup
/// Struct that can be specialized to allow DataBox items to have
/// subitems.  Specializations must define:
/// * `using type = tmpl::list<...>` listing the subtags of `Tag`
/// * A static member function to initialize a subitem of a simple
///   item:
///   ```
///   template <typename Subtag>
///   static void create_item(
///       const gsl::not_null<item_type<Tag>*> parent_value,
///       const gsl::not_null<item_type<Subtag>*> sub_value) noexcept;
///   ```
///   Mutating the subitems must also modify the main item.
/// * A static member function evaluating a subitem of a compute
///   item:
///   ```
///   template <typename Subtag>
///   static item_type<Subtag> create_compute_item(
///       const item_type<Tag>& parent_value) noexcept;
///   ```
template <typename Tag, typename = std::nullptr_t>
struct Subitems {
  using type = tmpl::list<>;
};
}  // namespace db
