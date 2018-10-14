// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/SizeOfElement.hpp"

namespace Initialization {

/// \brief Allocate items for minmod limiter
///
/// DataBox changes:
/// - Adds:
///   * `Tags::SizeOfElement<Dim>`
///
/// - Removes: nothing
/// - Modifies: nothing
template <size_t Dim>
struct MinMod {
  using simple_tags = db::AddSimpleTags<>;
  using compute_tags = tmpl::list<Tags::SizeOfElement<Dim>>;
  template <typename TagsList>
  static auto initialize(db::DataBox<TagsList>&& box) noexcept {
    return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
        std::move(box));
  }
};

}  // namespace Initialization
