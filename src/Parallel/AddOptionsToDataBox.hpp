// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"

namespace Parallel {
/// \ingroup ParallelGroup
/// An `add_options_to_databox` struct that does not add any options to the
/// DataBox and should be used when a parallel component does not take any input
/// file options.
struct AddNoOptionsToDataBox {
  using simple_tags = tmpl::list<>;

  template <typename DbTagsList, typename... Args>
  static db::DataBox<DbTagsList>&& apply(db::DataBox<DbTagsList>&& box,
                                         Args&&... /*args*/) noexcept {
    return std::move(box);
  }
};

/// Given the tags `SimpleTags`, forwards them into the `DataBox`.
template <typename SimpleTagsList>
struct ForwardAllOptionsToDataBox;

/// \cond
template <typename... SimpleTags>
struct ForwardAllOptionsToDataBox<tmpl::list<SimpleTags...>> {
  using simple_tags = tmpl::list<SimpleTags...>;

  template <typename DbTagsList, typename... Args>
  static auto apply(db::DataBox<DbTagsList>&& box, Args&&... args) noexcept {
    static_assert(
        sizeof...(SimpleTags) == sizeof...(Args),
        "The number of arguments passed to ForwardAllOptionsToDataBox must "
        "match the number of SimpleTags passed.");
    return db::create_from<db::RemoveTags<>, simple_tags>(
        std::move(box), std::forward<Args>(args)...);
  }
};
/// \endcond
}  // namespace Parallel
