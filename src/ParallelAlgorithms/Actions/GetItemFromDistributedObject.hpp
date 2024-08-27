// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/NodeLock.hpp"

namespace Parallel::Actions {
/*!
 * \brief A local synchronous action that returns a pointer to the item
 * specified by the tag.
 *
 * The action uses `db::get_mutable_reference` to avoid DataBox locking
 * interference. However, this means that thread safety with respect to the
 * retrieved tag must be ensured by the user.
 */
template <typename Tag>
struct GetItemFromDistributedOject {
  using return_type = typename Tag::type*;

  template <typename ParallelComponent, typename DbTagList>
  static return_type apply(
      db::DataBox<DbTagList>& box,
      const gsl::not_null<Parallel::NodeLock*> /*node_lock*/) {
    return std::addressof(db::get_mutable_reference<Tag>(make_not_null(&box)));
  }
};
}  // namespace Parallel::Actions
