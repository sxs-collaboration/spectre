// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"

namespace evolution::dg::subcell::Tags {
/// \brief Tag indicating whether we are retrying a step after a rollback of a
/// failed DG step
///
/// Set to `true` by the DG scheme when the predicted step failed and a rollback
/// is performed. The subcell solver checks the tag, and uses the DG boundary
/// data if a rollback occurred in order to maintain conservation. The subcell
/// solver then sets `DidRollback` to `false`.
struct DidRollback : db::SimpleTag {
  using type = bool;
};
}  // namespace evolution::dg::subcell::Tags
