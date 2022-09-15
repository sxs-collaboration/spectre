// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
/// \endcond

namespace evolution::dg::subcell::Tags {
/// Stores the status of the troubled cell indicator in the element as a
/// `Scalar<DataVector>` so it can be observed.
struct TciStatus : db::SimpleTag {
  using type = Scalar<DataVector>;
};
}  // namespace evolution::dg::subcell::Tags
