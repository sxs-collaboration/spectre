// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"

namespace Tags {
/// \ingroup TimeGroup
/// \brief Tag for reporting whether the `ErrorControl` step chooser is in
/// use.
struct IsUsingTimeSteppingErrorControl : db::SimpleTag {
  using type = bool;
};
}  // namespace Tags
