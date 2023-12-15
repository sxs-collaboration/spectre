// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/PrettyType.hpp"

namespace imex::Tags {
/*!
 * Tag for a count of the pointwise implicit solve failures during the
 * most recent solve.  A value of 0 means the solve succeeded, a value
 * of 1 means the first fallback succeeded, and so on.
 */
template <typename Sector>
struct SolveFailures : db::SimpleTag {
  static std::string name() {
    return "SolveFailures(" + pretty_type::name<Sector>() + ")";
  }
  using type = Scalar<DataVector>;
};
}  // namespace imex::Tags
