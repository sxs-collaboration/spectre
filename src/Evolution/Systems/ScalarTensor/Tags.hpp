// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarTensor::Tags {

/*!
 * \brief Represents the trace-reversed stress-energy tensor of the scalar
 * field.
 */
struct TraceReversedStressEnergy : db::SimpleTag {
  using type = tnsr::aa<DataVector, 3>;
};

}  // namespace ScalarTensor::Tags
