// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"

namespace ScalarAdvection {
/*!
 * \brief Tags for the ScalarAdvection system
 */
namespace Tags {
/// The scalar field to evolve
struct U : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// The advection velocity field
template <size_t Dim>
struct VelocityField : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim>;
};

/// The largest characteristic speed
struct LargestCharacteristicSpeed : db::SimpleTag {
  using type = double;
};

}  // namespace Tags
}  // namespace ScalarAdvection
