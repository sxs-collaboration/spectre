// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines enum class amr::Flag.

#pragma once

#include <iosfwd>

/// \ingroup ComputationalDomainGroup
/// Items for adaptive mesh refinement
namespace amr {

/// \ingroup ComputationalDomainGroup
/// \brief Flags that represent decisions about mesh refinement
///
/// In order to support anisotropic mesh refinement, a flag is specified for
/// each dimension.
enum class Flag {
  Undefined,          /**< used to initialize flags before a decision is made */
  Join,               /**< join the sibling of an Element */
  DecreaseResolution, /**< decrease number of points in an Element */
  DoNothing,          /**< stay the same */
  IncreaseResolution, /**< increase number of points in an Element */
  Split               /**< split the Element into two smaller elements */
};

/// Output operator for a Flag.
std::ostream& operator<<(std::ostream& os, const Flag& flag);
}  // namespace amr
