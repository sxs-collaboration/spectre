// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Punctures {
/// Tags related to the puncture equation
namespace Tags {

/*!
 * \brief The puncture field $u(x)$ to solve for
 *
 * \see Punctures
 */
struct Field : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief The source field $\alpha(x)$
 *
 * \see Punctures
 */
struct Alpha : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief The source field $\beta(x)$
 *
 * \see Punctures
 */
struct Beta : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief The traceless conformal extrinsic curvature $\bar{A}_{ij}$
 *
 * \see Punctures
 */
struct TracelessConformalExtrinsicCurvature : db::SimpleTag {
  using type = tnsr::II<DataVector, 3>;
};

}  // namespace Tags
}  // namespace Punctures
