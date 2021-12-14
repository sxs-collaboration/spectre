// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

/// \cond
class DataVector;
/// \endcond

namespace domain::Tags {

/*!
 * \brief The determinant of the induced Jacobian on a surface
 *
 * The surface Jacobian determinant on a surface \f$\Sigma\f$ with constant
 * logical coordinate \f$\xi^i\f$ is:
 *
 * \f{equation}
 * J^\Sigma = J \sqrt{\gamma^{jk} (J^{-1})^i_j (J^{-1})^i_k}
 * \f}
 *
 * where \f$J^i_j = \partial x^i / \xi^j\f$ is the volume Jacobian with
 * determinant \f$J\f$ and inverse \f$(J^{-1})^i_j = \partial \xi^i / \partial
 * x^j\f$. Note that the square root in the expression above is the magnitude of
 * the unnormalized face normal, where \f$\gamma^{jk}\f$ is the inverse spatial
 * metric.
 */
template <typename SourceFrame, typename TargetFrame>
struct DetSurfaceJacobian : db::SimpleTag {
  using type = Scalar<DataVector>;
};

}  // namespace domain::Tags
