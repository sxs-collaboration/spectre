// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/*!
 * \ingroup EllipticSystemsGroup
 * \brief Items related to solving the Extended Conformal Thin Sandwich (XCTS)
 * equations.
 */
namespace Xcts {
namespace Tags {

/*!
 * \brief The conformal factor \f$\psi(x)\f$ that rescales the spatial metric
 * \f$\gamma_{ij}=\psi^4\overline{\gamma}_{ij}\f$.
 */
template <typename DataType>
struct ConformalFactor : db::SimpleTag {
  using type = Scalar<DataType>;
};

}  // namespace Tags
}  // namespace Xcts
