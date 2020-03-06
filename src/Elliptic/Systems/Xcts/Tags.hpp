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

/*!
 * \brief The gradient of the conformal factor \f$\psi(x)\f$
 *
 * \details This quantity can be used as an auxiliary variable in a first-order
 * formulation of the XCTS equations.
 */
template <size_t Dim, typename Frame, typename DataType>
struct ConformalFactorGradient : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
};

}  // namespace Tags
}  // namespace Xcts
