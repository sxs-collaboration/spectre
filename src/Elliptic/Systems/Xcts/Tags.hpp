// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
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
  static std::string name() noexcept { return "ConformalFactor"; }
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
  static std::string name() noexcept { return "ConformalFactorGradient"; }
};

}  // namespace Tags
}  // namespace Xcts
