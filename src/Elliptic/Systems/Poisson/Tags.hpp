// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DataBox tags for the Poisson system

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
/// \endcond

/*!
 * \ingroup EllipticSystemsGroup
 * \brief Items related to solving a Poisson equation \f$-\Delta u(x)=f(x)\f$.
 */
namespace Poisson {

/*!
 * \brief The scalar field \f$u(x)\f$ to solve for
 */
struct Field : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Field"; }
};

/*!
 * \brief The auxiliary field \f$\boldsymbol{v}(x)=\nabla u(x)\f$ to formulate
 * the first-order Poisson equation \f$-\nabla \cdot \boldsymbol{v}(x) = f(x)\f$
 */
template <size_t Dim>
struct AuxiliaryField : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
  static std::string name() noexcept { return "AuxiliaryField"; }
};

}  // namespace Poisson
