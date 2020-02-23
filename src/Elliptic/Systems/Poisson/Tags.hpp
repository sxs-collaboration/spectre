// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DataBox tags for the Poisson system

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
/// \endcond

/*!
 * \ingroup EllipticSystemsGroup
 * \brief Items related to solving a Poisson equation \f$-\Delta u(x)=f(x)\f$.
 */
namespace Poisson {
namespace Tags {

/*!
 * \brief The scalar field \f$u(x)\f$ to solve for
 */
struct Field : db::SimpleTag {
  using type = Scalar<DataVector>;
};

}  // namespace Tags
}  // namespace Poisson
