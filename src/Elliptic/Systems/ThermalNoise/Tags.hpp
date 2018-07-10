// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DataBox tags for the ThermalNoise system

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
/// \endcond

/*!
 * \ingroup EllipticSystemsGroup
 * \brief Items related to solving for the thermal noise in crystalline thin
 * coatings.
 *
 * \details An introduction to this problem is provided in e.g.
 * https://doi.org/10.1088/1361-6382/aa9ccc.
 */
namespace ThermalNoise {

/*!
 * \brief The amount of material deformation \f$\boldsymbol{u}(x)\f$ when a
 * static force is applied.
 */
template <size_t Dim>
struct Displacement : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim>;
  static std::string name() noexcept { return "Displacement"; }
};

/*!
 * \brief The strain \f$S_{ij}=\nabla_{(i}u_{j)}\f$.
 */
template <size_t Dim>
struct Strain : db::SimpleTag {
  using type = tnsr::ii<DataVector, Dim>;
  static std::string name() noexcept { return "Strain"; }
};

}  // namespace ThermalNoise
