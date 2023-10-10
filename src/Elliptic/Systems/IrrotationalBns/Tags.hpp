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
 * \brief Items related to solving relativisitic hydrostatic equalibrium.
 */
namespace Hydrostatic {
namespace Tags {

/*!
 * \brief The shift plus a spatial vector \f$ k^i\f$
 * \f$B^i = \beta^i + k^i\f$
 */
struct RotationalShift : db::SimpleTag {
  using type = tnsr::I<DataVector, 3>;
};
/*!
 * \brief The stress-energy corresponding to the rotation shift
 *
 *
 * \f[\Sigma^i_j = \frac{1}{2}\frac{B^iB_j}{\alpha^2}\f]
 */
struct RotationalShiftStress : db::SimpleTag {
  using type = tnsr::Ij<DataVector, 3>;
};

/*!
 * \brief The divergence of the stress-energy corresponding
 * to the rotation shift
 *
 * \f[ \D_i \Sigma^i_j = \frac{D_i B^i B_j + B^i D_i B_j}{\alpha^2}\f]
 */
struct DivergenceRotationalShiftStress : db::SimpleTag {
  using type = tnsr::i<DataVector, 3>;
};

/*!
 * \brief  The derivative  \f$D_i \ln (\alpha/h)\f$
 */
struct DerivLogLapseOverSpecificEnthalpy : db::SimpleTag {
  using type = tnsr::i<DataVector, 3>;
};

/*!
 * \brief The velocity potential for the fluid flow \f$\Phi\f$, i.e. the
 * curl-free part of the fluid is given by \f$\nabla_a \Phi = h u_a\f$
 */
struct VelocityPotential : db::SimpleTag {
  using type = Scalar<DataVector>;
};
/*!
 * \brief The auxiliary velocity variable that represents
 * the flux of the velocity potential
 * /f[
 *  U_i \equiv D_i \Phi  - \frac{C + B^jD_j \Phi}{\alpha^2}B_i
 * \f]
 */
struct AuxiliaryVelocity : db::simpleTag {
  using type = tnsr::i<Datavector, 3>;
};

}  // namespace Tags
}  // namespace Hydrostatic
