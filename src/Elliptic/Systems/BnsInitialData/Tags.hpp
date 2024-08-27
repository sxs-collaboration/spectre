// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
/// \endcond

/*!
 * \ingroup EllipticSystemsGroup
 * \brief Items related to solving for irrotational bns initial data
 */
namespace BnsInitialData::Tags {

/*!
 * \brief The shift plus a spatial vector \f$ k^i\f$
 * \f$B^i = \beta^i + k^i\f$
 */
template <typename DataType>
struct RotationalShift : db::SimpleTag {
  using type = tnsr::I<DataType, 3>;
};
/*!
 * \brief The stress-energy corresponding to the rotation shift
 *
 *
 * \f[\Sigma^i_j = \frac{1}{2}\frac{B^iB_j}{\alpha^2}\f]
 */
template <typename DataType>
struct RotationalShiftStress : db::SimpleTag {
  using type = tnsr::II<DataType, 3>;
};
/*!
 * \brief  The derivative  \f$D_i \ln (\alpha \rho/h)\f$
 */
template <typename DataType>
struct DerivLogLapseTimesDensityOverSpecificEnthalpy : db::SimpleTag {
  using type = tnsr::i<DataType, 3>;
};

/*!
 * \brief The velocity potential for the fluid flow \f$\Phi\f$, i.e. the
 * curl-free part of the fluid is given by \f$\nabla_a \Phi = h u_a\f$
 */
template <typename DataType>
struct VelocityPotential : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType>
struct SpatialRotationalKillingVector : db::SimpleTag {
  using type = tnsr::I<DataType, 3>;
};

template <typename DataType>
struct DerivSpatialRotationalKillingVector : db::SimpleTag {
  using type = tnsr::iJ<DataType, 3>;
};

struct EulerEnthalpyConstant : db::SimpleTag {
  using type = double;
};

}  // namespace BnsInitialData::Tags
