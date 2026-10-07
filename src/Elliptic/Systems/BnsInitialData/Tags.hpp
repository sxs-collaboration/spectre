// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "Options/String.hpp"

/// \cond
class DataVector;
/// \endcond

/*!
 * \ingroup EllipticSystemsGroup
 * \brief Items related to solving for irrotational bns initial data
 */
namespace BnsInitialData::Tags {
namespace OptionTags {
struct EulerEnthalpyConstant {
  using type = double;
  static constexpr Options::String help =
      "The Euler Enthalpy constant of the star";
};

}  // namespace OptionTags

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
 * \f[\Sigma^{ij} = \frac{B^iB^j}{\alpha^2}\f]
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
  using option_tags = tmpl::list<OptionTags::EulerEnthalpyConstant>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double value) { return value; };
};

}  // namespace BnsInitialData::Tags
