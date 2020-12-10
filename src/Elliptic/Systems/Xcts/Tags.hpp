// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

/*!
 * \ingroup EllipticSystemsGroup
 * \brief Items related to solving the Extended Conformal Thin Sandwich (XCTS)
 * equations.
 */
namespace Xcts {
/// Tags related to the XCTS equations
namespace Tags {

/*!
 * \brief The conformal factor \f$\psi(x)\f$ that rescales the spatial metric
 * \f$\gamma_{ij}=\psi^4\bar{\gamma}_{ij}\f$.
 */
template <typename DataType>
struct ConformalFactor : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief The quantity `Tag` scaled by the `Xcts::Tags::ConformalFactor` to the
 * given `Power`
 */
template <typename Tag, int Power>
struct Conformal : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
  static constexpr int conformal_factor_power = Power;
};

/*!
 * \brief The conformally scaled spatial metric
 * \f$\bar{\gamma}_{ij}=\psi^{-4}\gamma_{ij}\f$, where \f$\psi\f$ is the
 * `Xcts::Tags::ConformalFactor` and \f$gamma_{ij}\f$ is the
 * `gr::Tags::SpatialMetric`
 */
template <typename DataType, size_t Dim, typename Frame>
using ConformalMetric =
    Conformal<gr::Tags::SpatialMetric<Dim, Frame, DataType>, -4>;

/*!
 * \brief The product of lapse \f$\alpha(x)\f$ and conformal factor
 * \f$\psi(x)\f$
 *
 * This quantity is commonly used in formulations of the XCTS equations.
 */
template <typename DataType>
struct LapseTimesConformalFactor : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief The constant part \f$\beta^i_\mathrm{background}\f$ of the shift
 * \f$\beta^i=\beta^i_\mathrm{background} + \beta^i_\mathrm{excess}\f$
 *
 * \see `Xcts::Tags::ShiftExcess`
 */
template <typename DataType, size_t Dim, typename Frame>
struct ShiftBackground : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
};

/*!
 * \brief The dynamic part \f$\beta^i_\mathrm{excess}\f$ of the shift
 * \f$\beta^i=\beta^i_\mathrm{background} + \beta^i_\mathrm{excess}\f$
 *
 * We commonly split off the part of the shift that diverges at large coordinate
 * distances (the "background" shift \f$\beta^i_\mathrm{background}\f$) and
 * solve only for the remainder (the "excess" shift
 * \f$\beta^i_\mathrm{excess}\f$). For example, the background shift might be a
 * uniform rotation \f$\beta^i_\mathrm{background}=(-\Omega y, \Omega x, 0)\f$
 * with angular velocity \f$\Omega\f$ around the z-axis, given here in Cartesian
 * coordinates.
 *
 * \see `Xcts::Tags::Background`
 */
template <typename DataType, size_t Dim, typename Frame>
struct ShiftExcess : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
};

/*!
 * \brief The symmetric "strain" of the shift vector
 * \f$B_{ij} = \bar{\nabla}_{(i}\bar{\gamma}_{j)k}\beta^k =
 * \left(\partial_{(i}\bar{\gamma}_{j)k} - \bar{\Gamma}_{kij}\right)\beta^k\f$
 *
 * This quantity is used in our formulations of the XCTS equations.
 *
 * Note that the shift is not a conformal quantity, so its index is generally
 * raised and lowered with the spatial metric, not with the conformal metric.
 * However, to compute this "strain" we use the conformal metric as defined
 * above. The conformal longitudinal shift in terms of this quantity is then:
 *
 * \f{equation}
 * (\bar{L}\beta)^{ij} = 2\left(\bar{\gamma}^{ik}\bar{\gamma}^{jl}
 * - \frac{1}{3}\bar{\gamma}^{ij}\bar{\gamma}^{kl}\right) B_{kl}
 * \f}
 *
 * Note that the conformal longitudinal shift is (minus) the "stress" quantity
 * of a linear elasticity system in which the shift takes the role of the
 * displacement vector and the definition of its "strain" remains the same. This
 * auxiliary elasticity system is formulated on an isotropic constitutive
 * relation based on the conformal metric with vanishing bulk modulus \f$K=0\f$
 * (not to be confused with the extrinsic curvature trace \f$K\f$ in this
 * context) and unit shear modulus \f$\mu=1\f$. See the
 * `Elasticity::FirstOrderSystem` and the
 * `Elasticity::ConstitutiveRelations::IsotropicHomogeneous` for details.
 */
template <typename DataType, size_t Dim, typename Frame>
struct ShiftStrain : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
};

}  // namespace Tags
}  // namespace Xcts
