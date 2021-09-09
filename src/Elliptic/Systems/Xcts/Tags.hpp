// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"

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
 * \brief The conformally scaled spatial metric
 * \f$\bar{\gamma}_{ij}=\psi^{-4}\gamma_{ij}\f$, where \f$\psi\f$ is the
 * `Xcts::Tags::ConformalFactor` and \f$\gamma_{ij}\f$ is the
 * `gr::Tags::SpatialMetric`
 */
template <typename DataType, size_t Dim, typename Frame>
using ConformalMetric =
    gr::Tags::Conformal<gr::Tags::SpatialMetric<Dim, Frame, DataType>, -4>;

/*!
 * \brief The conformally scaled inverse spatial metric
 * \f$\bar{\gamma}^{ij}=\psi^{4}\gamma^{ij}\f$, where \f$\psi\f$ is the
 * `Xcts::Tags::ConformalFactor` and \f$\gamma^{ij}\f$ is the
 * `gr::Tags::InverseSpatialMetric`
 */
template <typename DataType, size_t Dim, typename Frame>
using InverseConformalMetric =
    gr::Tags::Conformal<gr::Tags::InverseSpatialMetric<Dim, Frame, DataType>,
                        4>;

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
 * \f$B_{ij} = \bar{D}_{(i}\bar{\gamma}_{j)k}\beta^k =
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

/*!
 * \brief The conformal longitudinal operator applied to the shift excess
 * \f$(\bar{L}\beta_\mathrm{excess})^{ij}\f$
 *
 * This quantity can be used as the "flux" for the momentum constraint in
 * formulations of the XCTS equations, because the principal part of the
 * momentum constraint is essentially the divergence of this quantity.
 */
template <typename DataType, size_t Dim, typename Frame>
struct LongitudinalShiftExcess : db::SimpleTag {
  using type = tnsr::II<DataType, Dim, Frame>;
};

/*!
 * \brief The conformal longitudinal operator applied to the background shift
 * vector minus the time derivative of the conformal metric
 * \f$(\bar{L}\beta_\mathrm{background})^{ij} - \bar{u}^{ij}\f$
 *
 * This quantity appears in formulation of the XCTS equations (see `Xcts`) and
 * serves to specify their free data \f$\bar{u}_{ij}\f$. It is combined with the
 * longitudinal background shift because the two quantities are degenerate.
 *
 * \note As usual for conformal quantities, the indices here are raised with the
 * conformal metric: \f$\bar{u}^{ij} = \bar{\gamma}^{ik}\bar{\gamma}^{jl}
 * \partial_t\bar{\gamma}_{kl}\f$.
 *
 * \see `Xcts::Tags::ShiftBackground`
 */
template <typename DataType, size_t Dim, typename Frame>
struct LongitudinalShiftBackgroundMinusDtConformalMetric : db::SimpleTag {
  using type = tnsr::II<DataType, Dim, Frame>;
};

/*!
 * \brief The conformal longitudinal operator applied to the shift vector minus
 * the time derivative of the conformal metric, squared:
 * \f$\left((\bar{L}\beta)^{ij} - \bar{u}^{ij}\right)
 * \left((\bar{L}\beta)_{ij} - \bar{u}_{ij}\right)\f$
 */
template <typename DataType>
struct LongitudinalShiftMinusDtConformalMetricSquare : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief The conformal longitudinal operator applied to the shift vector minus
 * the time derivative of the conformal metric, squared and divided by the
 * square of the lapse:
 * \f$\frac{1}{\alpha^2}\left((\bar{L}\beta)^{ij} - \bar{u}^{ij}\right)
 * \left((\bar{L}\beta)_{ij} - \bar{u}_{ij}\right)\f$
 */
template <typename DataType>
struct LongitudinalShiftMinusDtConformalMetricOverLapseSquare : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief The shift vector contracted with the gradient of the trace of the
 * extrinsic curvature: \f$\beta^i\partial_i K\f$
 *
 * The shift vector in this quantity is the full shift
 * \f$\beta^i=\beta^i_\mathrm{background}+\beta^i_\mathrm{excess}\f$ (see
 * `Xcts::Tags::ShiftExcess` for details on this split).
 */
template <typename DataType>
struct ShiftDotDerivExtrinsicCurvatureTrace : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief The Christoffel symbols of the first kind related to the conformal
 * metric \f$\bar{\gamma}_{ij}\f$
 */
template <typename DataType, size_t Dim, typename Frame>
struct ConformalChristoffelFirstKind : db::SimpleTag {
  using type = tnsr::ijj<DataType, Dim, Frame>;
};

/*!
 * \brief The Christoffel symbols of the second kind related to the conformal
 * metric \f$\bar{\gamma}_{ij}\f$
 */
template <typename DataType, size_t Dim, typename Frame>
struct ConformalChristoffelSecondKind : db::SimpleTag {
  using type = tnsr::Ijj<DataType, Dim, Frame>;
};

/*!
 * \brief The Christoffel symbols of the second kind (related to the conformal
 * metric \f$\bar{\gamma}_{ij}\f$) contracted in their first two indices:
 * \f$\bar{\Gamma}_k = \bar{\Gamma}^{i}_{ik}\f$
 */
template <typename DataType, size_t Dim, typename Frame>
struct ConformalChristoffelContracted : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

/*!
 * \brief The Ricci tensor related to the conformal metric
 * \f$\bar{\gamma}_{ij}\f$
 */
template <typename DataType, size_t Dim, typename Frame>
struct ConformalRicciTensor : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
};

/*!
 * \brief The Ricci scalar related to the conformal metric
 * \f$\bar{\gamma}_{ij}\f$
 */
template <typename DataType>
struct ConformalRicciScalar : db::SimpleTag {
  using type = Scalar<DataType>;
};

}  // namespace Tags
}  // namespace Xcts
