// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Ccz4/TagsDeclarations.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

namespace Ccz4 {
namespace Tags {
/*!
 * \brief The CCZ4 temporary expression
 * \f$\hat{\Gamma}^i - \tilde{\Gamma}^i\f$
 *
 * \details We define:
 *
 * \f{align}
 *     \hat{\Gamma}^i - \tilde{\Gamma}^i &= 2 \tilde{\gamma}^{ij} Z_j
 * \f}
 *
 * where \f$\hat{\Gamma}^{i}\f$ is the CCZ4 evolved variable defined by
 * `Ccz4::Tags::GammaHat`, \f$\tilde{\Gamma}^{i}\f$ is the contraction of the
 * conformal spatial Christoffel symbols of the second kind defined by
 * `Ccz4::Tags::ContractedConformalChristoffelSecondKind`,
 * \f$\tilde{\gamma}^{ij}\f$ is the inverse conformal spatial metric defined by
 * `Ccz4::Tags::InverseConformalMetric`, and \f$Z_i\f$ is the spatial part of
 * the Z4 constraint defined by `Ccz4::Tags::SpatialZ4Constraint`.
 */
template <size_t Dim, typename Frame, typename DataType>
struct GammaHatMinusContractedConformalChristoffel : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
};

/*!
 * \brief The CCZ4 temporary expression \f$K - 2 \Theta c\f$
 *
 * \details Here, \f$K\f$ is the trace of the extrinsic curvature defined by
 * `gr::Tags::TraceExtrinsicCurvature`, \f$\Theta\f$ is the projection of the Z4
 * four-vector along the normal direction, and \f$c\f$ controls whether to
 * include algebraic source terms proportional to \f$\Theta\f$.
 */
template <typename DataType>
struct KMinus2ThetaC : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief The CCZ4 temporary expression \f$K - K_0 - 2 \Theta c\f$
 *
 * \details Here, \f$K\f$ is the trace of the extrinsic curvature defined by
 * `gr::Tags::TraceExtrinsicCurvature`, \f$K_0\f$ is the initial time derivative
 * of the lapse, \f$\Theta\f$ is the projection of the Z4 four-vector along the
 * normal direction, and \f$c\f$ controls whether to include algebraic source
 * terms proportional to \f$\Theta\f$.
 */
template <typename DataType>
struct KMinusK0Minus2ThetaC : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief The CCZ4 temporary expression \f$B_k{}^k\f$
 *
 * \details Here, \f$B_k{}^k\f$ is the contraction of the CCZ4 auxiliary
 * variable defined by `Ccz4::Tags::FieldB`.
 */
template <typename DataType>
struct ContractedFieldB : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief The CCZ4 temporary expression \f$\tilde{\gamma}_{ki} B_j{}^k\f$
 *
 * \details Here, \f$\tilde{\gamma}_{ij}\f$ is the conformal spatial metric
 * defined by `Ccz4::Tags::ConformalMetric` and \f$B_i{}^j\f$ is the CCZ4
 * auxiliary variable defined by `Ccz4::Tags::FieldB`.
 */
template <size_t Dim, typename Frame, typename DataType>
struct ConformalMetricTimesFieldB : db::SimpleTag {
  using type = tnsr::ij<DataType, Dim, Frame>;
};

/*!
 * \brief The CCZ4 temporary expression \f$\alpha (R + 2 \nabla_k Z^k)\f$
 *
 * \details Here, \f$\alpha\f$ is the lapse defined by `gr::Tags::Lapse` and
 * \f$(R + 2 \nabla_k Z^k)\f$ is the Ricci scalar plus twice the divergence of
 * the spatial Z4 constraint defined by
 * `Ccz4::Tags::RicciScalarPlusDivergenceZ4Constraint`.
 */
template <typename DataType>
struct LapseTimesRicciScalarPlus2DivergenceZ4Constraint : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief The CCZ4 temporary expression \f$\tilde{\gamma}_{ij} tr \tilde{A}\f$
 *
 * \details Here, \f$\tilde{\gamma}_{ij}\f$ is the conformal spatial metric
 * defined by `Ccz4::Tags::ConformalMetric` and \f$tr \tilde{A}\f$ is the trace
 * of the trace-free part of the extrinsic curvature defined by
 * `Ccz4::Tags::TraceATilde`.
 */
template <size_t Dim, typename Frame, typename DataType>
struct ConformalMetricTimesTraceATilde : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
};

/*!
 * \brief The CCZ4 temporary expression \f$\alpha \tilde{A}_{ij}\f$
 *
 * \details Here, \f$\alpha\f$ is the lapse defined by `gr::Tags::Lapse` and
 * \f$\tilde{A}_{ij}\f$ is the trace-free part of the extrinsic curvature
 * defined by `Ccz4::Tags::ATilde`.
 */
template <size_t Dim, typename Frame, typename DataType>
struct LapseTimesATilde : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
};

/*!
 * \brief The CCZ4 temporary expression \f$D_k{}^{nm} \tilde{A}_{nm}\f$
 *
 * \details Here, \f$D_k{}^{nm}\f$ is analytically negative one half the spatial
 * derivative of the inverse conformal spatial metric defined by
 * `Ccz4::Tags::FieldDUp` and \f$\tilde{A}_{nm}\f$ is the trace-free part of the
 * extrinsic curvature defined by `Ccz4::Tags::ATilde`.
 */
template <size_t Dim, typename Frame, typename DataType>
struct FieldDUpTimesATilde : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

/*!
 * \brief The CCZ4 temporary expression \f$\alpha \partial_k \tilde{A}_{ij}\f$
 *
 * \details Here, \f$\alpha\f$ is the lapse defined by `gr::Tags::Lapse` and
 * \f$\tilde{A}_{ij}\f$ is the trace-free part of the extrinsic curvature
 * defined by `Ccz4::Tags::ATilde`, and \f$\partial_k \tilde{A}_{ij}\f$ is its
 * spatial derivative.
 */
template <size_t Dim, typename Frame, typename DataType>
struct LapseTimesDerivATilde : db::SimpleTag {
  using type = tnsr::ijj<DataType, Dim, Frame>;
};

/*!
 * \brief The CCZ4 temporary expression
 * \f$\tilde{\gamma}^{nm} \partial_k \tilde{A}_{nm}\f$
 *
 * \details Here, \f$\tilde{\gamma}^{nm}\f$ is the inverse conformal spatial
 * metric defined by `Ccz4::Tags::InverseConformalMetric`, \f$\tilde{A}_{ij}\f$
 * is the trace-free part of the extrinsic curvature defined by
 * `Ccz4::Tags::ATilde`, and \f$\partial_k \tilde{A}_{ij}\f$ is its spatial
 * derivative.
 */
template <size_t Dim, typename Frame, typename DataType>
struct InverseConformalMetricTimesDerivATilde : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

/*!
 * \brief The CCZ4 temporary expression
 * \f$\tilde{A}_{ij} - \frac{1}{3} \tilde{\gamma}_{ij} tr \tilde{A}\f$
 *
 * \details Here, \f$\tilde{A}_{ij}\f$ is the trace-free part of the extrinsic
 * curvature defined by `Ccz4::Tags::ATilde`, \f$tr \tilde{A}\f$ is its trace
 * defined by `Ccz4::Tags::TraceATilde`, and \f$\tilde{\gamma}_{ij}\f$ is the
 * conformal spatial metric defined by `Ccz4::Tags::ConformalMetric`.
 */
template <size_t Dim, typename Frame, typename DataType>
struct ATildeMinusOneThirdConformalMetricTimesTraceATilde : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
};

/*!
 * \brief The CCZ4 temporary expression \f$\alpha A_k\f$
 *
 * \details Here, \f$\alpha\f$ is the lapse defined by `gr::Tags::Lapse` and
 * \f$A_k\f$ is the CCZ4 auxiliary variable defined by `Ccz4::Tags::FieldA`.
 */
template <size_t Dim, typename Frame, typename DataType>
struct LapseTimesFieldA : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

/*!
 * \brief The CCZ4 temporary expression \f$\beta^k \partial_k \hat{\Gamma}^i\f$
 *
 * \details Here, \f$\beta^k\f$ is the shift defined by `gr::Tags::Shift`,
 * \f$\hat{\Gamma}^i\f$ is the CCZ4 evolved variable defined by
 * `Ccz4::Tags::GammaHat`, and \f$\partial_k \hat{\Gamma}^i\f$ is its spatial
 * derivative.
 */
template <size_t Dim, typename Frame, typename DataType>
struct ShiftTimesDerivGammaHat : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
};

/*!
 * \brief The CCZ4 temporary expression \f$\tau^{-1} \tilde{\gamma}_{ij}\f$
 *
 * \details Here, \f$\tilde{\gamma}_{ij}\f$ is the conformal spatial metric
 * defined by `Ccz4::Tags::ConformalMetric` and \f$\tau\f$ is the relaxation
 * time to enforce the algebraic constraints on the determinant of the conformal
 * spatial metric and on the trace of the trace-free part of the extrinsic
 * curvature that is defined by `Ccz4::Tags::ATilde`.
 */
template <size_t Dim, typename Frame, typename DataType>
struct InverseTauTimesConformalMetric : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
};

/*!
 * \brief The CCZ4 temporary expression \f$\alpha g(\alpha)\f$
 *
 * \details Here, \f$\alpha\f$ is the lapse defined by `gr::Tags::Lapse` and
 * \f$g(\alpha)\f$ is a constant that controls the slicing conditions.
 * \f$g(\alpha) = 1\f$ leads to harmonic slicing and
 * \f$g(\alpha) = 2 / \alpha\f$ leads to 1 + log slicing.
 */
template <typename DataType>
struct LapseTimesSlicingCondition : db::SimpleTag {
  using type = Scalar<DataType>;
};
}  // namespace Tags
}  // namespace Ccz4
