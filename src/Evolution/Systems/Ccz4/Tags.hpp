// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Ccz4/TagsDeclarations.hpp"
#include "Evolution/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"

namespace Ccz4 {
namespace Tags {
/*!
 * \brief The conformal factor that rescales the spatial metric
 *
 * \details If \f$\gamma_{ij}\f$ is the spatial metric, then we define
 * \f$\phi = (det(\gamma_{ij}))^{-1/6}\f$.
 */
template <typename DataType>
struct ConformalFactor : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief The square of the conformal factor that rescales the spatial metric
 *
 * \details If \f$\gamma_{ij}\f$ is the spatial metric, then we define
 * \f$\phi^2 = (det(\gamma_{ij}))^{-1/3}\f$.
 */
template <typename DataType>
struct ConformalFactorSquared : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief The conformally scaled spatial metric
 *
 * \details If \f$\phi\f$ is the conformal factor and \f$\gamma_{ij}\f$ is the
 * spatial metric, then we define
 * \f$\bar{\gamma}_{ij} = \phi^2 \gamma_{ij}\f$.
 */
template <size_t Dim, typename Frame, typename DataType>
using ConformalMetric =
    gr::Tags::Conformal<gr::Tags::SpatialMetric<Dim, Frame, DataType>, 2>;

/*!
 * \brief The conformally scaled inverse spatial metric
 *
 * \details If \f$\phi\f$ is the conformal factor and \f$\gamma^{ij}\f$ is the
 * inverse spatial metric, then we define
 * \f$\bar{\gamma}^{ij} = \phi^{-2} \gamma^{ij}\f$.
 */
template <size_t Dim, typename Frame, typename DataType>
using InverseConformalMetric =
    gr::Tags::Conformal<gr::Tags::InverseSpatialMetric<Dim, Frame, DataType>,
                        -2>;

/*!
 * \brief The trace-free part of the extrinsic curvature
 *
 * \details See `Ccz4::a_tilde()` for details.
 */
template <size_t Dim, typename Frame, typename DataType>
struct ATilde : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
};

/*!
 * \brief The trace of the trace-free part of the extrinsic curvature
 *
 * \details We define:
 *
 * \f{align}
 *     tr\tilde{A} &= \tilde{\gamma}^{ij} \tilde{A}_{ij}
 * \f}
 *
 * where \f$\tilde{\gamma}^{ij}\f$ is the inverse conformal spatial metric
 * defined by `Ccz4::Tags::InverseConformalMetric` and \f$\tilde{A}_{ij}\f$ is
 * the trace-free part of the extrinsic curvature defined by
 * `Ccz4::Tags::ATilde`.
 */
template <typename DataType>
struct TraceATilde : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief The natural log of the lapse
 */
template <typename DataType>
struct LogLapse : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief Auxiliary variable which is analytically the spatial derivative of the
 * natural log of the lapse
 *
 * \details If \f$ \alpha \f$ is the lapse, then we define
 * \f$A_i = \partial_i ln(\alpha) = \frac{\partial_i \alpha}{\alpha}\f$.
 */
template <size_t Dim, typename Frame, typename DataType>
struct FieldA : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

/*!
 * \brief Auxiliary variable which is analytically the spatial derivative of the
 * shift
 */
template <size_t Dim, typename Frame, typename DataType>
struct FieldB : db::SimpleTag {
  using type = tnsr::iJ<DataType, Dim, Frame>;
};

/*!
 * \brief Auxiliary variable which is analytically half the spatial derivative
 * of the conformal spatial metric
 *
 * \details If \f$\bar{\gamma}_{ij}\f$ is the conformal spatial metric, then we
 * define
 * \f$D_{kij} = \frac{1}{2} \partial_k \bar{\gamma}_{ij}\f$.
 */
template <size_t Dim, typename Frame, typename DataType>
struct FieldD : db::SimpleTag {
  using type = tnsr::ijj<DataType, Dim, Frame>;
};

/*!
 * \brief The natural log of the conformal factor
 */
template <typename DataType>
struct LogConformalFactor : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief Auxiliary variable which is analytically the spatial derivative of the
 * natural log of the conformal factor
 *
 * \details If \f$\phi\f$ is the conformal factor, then we define
 * \f$P_i = \partial_i ln(\phi) = \frac{\partial_i \phi}{\phi}\f$.
 */
template <size_t Dim, typename Frame, typename DataType>
struct FieldP : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

/*!
 * \brief Identity which is analytically negative one half the spatial
 * derivative of the inverse conformal spatial metric
 *
 * \details We define:
 * \f{align}
 *     D_k{}^{ij} &=
 *         \tilde{\gamma}^{in} \tilde{\gamma}^{mj} D_{knm} =
 *         -\frac{1}{2} \partial_k \tilde{\gamma}^{ij}
 * \f}
 * where \f$\tilde{\gamma}^{ij}\f$ and \f$D_{ijk}\f$ are the inverse conformal
 * spatial metric and the CCZ4 auxiliary variable defined by
 * `Ccz4::Tags::FieldD`, respectively.
 */
template <size_t Dim, typename Frame, typename DataType>
struct FieldDUp : db::SimpleTag {
  using type = tnsr::iJJ<DataType, Dim, Frame>;
};

/*!
 * \brief The conformal spatial christoffel symbols of the second kind
 *
 * \details We define:
 * \f{align}
 *     \tilde{\Gamma}^k_{ij} &=
 *         \tilde{\gamma}^{kl} (D_{ijl} + D_{jil} - D_{lij})
 * \f}
 * where \f$\tilde{\gamma}^{ij}\f$ and \f$D_{ijk}\f$ are the inverse conformal
 * spatial metric and the CCZ4 auxiliary variable defined by
 * `Ccz4::Tags::InverseConformalMetric` and `Ccz4::Tags::FieldD`, respectively.
 */
template <size_t Dim, typename Frame, typename DataType>
struct ConformalChristoffelSecondKind : db::SimpleTag {
  using type = tnsr::Ijj<DataType, Dim, Frame>;
};

/*!
 * \brief The spatial derivative of the conformal spatial christoffel symbols
 * of the second kind
 *
 * \details We define:
 * \f{align}
 *     \partial_k \tilde{\Gamma}^m{}_{ij} &=
 *       -2 D_k{}^{ml} (D_{ijl} + D_{jil} - D_{lij}) +
 *       \tilde{\gamma}^{ml}(\partial_{(k} D_{i)jl} + \partial_{(k} D_{j)il} -
 *       \partial_{(k} D_{l)ij})
 * \f}
 * where \f$\tilde{\gamma}^{ij}\f$, \f$D_{ijk}\f$, \f$\partial_l D_{ijk}\f$, and
 * \f$D_k{}^{ij}\f$ are the inverse conformal spatial metric defined by
 * `Ccz4::Tags::InverseConformalMetric`, the CCZ4 auxiliary variable defined by
 * `Ccz4::Tags::FieldD`, its spatial derivative, and the CCZ4 identity defined
 * by `Ccz4::Tags::FieldDUp`.
 */
template <size_t Dim, typename Frame, typename DataType>
struct DerivConformalChristoffelSecondKind : db::SimpleTag {
  using type = tnsr::iJkk<DataType, Dim, Frame>;
};

/*!
 * \brief The spatial christoffel symbols of the second kind
 *
 * \details We define:
 * \f{align}
 *     \Gamma^k_{ij} &= \tilde{\Gamma}^k_{ij} -
 *         \tilde{\gamma}^{kl} (\tilde{\gamma}_{jl} P_i +
 *                              \tilde{\gamma}_{il} P_j -
 *                              \tilde{\gamma}_{ij} P_l)
 * \f}
 * where \f$\tilde{\gamma}^{ij}\f$, \f$\tilde{\gamma}_{ij}\f$,
 * \f$\tilde{\Gamma}^k_{ij}\f$, and \f$P_i\f$ are the conformal spatial metric,
 * the inverse conformal spatial metric, the conformal spatial christoffel
 * symbols of the second kind, and the CCZ4 auxiliary variable defined by
 * `Ccz4::Tags::ConformalMetric`, `Ccz4::Tags::InverseConformalMetric`,
 * `Ccz4::Tags::ConformalChristoffelSecondKind`, and `Ccz4::Tags::FieldP`,
 * respectively.
 */
template <size_t Dim, typename Frame, typename DataType>
struct ChristoffelSecondKind : db::SimpleTag {
  using type = tnsr::Ijj<DataType, Dim, Frame>;
};

/*!
 * \brief The spatial Ricci tensor
 *
 * \details See `Ccz4::spatial_ricci_tensor()` for details.
 */
template <size_t Dim, typename Frame, typename DataType>
struct Ricci : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
};

/*!
 * \brief The gradient of the gradient of the lapse
 *
 * \details We define:
 * \f{align}
 *     \nabla_i \nabla_j \alpha &= \alpha A_i A_j -
 *                 \alpha \Gamma^k{}_{ij} A_k + \alpha \partial_{(i} A_{j)}
 * \f}
 * where \f$\alpha\f$, \f$\Gamma^k{}_{ij}\f$, \f$A_i\f$, and
 * \f$\partial_j A_i\f$ are the lapse, spatial christoffel symbols of the second
 * kind, the CCZ4 auxiliary variable defined by `Ccz4::Tags::FieldA`, and its
 * spatial derivative, respectively.
 */
template <size_t Dim, typename Frame, typename DataType>
struct GradGradLapse : db::SimpleTag {
  using type = tnsr::ij<DataType, Dim, Frame>;
};

/*!
 * \brief The divergence of the lapse
 *
 * \details We define:
 * \f{align}
 *     \nabla^i \nabla_i \alpha &= \phi^2 \tilde{\gamma}^{ij}
 *         (\nabla_i \nabla_j \alpha)
 * \f}
 * where \f$\phi\f$, \f$\tilde{\gamma}^{ij}\f$, and
 * \f$\nabla_i \nabla_j \alpha\f$ are the conformal factor, inverse conformal
 * spatial metric, and the gradient of the gradient of the lapse defined by
 * `Ccz4::Tags::ConformalFactor`, `Ccz4::Tags::InverseConformalMetric`, and
 * `Ccz4::Tags::GradGradLapse`, respectively.
 */
template <typename DataType>
struct DivergenceLapse : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief The contraction of the conformal spatial Christoffel symbols of the
 * second kind
 *
 * \details See `Ccz4::contracted_conformal_christoffel_second_kind()` for
 * details.
 */
template <size_t Dim, typename Frame, typename DataType>
struct ContractedConformalChristoffelSecondKind : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
};

/*!
 * \brief The spatial derivative of the contraction of the conformal spatial
 * Christoffel symbols of the second kind
 *
 * \details See `Ccz4::deriv_contracted_conformal_christoffel_second_kind()` for
 * details.
 */
template <size_t Dim, typename Frame, typename DataType>
struct DerivContractedConformalChristoffelSecondKind : db::SimpleTag {
  using type = tnsr::iJ<DataType, Dim, Frame>;
};

/*!
 * \brief The CCZ4 evolved variable \f$\hat{\Gamma}^i\f$
 *
 * \details This must satisfy the identity:
 *
 * \f{align}
 *     \hat{\Gamma}^i &= \tilde{\Gamma}^i + 2 \tilde{\gamma}^{ij} Z_j
 * \f}
 *
 * where \f$\tilde{\gamma}^{ij}\f$ is the inverse conformal spatial metric
 * defined by `Ccz4::Tags::InverseConformalMetric`, \f$Z_i\f$ is the spatial
 * part of the Z4 constraint defined by `Ccz4::Tags::SpatialZ4Constraint`, and
 * \f$\tilde{\Gamma}^i\f$ is the contraction of the conformal spatial
 * christoffel symbols of the second kind defined by
 * `Ccz4::Tags::ContractedConformalChristoffelSecondKind`.
 */
template <size_t Dim, typename Frame, typename DataType>
struct GammaHat : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
};

/*!
 * \brief The spatial part of the Z4 constraint
 *
 * \details See `Ccz4::spatial_z4_constraint` for details.
 */
template <size_t Dim, typename Frame, typename DataType>
struct SpatialZ4Constraint : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

/*!
 * \brief The spatial part of the upper Z4 constraint
 *
 * \details See `Ccz4::upper_spatial_z4_constraint` for details.
 */
template <size_t Dim, typename Frame, typename DataType>
struct SpatialZ4ConstraintUp : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
};

/*!
 * \brief The gradient of the spatial part of the Z4 constraint
 *
 * \details See `Ccz4::grad_spatial_z4_constraint` for details.
 */
template <size_t Dim, typename Frame, typename DataType>
struct GradSpatialZ4Constraint : db::SimpleTag {
  using type = tnsr::ij<DataType, Dim, Frame>;
};
}  // namespace Tags

namespace OptionTags {
/*!
 * \ingroup OptionGroupsGroup
 * Groups option tags related to the CCZ4 evolution system.
 */
struct Group {
  static std::string name() { return "Ccz4"; }
  static constexpr Options::String help{
      "Options for the CCZ4 evolution system"};
  using group = evolution::OptionTags::SystemGroup;
};
}  // namespace OptionTags
}  // namespace Ccz4
