// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"

namespace gr {
namespace Tags {
template <typename DataType, size_t Dim, typename Frame>
struct SpacetimeMetric : db::SimpleTag {
  using type = tnsr::aa<DataType, Dim, Frame>;
};
template <typename DataType, size_t Dim, typename Frame>
struct InverseSpacetimeMetric : db::SimpleTag {
  using type = tnsr::AA<DataType, Dim, Frame>;
};

template <typename DataType, size_t Dim, typename Frame>
struct SpatialMetric : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
};
/*!
 * \brief Inverse of the spatial metric.
 */
template <typename DataType, size_t Dim, typename Frame>
struct InverseSpatialMetric : db::SimpleTag {
  using type = tnsr::II<DataType, Dim, Frame>;
};
/*!
 * \brief Determinant of the spatial metric.
 */
template <typename DataType>
struct DetSpatialMetric : db::SimpleTag {
  using type = Scalar<DataType>;
};
template <typename DataType>
struct SqrtDetSpatialMetric : db::SimpleTag {
  using type = Scalar<DataType>;
};
/*!
 * \brief Derivative of the determinant of the spatial metric.
 */
template <typename DataType, size_t Dim, typename Frame>
struct DerivDetSpatialMetric : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};
/*!
 * \brief Spatial derivative of the inverse of the spatial metric.
 */
template <typename DataType, size_t Dim, typename Frame>
struct DerivInverseSpatialMetric : db::SimpleTag {
  using type = tnsr::iJJ<DataType, Dim, Frame>;
};
template <typename DataType, size_t Dim, typename Frame>
struct Shift : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
};
template <typename DataType>
struct Lapse : db::SimpleTag {
  using type = Scalar<DataType>;
};
/*!
 * \brief Spacetime derivatives of the spacetime metric
 *
 * \details Spacetime derivatives of the spacetime metric
 * \f$\partial_a g_{bc}\f$ assembled from the spatial and temporal
 * derivatives of evolved 3+1 variables.
 */
template <typename DataType, size_t Dim, typename Frame>
struct DerivativesOfSpacetimeMetric : db::SimpleTag {
  using type = tnsr::abb<DataType, Dim, Frame>;
};
template <typename DataType, size_t Dim, typename Frame>
struct SpacetimeChristoffelFirstKind : db::SimpleTag {
  using type = tnsr::abb<DataType, Dim, Frame>;
};
template <typename DataType, size_t Dim, typename Frame>
struct SpacetimeChristoffelSecondKind : db::SimpleTag {
  using type = tnsr::Abb<DataType, Dim, Frame>;
};
template <typename DataType, size_t Dim, typename Frame>
struct SpatialChristoffelFirstKind : db::SimpleTag {
  using type = tnsr::ijj<DataType, Dim, Frame>;
};
template <typename DataType, size_t Dim, typename Frame>
struct SpatialChristoffelSecondKind : db::SimpleTag {
  using type = tnsr::Ijj<DataType, Dim, Frame>;
};
template <typename DataType, size_t Dim, typename Frame>
struct SpacetimeNormalOneForm : db::SimpleTag {
  using type = tnsr::a<DataType, Dim, Frame>;
};
template <typename DataType, size_t Dim, typename Frame>
struct SpacetimeNormalVector : db::SimpleTag {
  using type = tnsr::A<DataType, Dim, Frame>;
};
template <typename DataType, size_t Dim, typename Frame>
struct TraceSpacetimeChristoffelFirstKind : db::SimpleTag {
  using type = tnsr::a<DataType, Dim, Frame>;
};
/*!
 * \brief Trace of the spacetime Christoffel symbols of the second kind
 * \f$\Gamma^{i} = \Gamma^i_{jk}g^{jk}\f$, where \f$\Gamma^i_{jk}\f$ are
 * Christoffel symbols of the second kind and \f$g^{jk}\f$ is the
 * inverse spacetime metric.
 */
template <typename DataType, size_t Dim, typename Frame>
struct TraceSpacetimeChristoffelSecondKind : db::SimpleTag {
  using type = tnsr::A<DataType, Dim, Frame>;
};
/*!
 * \brief Trace of the spatial Christoffel symbols of the first kind
 * \f$\Gamma_{i} = \Gamma_{ijk}\gamma^{jk}\f$, where \f$\Gamma_{ijk}\f$ are
 * Christoffel symbols of the first kind and \f$\gamma^{jk}\f$ is the
 * inverse spatial metric.
 */
template <typename DataType, size_t Dim, typename Frame>
struct TraceSpatialChristoffelFirstKind : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};
template <typename DataType, size_t Dim, typename Frame>
struct TraceSpatialChristoffelSecondKind : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
};
/// Contraction of the first two indices of the spatial Christoffel symbols:
/// \f$\Gamma^i_{ij}\f$. Useful for covariant divergences.
template <typename DataType, size_t Dim, typename Frame>
struct SpatialChristoffelSecondKindContracted : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

template <typename DataType, size_t Dim, typename Frame>
struct ExtrinsicCurvature : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
};
template <typename DataType>
struct TraceExtrinsicCurvature : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief Holds a quantity that's similar to the shift, but isn't the shift.
 *
 * \details This holds
 *
 * \f{equation}{
 * \beta^i \frac{\partial x^\hat{i}}{\partial x^i} =
 * \hat{beta}^\hat{i} + \frac{\partial x^\hat{i}}{\partial t}
 * \f}
 *
 * where hatted quantities are in the distorted frame and non-hatted quantities
 * are in the grid frame.
 */
template <typename DataType, size_t Dim, typename Frame>
struct ShiftyQuantity : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
};

/*!
 * \brief Computes the spatial Ricci tensor from the spatial
 * Christoffel symbol of the second kind and its derivative.
 */
template <typename DataType, size_t Dim, typename Frame>
struct SpatialRicci : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
};

/*!
 * \brief Simple tag for the spatial Ricci scalar
 */
template <typename DataType>
struct SpatialRicciScalar : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief Computes the real part of \f$\Psi_4\f$
 */
template <typename DataType>
struct Psi4Real : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief The energy density \f$E=n_a n_b T^{ab}\f$, where \f$n_a\f$ denotes the
 * normal to the spatial hypersurface
 */
template <typename DataType>
struct EnergyDensity : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief The trace of the spatial stress-energy tensor
 * \f$S=\gamma^{ij}\gamma_{ia}\gamma_{jb}T^{ab}\f$
 */
template <typename DataType>
struct StressTrace : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief The spatial momentum density \f$S^i=-\gamma^{ij}n^aT_{aj}\f$, where
 * \f$n_a\f$ denotes the normal to the spatial hypersurface
 */
template <typename DataType, size_t Dim, typename Frame>
struct MomentumDensity : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
};

/// The ADM Hamiltonian constraint
/// \f$\frac{1}{2} \left(R + K^2 - K_{ij} K^{ij}\right) - 8 \pi \rho\f$
/// (see e.g. Eq. (2.132) in \cite BaumgarteShapiro).
///
/// \note We include a factor of \f$1/2\f$ in the Hamiltonian constraint for
/// consistency with SpEC, and so the matter terms in the Hamiltonian and
/// momentum constraints are both scaled by $8\pi$.
template <typename DataType>
struct HamiltonianConstraint : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The ADM momentum constraint
/// \f$\nabla_j (K^{ij} - \gamma^{ij} K) - 8 \pi S^i\f$, where
/// \f$\nabla\f$ denotes the covariant derivative associated with the spatial
/// metric \f$\gamma_{ij}\f$ (see e.g. Eq. (2.133) in \cite BaumgarteShapiro).
template <typename DataType, size_t Dim, typename Frame>
struct MomentumConstraint : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
};

/*!
 * \brief Computes the electric part of the Weyl tensor in vacuum
 * as: \f$ E_{ij} = R_{ij} + KK_{ij} - K^m_{i}K_{mj}\f$ where \f$R_{ij}\f$ is
 * the spatial Ricci tensor, \f$K_{ij}\f$ is the extrinsic curvature, and
 * \f$K\f$ is the trace of \f$K_{ij}\f$.
 */
template <typename DataType, size_t Dim, typename Frame>
struct WeylElectric : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
};

/*!
 * \brief The magnetic part of the Weyl tensor in vacuum \f$B_{ij}\f$.
 */
template <typename DataType, size_t Dim, typename Frame>
struct WeylMagnetic : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
};

/*!
 * \brief Computes the scalar \f$E_{ij} E^{ij}\f$ from the electric part of the
 * Weyl tensor \f$E_{ij}\f$ and the inverse spatial metric \f$\gamma^{ij}\f$,
 * i.e. \f$E_{ij} E^{ij} = \gamma^{ik}\gamma^{jl}E_{ij}E_{kl}\f$.
 */
template <typename DataType>
struct WeylElectricScalar : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief The square \f$B_{ij} B^{ij}\f$ of the magnetic part of the Weyl tensor
 * \f$B_{ij}\f$.
 */
template <typename DataType>
struct WeylMagneticScalar : db::SimpleTag {
  using type = Scalar<DataType>;
};
}  // namespace Tags

/// GR Tags commonly needed for the evolution of hydro systems
template <size_t Dim, typename DataType>
using tags_for_hydro =
    tmpl::list<gr::Tags::Lapse<DataType>, gr::Tags::Shift<DataType, Dim>,
               gr::Tags::SpatialMetric<DataType, Dim>,
               gr::Tags::InverseSpatialMetric<DataType, Dim>,
               gr::Tags::SqrtDetSpatialMetric<DataType>,
               ::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<Dim>,
                             Frame::Inertial>,
               ::Tags::deriv<gr::Tags::Shift<DataType, Dim>, tmpl::size_t<Dim>,
                             Frame::Inertial>,
               ::Tags::deriv<gr::Tags::SpatialMetric<DataType, Dim>,
                             tmpl::size_t<Dim>, Frame::Inertial>,
               gr::Tags::ExtrinsicCurvature<DataType, Dim>>;

/// The tags for the variables returned by GR analytic solutions.
template <size_t Dim, typename DataType>
using analytic_solution_tags =
    tmpl::list<gr::Tags::Lapse<DataType>, ::Tags::dt<gr::Tags::Lapse<DataType>>,
               ::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<Dim>,
                             Frame::Inertial>,
               gr::Tags::Shift<DataType, Dim>,
               ::Tags::dt<gr::Tags::Shift<DataType, Dim>>,
               ::Tags::deriv<gr::Tags::Shift<DataType, Dim>, tmpl::size_t<Dim>,
                             Frame::Inertial>,
               gr::Tags::SpatialMetric<DataType, Dim>,
               ::Tags::dt<gr::Tags::SpatialMetric<DataType, Dim>>,
               ::Tags::deriv<gr::Tags::SpatialMetric<DataType, Dim>,
                             tmpl::size_t<Dim>, Frame::Inertial>,
               gr::Tags::SqrtDetSpatialMetric<DataType>,
               gr::Tags::ExtrinsicCurvature<DataType, Dim>,
               gr::Tags::InverseSpatialMetric<DataType, Dim>>;
}  // namespace gr
