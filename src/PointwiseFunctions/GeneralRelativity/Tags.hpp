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
template <size_t Dim, typename Frame, typename DataType>
struct SpacetimeMetric : db::SimpleTag {
  using type = tnsr::aa<DataType, Dim, Frame>;
};
template <size_t Dim, typename Frame, typename DataType>
struct InverseSpacetimeMetric : db::SimpleTag {
  using type = tnsr::AA<DataType, Dim, Frame>;
};

template <size_t Dim, typename Frame, typename DataType>
struct SpatialMetric : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
};
/*!
 * \brief Inverse of the spatial metric.
 */
template <size_t Dim, typename Frame, typename DataType>
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
template <size_t Dim, typename Frame, typename DataType>
struct DerivDetSpatialMetric : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};
/*!
 * \brief Spatial derivative of the inverse of the spatial metric.
 */
template <size_t Dim, typename Frame, typename DataType>
struct DerivInverseSpatialMetric : db::SimpleTag {
  using type = tnsr::iJJ<DataType, Dim, Frame>;
};
template <size_t Dim, typename Frame, typename DataType>
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
 * \f$\partial_a \psi_{bc}\f$ assembled from the spatial and temporal
 * derivatives of evolved 3+1 variables.
 */
template <size_t Dim, typename Frame, typename DataType>
struct DerivativesOfSpacetimeMetric : db::SimpleTag {
  using type = tnsr::abb<DataType, Dim, Frame>;
};
template <size_t Dim, typename Frame, typename DataType>
struct SpacetimeChristoffelFirstKind : db::SimpleTag {
  using type = tnsr::abb<DataType, Dim, Frame>;
};
template <size_t Dim, typename Frame, typename DataType>
struct SpacetimeChristoffelSecondKind : db::SimpleTag {
  using type = tnsr::Abb<DataType, Dim, Frame>;
};
template <size_t Dim, typename Frame, typename DataType>
struct SpatialChristoffelFirstKind : db::SimpleTag {
  using type = tnsr::ijj<DataType, Dim, Frame>;
};
template <size_t Dim, typename Frame, typename DataType>
struct SpatialChristoffelSecondKind : db::SimpleTag {
  using type = tnsr::Ijj<DataType, Dim, Frame>;
};
template <size_t Dim, typename Frame, typename DataType>
struct SpacetimeNormalOneForm : db::SimpleTag {
  using type = tnsr::a<DataType, Dim, Frame>;
};
template <size_t Dim, typename Frame, typename DataType>
struct SpacetimeNormalVector : db::SimpleTag {
  using type = tnsr::A<DataType, Dim, Frame>;
};
template <size_t Dim, typename Frame, typename DataType>
struct TraceSpacetimeChristoffelFirstKind : db::SimpleTag {
  using type = tnsr::a<DataType, Dim, Frame>;
};
/*!
 * \brief Trace of the spatial Christoffel symbols of the first kind
 * \f$\Gamma_{i} = \Gamma_{ijk}g^{jk}\f$, where \f$\Gamma_{ijk}\f$ are
 * Christoffel symbols of the first kind and \f$g^{jk}\f$ is the
 * inverse spatial metric.
 */
template <size_t Dim, typename Frame, typename DataType>
struct TraceSpatialChristoffelFirstKind : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};
template <size_t Dim, typename Frame, typename DataType>
struct TraceSpatialChristoffelSecondKind : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
};
/// Contraction of the first two indices of the spatial Christoffel symbols:
/// \f$\Gamma^i_{ij}\f$. Useful for covariant divergences.
template <size_t Dim, typename Frame, typename DataType>
struct SpatialChristoffelSecondKindContracted : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

template <size_t Dim, typename Frame, typename DataType>
struct ExtrinsicCurvature : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
};
template <typename DataType>
struct TraceExtrinsicCurvature : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief Computes the spatial Ricci tensor from the spatial
 * Christoffel symbol of the second kind and its derivative.
 */
template <size_t Dim, typename Frame, typename DataType>
struct SpatialRicci : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
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
template <size_t Dim, typename Frame, typename DataType>
struct MomentumDensity : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
};

/// The ADM Hamiltonian constraint \f$R + K^2 - K_{ij} K^{ij} - 16 \pi \rho\f$
/// (see e.g. Eq. (2.132) in \cite BaumgarteShapiro).
///
/// \warning Some authors include a factor of \f$1/2\f$ in the Hamiltonian
/// constraint, as does SpEC.
template <typename DataType>
struct HamiltonianConstraint : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The ADM momentum constraint
/// \f$\gamma^{jk} (\nabla_j K_{ki} - \nabla_i K_{jk}) - 8 \pi S_i\f$, where
/// \f$\nabla\f$ denotes the covariant derivative associated with the spatial
/// metric \f$\gamma_{ij}\f$ (see e.g. Eq. (2.133) in \cite BaumgarteShapiro).
template <size_t Dim, typename Frame, typename DataType>
struct MomentumConstraint : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

/*!
 * \brief Computes the electric part of the Weyl tensor in vacuum
 * as: \f$ E_{ij} = R_{ij} + KK_{ij} - K^m_{i}K_{mj}\f$ where \f$R_{ij}\f$ is
 * the spatial Ricci tensor, \f$K_{ij}\f$ is the extrinsic curvature, and
 * \f$K\f$ is the trace of \f$K_{ij}\f$.
 */
template <size_t Dim, typename Frame, typename DataType>
struct WeylElectric : db::SimpleTag {
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
}  // namespace Tags

/// GR Tags commonly needed for the evolution of hydro systems
template <size_t Dim, typename DataType>
using tags_for_hydro = tmpl::list<
    gr::Tags::Lapse<DataType>, gr::Tags::Shift<Dim, Frame::Inertial, DataType>,
    gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>,
    gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataType>,
    gr::Tags::SqrtDetSpatialMetric<DataType>,
    ::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<Dim>,
                  Frame::Inertial>,
    ::Tags::deriv<gr::Tags::Shift<Dim, Frame::Inertial, DataType>,
                  tmpl::size_t<Dim>, Frame::Inertial>,
    ::Tags::deriv<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>,
                  tmpl::size_t<Dim>, Frame::Inertial>,
    gr::Tags::ExtrinsicCurvature<Dim, Frame::Inertial, DataType>>;

/// The tags for the variables returned by GR analytic solutions.
template <size_t Dim, typename DataType>
using analytic_solution_tags = tmpl::list<
    gr::Tags::Lapse<DataType>, ::Tags::dt<gr::Tags::Lapse<DataType>>,
    ::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<Dim>,
                  Frame::Inertial>,
    gr::Tags::Shift<Dim, Frame::Inertial, DataType>,
    ::Tags::dt<gr::Tags::Shift<Dim, Frame::Inertial, DataType>>,
    ::Tags::deriv<gr::Tags::Shift<Dim, Frame::Inertial, DataType>,
                  tmpl::size_t<Dim>, Frame::Inertial>,
    gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>,
    ::Tags::dt<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>>,
    ::Tags::deriv<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>,
                  tmpl::size_t<Dim>, Frame::Inertial>,
    gr::Tags::SqrtDetSpatialMetric<DataType>,
    gr::Tags::ExtrinsicCurvature<Dim, Frame::Inertial, DataType>,
    gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataType>>;
}  // namespace gr
