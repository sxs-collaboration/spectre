// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Defines functions to calculate Christoffel symbols

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

/// \cond
namespace gsl {
template <class>
class not_null;
}  // namespace gsl
/// \endcond

namespace gr {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes Christoffel symbol of the first kind from derivative of
 * metric
 *
 * \details Computes Christoffel symbol \f$\Gamma_{abc}\f$ as:
 * \f$ \Gamma_{cab} = \frac{1}{2} ( \partial_a g_{bc} + \partial_b g_{ac}
 *  -  \partial_c g_{ab}) \f$
 * where \f$g_{bc}\f$ is either a spatial or spacetime metric
 */
template <size_t SpatialDim, typename Frame, IndexType Index, typename DataType>
void christoffel_first_kind(
    gsl::not_null<tnsr::abb<DataType, SpatialDim, Frame, Index>*> christoffel,
    const tnsr::abb<DataType, SpatialDim, Frame, Index>& d_metric);

template <size_t SpatialDim, typename Frame, IndexType Index, typename DataType>
tnsr::abb<DataType, SpatialDim, Frame, Index> christoffel_first_kind(
    const tnsr::abb<DataType, SpatialDim, Frame, Index>& d_metric);
/// @}

/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes Christoffel symbol of the second kind from derivative of
 * metric and the inverse metric.
 *
 * \details Computes Christoffel symbol \f$\Gamma^a_{bc}\f$ as:
 * \f$ \Gamma^d_{ab} = \frac{1}{2} g^{cd} (\partial_a g_{bc} + \partial_b g_{ac}
 *  -  \partial_c g_{ab}) \f$
 * where \f$g_{bc}\f$ is either a spatial or spacetime metric.
 *
 * Avoids the extra memory allocation that occurs by computing the
 * Christoffel symbol of the first kind and then raising the index.
 */
template <size_t SpatialDim, typename Frame, IndexType Index, typename DataType>
void christoffel_second_kind(
    const gsl::not_null<tnsr::Abb<DataType, SpatialDim, Frame, Index>*>
        christoffel,
    const tnsr::abb<DataType, SpatialDim, Frame, Index>& d_metric,
    const tnsr::AA<DataType, SpatialDim, Frame, Index>& inverse_metric);

template <size_t SpatialDim, typename Frame, IndexType Index, typename DataType>
auto christoffel_second_kind(
    const tnsr::abb<DataType, SpatialDim, Frame, Index>& d_metric,
    const tnsr::AA<DataType, SpatialDim, Frame, Index>& inverse_metric)
    -> tnsr::Abb<DataType, SpatialDim, Frame, Index>;
/// @}

namespace Tags {
/// Compute item for spatial Christoffel symbols of the first kind
/// \f$\Gamma_{ijk}\f$ computed from the first derivative of the
/// spatial metric.
///
/// Can be retrieved using `gr::Tags::SpatialChristoffelFirstKind`
template <typename DataType, size_t SpatialDim, typename Frame>
struct SpatialChristoffelFirstKindCompute
    : SpatialChristoffelFirstKind<DataType, SpatialDim, Frame>,
      db::ComputeTag {
  using argument_tags = tmpl::list<
      ::Tags::deriv<gr::Tags::SpatialMetric<DataType, SpatialDim, Frame>,
                    tmpl::size_t<SpatialDim>, Frame>>;

  using return_type = tnsr::ijj<DataType, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<
          tnsr::abb<DataType, SpatialDim, Frame, IndexType::Spatial>*>,
      const tnsr::ijj<DataType, SpatialDim, Frame>&)>(
      &christoffel_first_kind<SpatialDim, Frame, IndexType::Spatial, DataType>);

  using base = SpatialChristoffelFirstKind<DataType, SpatialDim, Frame>;
};

/// Compute item for spatial Christoffel symbols of the second kind
/// \f$\Gamma^i_{jk}\f$ computed from the Christoffel symbols of the
/// first kind and the inverse spatial metric.
///
/// Can be retrieved using `gr::Tags::SpatialChristoffelSecondKind`
template <typename DataType, size_t SpatialDim, typename Frame>
struct SpatialChristoffelSecondKindCompute
    : SpatialChristoffelSecondKind<DataType, SpatialDim, Frame>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<SpatialChristoffelFirstKind<DataType, SpatialDim, Frame>,
                 InverseSpatialMetric<DataType, SpatialDim, Frame>>;

  using return_type = tnsr::Ijj<DataType, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::Ijj<DataType, SpatialDim, Frame>*>,
      const tnsr::ijj<DataType, SpatialDim, Frame>&,
      const tnsr::II<DataType, SpatialDim, Frame>&)>(
      &raise_or_lower_first_index<DataType,
                                  SpatialIndex<SpatialDim, UpLo::Lo, Frame>,
                                  SpatialIndex<SpatialDim, UpLo::Lo, Frame>>);

  using base = SpatialChristoffelSecondKind<DataType, SpatialDim, Frame>;
};

/// Compute item for the trace of the spatial Christoffel symbols
/// of the first kind
/// \f$\Gamma_{i} = \Gamma_{ijk}\gamma^{jk}\f$ computed from the
/// Christoffel symbols of the first kind and the inverse spatial metric.
///
/// Can be retrieved using `gr::Tags::TraceSpatialChristoffelFirstKind`
template <typename DataType, size_t SpatialDim, typename Frame>
struct TraceSpatialChristoffelFirstKindCompute
    : TraceSpatialChristoffelFirstKind<DataType, SpatialDim, Frame>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<SpatialChristoffelFirstKind<DataType, SpatialDim, Frame>,
                 InverseSpatialMetric<DataType, SpatialDim, Frame>>;

  using return_type = tnsr::i<DataType, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::i<DataType, SpatialDim, Frame>*>,
      const tnsr::ijj<DataType, SpatialDim, Frame>&,
      const tnsr::II<DataType, SpatialDim, Frame>&)>(
      &trace_last_indices<DataType, SpatialIndex<SpatialDim, UpLo::Lo, Frame>,
                          SpatialIndex<SpatialDim, UpLo::Lo, Frame>>);

  using base = TraceSpatialChristoffelFirstKind<DataType, SpatialDim, Frame>;
};

/// Compute item for the trace of the spatial Christoffel symbols
/// of the second kind
/// \f$\Gamma^{i} = \Gamma^{i}_{jk}\gamma^{jk}\f$ computed from the
/// Christoffel symbols of the second kind and the inverse spatial metric.
///
/// Can be retrieved using `gr::Tags::TraceSpatialChristoffelSecondKind`
template <typename DataType, size_t SpatialDim, typename Frame>
struct TraceSpatialChristoffelSecondKindCompute
    : TraceSpatialChristoffelSecondKind<DataType, SpatialDim, Frame>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<SpatialChristoffelSecondKind<DataType, SpatialDim, Frame>,
                 InverseSpatialMetric<DataType, SpatialDim, Frame>>;

  using return_type = tnsr::I<DataType, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::I<DataType, SpatialDim, Frame>*>,
      const tnsr::Ijj<DataType, SpatialDim, Frame>&,
      const tnsr::II<DataType, SpatialDim, Frame>&)>(
      &trace_last_indices<DataType, SpatialIndex<SpatialDim, UpLo::Up, Frame>,
                          SpatialIndex<SpatialDim, UpLo::Lo, Frame>>);

  using base = TraceSpatialChristoffelSecondKind<DataType, SpatialDim, Frame>;
};

/// Compute item for spacetime Christoffel symbols of the first kind
/// \f$\Gamma_{abc}\f$ computed from the first derivative of the
/// spacetime metric.
///
/// Can be retrieved using `gr::Tags::SpacetimeChristoffelFirstKind`
template <typename DataType, size_t SpatialDim, typename Frame>
struct SpacetimeChristoffelFirstKindCompute
    : SpacetimeChristoffelFirstKind<DataType, SpatialDim, Frame>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<DerivativesOfSpacetimeMetric<DataType, SpatialDim, Frame>>;

  using return_type =
      tnsr::abb<DataType, SpatialDim, Frame, IndexType::Spacetime>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<
          tnsr::abb<DataType, SpatialDim, Frame, IndexType::Spacetime>*>,
      const tnsr::abb<DataType, SpatialDim, Frame, IndexType::Spacetime>&)>(
      &christoffel_first_kind<SpatialDim, Frame, IndexType::Spacetime,
                              DataType>);

  using base = SpacetimeChristoffelFirstKind<DataType, SpatialDim, Frame>;
};

/// Compute item for spacetime Christoffel symbols of the second kind
/// \f$\Gamma^a_{bc}\f$ computed from the Christoffel symbols of the
/// first kind and the inverse spacetime metric.
///
/// Can be retrieved using `gr::Tags::SpacetimeChristoffelSecondKind`
template <typename DataType, size_t SpatialDim, typename Frame>
struct SpacetimeChristoffelSecondKindCompute
    : SpacetimeChristoffelSecondKind<DataType, SpatialDim, Frame>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<SpacetimeChristoffelFirstKind<DataType, SpatialDim, Frame>,
                 InverseSpacetimeMetric<DataType, SpatialDim, Frame>>;

  using return_type = tnsr::Abb<DataType, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::Abb<DataType, SpatialDim, Frame>*>,
      const tnsr::abb<DataType, SpatialDim, Frame>&,
      const tnsr::AA<DataType, SpatialDim, Frame>&)>(
      &raise_or_lower_first_index<DataType,
                                  SpacetimeIndex<SpatialDim, UpLo::Lo, Frame>,
                                  SpacetimeIndex<SpatialDim, UpLo::Lo, Frame>>);

  using base = SpacetimeChristoffelSecondKind<DataType, SpatialDim, Frame>;
};

/// Compute item for the trace of the spacetime Christoffel symbols
/// of the first kind
/// \f$\Gamma_{a} = \Gamma_{abc}g^{bc}\f$ computed from the
/// Christoffel symbols of the first kind and the inverse spacetime metric.
///
/// Can be retrieved using `gr::Tags::TraceSpacetimeChristoffelFirstKind`
template <typename DataType, size_t SpatialDim, typename Frame>
struct TraceSpacetimeChristoffelFirstKindCompute
    : TraceSpacetimeChristoffelFirstKind<DataType, SpatialDim, Frame>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<SpacetimeChristoffelFirstKind<DataType, SpatialDim, Frame>,
                 InverseSpacetimeMetric<DataType, SpatialDim, Frame>>;

  using return_type = tnsr::a<DataType, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*>,
      const tnsr::abb<DataType, SpatialDim, Frame>&,
      const tnsr::AA<DataType, SpatialDim, Frame>&)>(
      &trace_last_indices<DataType, SpacetimeIndex<SpatialDim, UpLo::Lo, Frame>,
                          SpacetimeIndex<SpatialDim, UpLo::Lo, Frame>>);

  using base = TraceSpacetimeChristoffelFirstKind<DataType, SpatialDim, Frame>;
};
}  // namespace Tags
}  // namespace gr
