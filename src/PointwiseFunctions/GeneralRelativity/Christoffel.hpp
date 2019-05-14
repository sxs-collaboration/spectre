// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Defines functions to calculate Christoffel symbols

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

/// \cond
namespace gsl {
template <class>
class not_null;
}  // namespace gsl
/// \endcond

namespace gr {
// @{
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
    const tnsr::abb<DataType, SpatialDim, Frame, Index>& d_metric) noexcept;

template <size_t SpatialDim, typename Frame, IndexType Index, typename DataType>
tnsr::abb<DataType, SpatialDim, Frame, Index> christoffel_first_kind(
    const tnsr::abb<DataType, SpatialDim, Frame, Index>& d_metric) noexcept;
// @}
namespace Tags {
/// Compute item for spatial Christoffel symbols of the first kind
/// \f$\Gamma_{ijk}\f$ computed from the first derivative of the
/// spatial metric.
///
/// Can be retrieved using `gr::Tags::SpatialChristoffelFirstKind`
template <size_t SpatialDim, typename Frame, typename DataType>
struct SpatialChristoffelFirstKindCompute
    : SpatialChristoffelFirstKind<SpatialDim, Frame, DataType>,
      db::ComputeTag {
  using argument_tags = tmpl::list<
      ::Tags::deriv<gr::Tags::SpatialMetric<SpatialDim, Frame, DataType>,
                    tmpl::size_t<SpatialDim>, Frame>>;
  static constexpr tnsr::ijj<DataType, SpatialDim, Frame> (*function)(
      const tnsr::ijj<DataType, SpatialDim, Frame>&) =
      &christoffel_first_kind<SpatialDim, Frame, IndexType::Spatial, DataType>;
  using base = SpatialChristoffelFirstKind<SpatialDim, Frame, DataType>;
};

/// Compute item for spatial Christoffel symbols of the second kind
/// \f$\Gamma^i_{jk}\f$ computed from the Christoffel symbols of the
/// first kind and the inverse spatial metric.
///
/// Can be retrieved using `gr::Tags::SpatialChristoffelSecondKind`
template <size_t SpatialDim, typename Frame, typename DataType>
struct SpatialChristoffelSecondKindCompute
    : SpatialChristoffelSecondKind<SpatialDim, Frame, DataType>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<SpatialChristoffelFirstKind<SpatialDim, Frame, DataType>,
                 InverseSpatialMetric<SpatialDim, Frame, DataType>>;
  static constexpr tnsr::Ijj<DataType, SpatialDim, Frame> (*function)(
      const tnsr::ijj<DataType, SpatialDim, Frame>&,
      const tnsr::II<DataType, SpatialDim, Frame>&) =
      &raise_or_lower_first_index<DataType,
                                  SpatialIndex<SpatialDim, UpLo::Lo, Frame>,
                                  SpatialIndex<SpatialDim, UpLo::Lo, Frame>>;
  using base = SpatialChristoffelSecondKind<SpatialDim, Frame, DataType>;
};

/// Compute item for the trace of the spatial Christoffel symbols
/// of the first kind
/// \f$\Gamma_{i} = \Gamma_{ijk}g^{jk}\f$ computed from the
/// Christoffel symbols of the first kind and the inverse spatial metric.
///
/// Can be retrieved using `gr::Tags::TraceSpatialChristoffelFirstKind`
template <size_t SpatialDim, typename Frame, typename DataType>
struct TraceSpatialChristoffelFirstKindCompute
    : TraceSpatialChristoffelFirstKind<SpatialDim, Frame, DataType>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<SpatialChristoffelFirstKind<SpatialDim, Frame, DataType>,
                 InverseSpatialMetric<SpatialDim, Frame, DataType>>;
  static constexpr tnsr::i<DataType, SpatialDim, Frame> (*function)(
      const tnsr::ijj<DataType, SpatialDim, Frame>&,
      const tnsr::II<DataType, SpatialDim, Frame>&) =
      &trace_last_indices<DataType, SpatialIndex<SpatialDim, UpLo::Lo, Frame>,
                          SpatialIndex<SpatialDim, UpLo::Lo, Frame>>;
  using base = TraceSpatialChristoffelFirstKind<SpatialDim, Frame, DataType>;
};
}  // namespace Tags
}  // namespace gr
