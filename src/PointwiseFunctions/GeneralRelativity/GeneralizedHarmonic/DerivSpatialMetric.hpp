// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tags::deriv

/// \cond
namespace domain {
namespace Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace Tags
}  // namespace domain
class DataVector;
template <typename X, typename Symm, typename IndexList>
class Tensor;
/// \endcond

namespace GeneralizedHarmonic {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes spatial derivatives of the spatial metric from
 *        the generalized harmonic spatial derivative variable.
 *
 * \details If \f$ \Phi_{kab} \f$ is the generalized
 * harmonic spatial derivative variable, then the derivatives of the
 * spatial metric are
 * \f[
 *      \partial_k g_{ij} = \Phi_{kij}
 * \f]
 *
 * This quantity is needed for computing spatial Christoffel symbols.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void deriv_spatial_metric(
    gsl::not_null<tnsr::ijj<DataType, SpatialDim, Frame>*> d_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi);

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ijj<DataType, SpatialDim, Frame> deriv_spatial_metric(
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi);
/// @}

namespace Tags {
/*!
 * \brief Compute item to get spatial derivatives of the spatial metric from
 *        the generalized harmonic spatial derivative variable.
 *
 * \details See `deriv_spatial_metric()`. Can be retrieved using
 * `gr::Tags::SpatialMetric` wrapped in `::Tags::deriv`.
 */
template <size_t SpatialDim, typename Frame>
struct DerivSpatialMetricCompute
    : ::Tags::deriv<gr::Tags::SpatialMetric<SpatialDim, Frame, DataVector>,
                    tmpl::size_t<SpatialDim>, Frame>,
      db::ComputeTag {
  using argument_tags = tmpl::list<Phi<SpatialDim, Frame>>;

  using return_type = tnsr::ijj<DataVector, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::ijj<DataVector, SpatialDim, Frame>*>,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&)>(
      &deriv_spatial_metric<SpatialDim, Frame>);

  using base =
      ::Tags::deriv<gr::Tags::SpatialMetric<SpatialDim, Frame, DataVector>,
                    tmpl::size_t<SpatialDim>, Frame>;
};
}  // namespace Tags
}  // namespace GeneralizedHarmonic
