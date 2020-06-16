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
// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes time derivative of the spatial metric.
 *
 * \details Let the generalized harmonic conjugate momentum and spatial
 * derivative variables be \f$\Pi_{ab} = -t^c \partial_c \psi_{ab} \f$ and
 * \f$\Phi_{iab} = \partial_i \psi_{ab} \f$. As \f$ t_i \equiv 0 \f$. The time
 * derivative of the spatial metric is given by the time derivative of the
 * spatial sector of the spacetime metric, i.e.
 * \f$ \partial_0 g_{ij} = \partial_0 \psi_{ij} \f$.
 *
 * To compute the latter, we use the evolution equation for \f$ \psi_{ij} \f$,
 * c.f. eq.(35) of \cite Lindblom2005qh (with \f$\gamma_1 = -1\f$):
 *
 * \f[
 * \partial_0 \psi_{ab} = - N \Pi_{ab} + N^k \Phi_{kab}
 * \f]
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void time_deriv_of_spatial_metric(
    gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*> dt_spatial_metric,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> time_deriv_of_spatial_metric(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept;
// @}

namespace Tags {
/*!
 * \brief Compute item to get time derivative of the spatial metric from
 *        generalized harmonic and geometric variables
 *
 * \details See `time_deriv_of_spatial_metric()`. Can be retrieved using
 * `gr::Tags::SpatialMetric` wrapped in `Tags::dt`.
 */
template <size_t SpatialDim, typename Frame>
struct TimeDerivSpatialMetricCompute
    : ::Tags::dt<gr::Tags::SpatialMetric<SpatialDim, Frame, DataVector>>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<SpatialDim, Frame, DataVector>,
                 Phi<SpatialDim, Frame>, Pi<SpatialDim, Frame>>;

  using return_type = tnsr::ii<DataVector, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::ii<DataVector, SpatialDim, Frame>*>,
      const Scalar<DataVector>&, const tnsr::I<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&,
      const tnsr::aa<DataVector, SpatialDim, Frame>&) noexcept>(
      &time_deriv_of_spatial_metric<SpatialDim, Frame>);

  using base =
      ::Tags::dt<gr::Tags::SpatialMetric<SpatialDim, Frame, DataVector>>;
};
}  // namespace Tags
}  // namespace GeneralizedHarmonic
