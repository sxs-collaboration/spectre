// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
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

namespace gh {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the conjugate momentum \f$\Pi_{ab}\f$ of the spacetime metric
 * \f$ g_{ab} \f$.
 *
 * \details If \f$ \alpha, \beta^i\f$ are the lapse and shift respectively, and
 * \f$ \Phi_{iab} = \partial_i g_{ab} \f$ then
 * \f$\Pi_{\mu\nu} = -\frac{1}{\alpha} ( \partial_t g_{\mu\nu}  -
 *      \beta^m \Phi_{m\mu\nu}) \f$ where \f$ \partial_t g_{ab} \f$ is computed
 * as
 *
 * \f{align}
 *     \partial_t g_{tt} &= - 2 \alpha \partial_t \alpha
 *                 + 2 \gamma_{mn} \beta^m \partial_t \beta^n
 *                 + \beta^m \beta^n \partial_t \gamma_{mn} \\
 *     \partial_t g_{ti} &= \gamma_{mi} \partial_t \beta^m
 *                 + \beta^m \partial_t \gamma_{mi} \\
 *     \partial_t g_{ij} &= \partial_t \gamma_{ij}
 * \f}
 */
template <typename DataType, size_t SpatialDim, typename Frame>
void pi(gsl::not_null<tnsr::aa<DataType, SpatialDim, Frame>*> pi,
        const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
        const tnsr::I<DataType, SpatialDim, Frame>& shift,
        const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
        const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
        const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
        const tnsr::iaa<DataType, SpatialDim, Frame>& phi);

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::aa<DataType, SpatialDim, Frame> pi(
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi);
/// @}

namespace Tags {
/*!
 * \brief Compute item the conjugate momentum \f$\Pi_{ab}\f$ of the spacetime
 * metric \f$ g_{ab} \f$.
 *
 * \details See `pi()`. Can be retrieved using `gh::Tags::Pi`.
 */
template <size_t SpatialDim, typename Frame>
struct PiCompute : Pi<DataVector, SpatialDim, Frame>, db::ComputeTag {
  using argument_tags = tmpl::list<
      gr::Tags::Lapse<DataVector>, ::Tags::dt<gr::Tags::Lapse<DataVector>>,
      gr::Tags::Shift<DataVector, SpatialDim, Frame>,
      ::Tags::dt<gr::Tags::Shift<DataVector, SpatialDim, Frame>>,
      gr::Tags::SpatialMetric<DataVector, SpatialDim, Frame>,
      ::Tags::dt<gr::Tags::SpatialMetric<DataVector, SpatialDim, Frame>>,
      Phi<DataVector, SpatialDim, Frame>>;

  using return_type = tnsr::aa<DataVector, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::aa<DataVector, SpatialDim, Frame>*>,
      const Scalar<DataVector>&, const Scalar<DataVector>&,
      const tnsr::I<DataVector, SpatialDim, Frame>&,
      const tnsr::I<DataVector, SpatialDim, Frame>&,
      const tnsr::ii<DataVector, SpatialDim, Frame>&,
      const tnsr::ii<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&)>(
      &pi<DataVector, SpatialDim, Frame>);

  using base = Pi<DataVector, SpatialDim, Frame>;
};
}  // namespace Tags
}  // namespace gh
