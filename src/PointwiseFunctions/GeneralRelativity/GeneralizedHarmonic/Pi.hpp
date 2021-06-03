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

namespace GeneralizedHarmonic {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the conjugate momentum \f$\Pi_{ab}\f$ of the spacetime metric
 * \f$ \psi_{ab} \f$.
 *
 * \details If \f$ N, N^i\f$ are the lapse and shift
 * respectively, and \f$ \Phi_{iab} = \partial_i \psi_{ab} \f$ then
 * \f$\Pi_{\mu\nu} = -\frac{1}{N} ( \partial_t \psi_{\mu\nu}  -
 *      N^m \Phi_{m\mu\nu}) \f$ where \f$ \partial_t \psi_{ab} \f$ is computed
 * as
 *
 * \f{align}
 *     \partial_t \psi_{tt} &= - 2 N \partial_t N
 *                 + 2 g_{mn} N^m \partial_t N^n
 *                 + N^m N^n \partial_t g_{mn} \\
 *     \partial_t \psi_{ti} &= g_{mi} \partial_t N^m
 *                 + N^m \partial_t g_{mi} \\
 *     \partial_t \psi_{ij} &= \partial_t g_{ij}
 * \f}
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void pi(gsl::not_null<tnsr::aa<DataType, SpatialDim, Frame>*> pi,
        const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
        const tnsr::I<DataType, SpatialDim, Frame>& shift,
        const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
        const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
        const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
        const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::aa<DataType, SpatialDim, Frame> pi(
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;
/// @}

namespace Tags {
/*!
 * \brief Compute item the conjugate momentum \f$\Pi_{ab}\f$ of the spacetime
 * metric \f$ \psi_{ab} \f$.
 *
 * \details See `pi()`. Can be retrieved using `GeneralizedHarmonic::Tags::Pi`.
 */
template <size_t SpatialDim, typename Frame>
struct PiCompute : Pi<SpatialDim, Frame>, db::ComputeTag {
  using argument_tags = tmpl::list<
      gr::Tags::Lapse<DataVector>, ::Tags::dt<gr::Tags::Lapse<DataVector>>,
      gr::Tags::Shift<SpatialDim, Frame, DataVector>,
      ::Tags::dt<gr::Tags::Shift<SpatialDim, Frame, DataVector>>,
      gr::Tags::SpatialMetric<SpatialDim, Frame, DataVector>,
      ::Tags::dt<gr::Tags::SpatialMetric<SpatialDim, Frame, DataVector>>,
      Phi<SpatialDim, Frame>>;

  using return_type = tnsr::aa<DataVector, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::aa<DataVector, SpatialDim, Frame>*>,
      const Scalar<DataVector>&, const Scalar<DataVector>&,
      const tnsr::I<DataVector, SpatialDim, Frame>&,
      const tnsr::I<DataVector, SpatialDim, Frame>&,
      const tnsr::ii<DataVector, SpatialDim, Frame>&,
      const tnsr::ii<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&) noexcept>(
      &pi<SpatialDim, Frame, DataVector>);

  using base = Pi<SpatialDim, Frame>;
};
}  // namespace Tags
}  // namespace GeneralizedHarmonic
