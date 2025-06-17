// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain::Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace domain::Tags
class DataVector;
template <typename X, typename Symm, typename IndexList>
class Tensor;
/// \endcond

namespace gh {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the expansion \f$\Theta\f$ for 1D apparent horizon finding.
 *
 * \details Calculate the expansion to find apparent horizon for 1D
 * cartoon domain. Implements Eq. (7.41) of \cite BaumgarteShapiro,
 * \f$\Theta\ = -\frac{1}{\sqrt{2}} m^{ij} \Big( s_k \Gamma^k_{ij}
 * + K_{ij} \Big) \f$, assuming the calculations are performed on the Cartesian
 * x axis.
 */
template <typename DataType, typename Frame>
void expansion1D(gsl::not_null<Scalar<DataType>*> expansion,
                 const tnsr::ii<DataType, 3, Frame>& spatial_metric,
                 const tnsr::ijj<DataType, 3, Frame>& deriv_spatial_metric,
                 const tnsr::ii<DataType, 3, Frame>& ext_curvature,
                 const tnsr::I<DataType, 3, Frame>& coords);

template <typename DataType, typename Frame>
Scalar<DataType> expansion1D(
    const tnsr::ii<DataType, 3, Frame>& spatial_metric,
    const tnsr::ijj<DataType, 3, Frame>& deriv_spatial_metric,
    const tnsr::ii<DataType, 3, Frame>& ext_curvature,
    const tnsr::I<DataType, 3, Frame>& coords);
/// @}

namespace Tags {
template <typename DataType>
struct Expansion1D : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief Compute item for the 1D expansion.
 *
 * \details Calculate the expansion for spherically symmetric cartoon horizon
 * finding.  See `expansion1D()`. Can be retrieved using
 * `gh::Tags::Expansion1D`.
 */
template <typename Frame>
struct Expansion1DCompute : Expansion1D<DataVector>, db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::SpatialMetric<DataVector, 3, Frame>,
                 ::Tags::deriv<gr::Tags::SpatialMetric<DataVector, 3, Frame>,
                               tmpl::size_t<3>, Frame>,
                 gr::Tags::ExtrinsicCurvature<DataVector, 3, Frame>,
                 domain::Tags::Coordinates<3, Frame> >;

  using return_type = Scalar<DataVector>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataVector>*>, const tnsr::ii<DataVector, 3, Frame>&,
      const tnsr::ijj<DataVector, 3, Frame>&,
      const tnsr::ii<DataVector, 3, Frame>&,
      const tnsr::I<DataVector, 3, Frame>&)>(&expansion1D<DataVector, Frame>);
  using base = Expansion1D<DataVector>;
};
}  // namespace Tags
}  // namespace gh
