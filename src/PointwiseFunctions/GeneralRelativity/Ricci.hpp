// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Declares function templates to calculate the Ricci tensor

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

namespace gr {

/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes Ricci tensor from the (spatial or spacetime)
 * Christoffel symbol of the second kind and its derivative.
 *
 * \details Computes Ricci tensor \f$R_{ab}\f$ as:
 * \f$ R_{ab} = \frac{1}{2} ( \partial_c \Gamma^{c}_{ab} - \partial_{(b}
 * \Gamma^{c}_{a)c}
 * + \Gamma^{d}_{ab}\Gamma^{c}_{cd} - \Gamma^{d}_{ac} \Gamma^{c}_{bd}  ) \f$
 * where \f$\Gamma^{a}_{bc}\f$ is the Christoffel symbol of the second kind.
 */
template <size_t SpatialDim, typename Frame, IndexType Index, typename DataType>
tnsr::aa<DataType, SpatialDim, Frame, Index> ricci_tensor(
    const tnsr::Abb<DataType, SpatialDim, Frame, Index>& christoffel_2nd_kind,
    const tnsr::aBcc<DataType, SpatialDim, Frame, Index>&
        d_christoffel_2nd_kind) noexcept;

namespace Tags {
/// Compute item for spatial Ricci tensor \f$R_{ij}\f$
/// computed from SpatialChristoffelSecondKind and its spatial derivatives.
///
/// Can be retrieved using `gr::Tags::RicciTensor`
template <size_t SpatialDim, typename Frame, typename DataType>
struct RicciTensorCompute : RicciTensor<SpatialDim, Frame, DataType>,
                            db::ComputeTag {
  using argument_tags = tmpl::list<
      gr::Tags::SpatialChristoffelSecondKind<SpatialDim, Frame, DataType>,
      ::Tags::deriv<
          gr::Tags::SpatialChristoffelSecondKind<SpatialDim, Frame, DataType>,
          tmpl::size_t<SpatialDim>, Frame>>;
  static constexpr tnsr::ii<DataType, SpatialDim, Frame> (*function)(
      const tnsr::Ijj<DataType, SpatialDim, Frame>&,
      const tnsr::iJkk<DataType, SpatialDim, Frame>&) =
      &ricci_tensor<SpatialDim, Frame, IndexType::Spatial, DataType>;
  using base = RicciTensor<SpatialDim, Frame, DataType>;
};
}  // namespace Tags
} // namespace gr
