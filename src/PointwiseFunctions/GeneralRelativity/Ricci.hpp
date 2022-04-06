// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Declares function templates to calculate the Ricci tensor

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace gr {

/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes Ricci tensor from the (spatial or spacetime)
 * Christoffel symbol of the second kind and its derivative.
 *
 * \details Computes Ricci tensor \f$R_{ab}\f$ as:
 * \f$ R_{ab} = \partial_c \Gamma^{c}_{ab} - \partial_{(b} \Gamma^{c}_{a)c}
 * + \Gamma^{d}_{ab}\Gamma^{c}_{cd} - \Gamma^{d}_{ac} \Gamma^{c}_{bd} \f$
 * where \f$\Gamma^{a}_{bc}\f$ is the Christoffel symbol of the second kind.
 */
template <size_t SpatialDim, typename Frame, IndexType Index, typename DataType>
void ricci_tensor(
    gsl::not_null<tnsr::aa<DataType, SpatialDim, Frame, Index>*> result,
    const tnsr::Abb<DataType, SpatialDim, Frame, Index>& christoffel_2nd_kind,
    const tnsr::aBcc<DataType, SpatialDim, Frame, Index>&
        d_christoffel_2nd_kind);

template <size_t SpatialDim, typename Frame, IndexType Index, typename DataType>
tnsr::aa<DataType, SpatialDim, Frame, Index> ricci_tensor(
    const tnsr::Abb<DataType, SpatialDim, Frame, Index>& christoffel_2nd_kind,
    const tnsr::aBcc<DataType, SpatialDim, Frame, Index>&
        d_christoffel_2nd_kind);
/// @}

/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the Ricci Scalar from the (spatial or spacetime) Ricci Tensor
 * and inverse metricss.
 *
 * \details Computes Ricci scalar using the inverse metric (spatial or
 * spacetime) and Ricci tensor \f$R = \gamma^{ab}R_{ab}\f$
 *
 *
 */
template <size_t SpatialDim, typename Frame, IndexType Index, typename DataType>
void ricci_scalar(
    const gsl::not_null<Scalar<DataType>*> ricci_scalar_result,
    const tnsr::aa<DataType, SpatialDim, Frame, Index>& ricci_tensor,
    const tnsr::AA<DataType, SpatialDim, Frame, Index>& inverse_metric);

template <size_t SpatialDim, typename Frame, IndexType Index, typename DataType>
Scalar<DataType> ricci_scalar(
    const tnsr::aa<DataType, SpatialDim, Frame, Index>& ricci_tensor,
    const tnsr::AA<DataType, SpatialDim, Frame, Index>& inverse_metric);
/// @}

namespace Tags {
/// Compute item for spatial Ricci tensor \f$R_{ij}\f$
/// computed from SpatialChristoffelSecondKind and its spatial derivatives.
///
/// Can be retrieved using `gr::Tags::SpatialRicci`
template <size_t SpatialDim, typename Frame, typename DataType>
struct SpatialRicciCompute : SpatialRicci<SpatialDim, Frame, DataType>,
                             db::ComputeTag {
  using argument_tags = tmpl::list<
      gr::Tags::SpatialChristoffelSecondKind<SpatialDim, Frame, DataType>,
      ::Tags::deriv<
          gr::Tags::SpatialChristoffelSecondKind<SpatialDim, Frame, DataType>,
          tmpl::size_t<SpatialDim>, Frame>>;

  using return_type = tnsr::ii<DataType, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*>,
      const tnsr::Ijj<DataType, SpatialDim, Frame>&,
      const tnsr::iJkk<DataType, SpatialDim, Frame>&)>(
      &ricci_tensor<SpatialDim, Frame, IndexType::Spatial, DataType>);

  using base = SpatialRicci<SpatialDim, Frame, DataType>;
};

/// Computes the patial ricci scalar using the spatial ricci tensor and the
/// inverse spatial metric.
///
/// Can be retrieved using 'gr::Tags::SpatialRicciScalar'
template <size_t SpatialDim, typename Frame, typename DataType>
struct SpatialRicciScalarCompute : SpatialRicciScalar<DataType>,
                                   db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::SpatialRicci<SpatialDim, Frame, DataType>,
                 gr::Tags::InverseSpatialMetric<SpatialDim, Frame, DataType>>;

  using return_type = Scalar<DataType>;

  static constexpr auto function =
      static_cast<void (*)(gsl::not_null<Scalar<DataType>*>,
                           const tnsr::ii<DataType, SpatialDim, Frame>&,
                           const tnsr::II<DataType, SpatialDim, Frame>&)>(
          &ricci_scalar<SpatialDim, Frame, IndexType::Spatial, DataType>);

  using base = SpatialRicciScalar<DataType>;
};
}  // namespace Tags
} // namespace gr
