// Distributed under the MIT License.
// See LICENSE.txt for details.

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
 * \brief Computes a quantity measuring how far from type D spacetime is.
 *
 * \details Computes a quantity measuring how far from type D spacetime is,
 * using measure D1 [Eq. (8) of \cite Bhagwat2017tkm]:
 *
 * \f{align}{
 * \frac{a}{12} \gamma_{ij} - \frac{b}{a} E_{ij} - 4
 * E_{i}^{k} E_{jk} = 0 \f}
 *
 * where \f$\gamma_{ij}\f$ is the spatial metric and \f$E_{ij}\f$ is the
 * electric part ofthe Weyl tensor.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::ii<DataType, SpatialDim, Frame> weyl_type_D1(
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_electric,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric);

template <typename DataType, size_t SpatialDim, typename Frame>
void weyl_type_D1(
    const gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*> weyl_type_D1,
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_electric,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric);
/// @}

/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the scalar \f$D_{ij} D^{ij}\f$ , a measure of a spacetime's
 * devitation from type D.
 *
 * \details Computes the scalar \f$D_{ij} D^{ij}\f$ from \f$D_{ij}\f$ (Eq. (8)
 * of \cite Bhagwat2017tkm] and the inverse spatial metric \f$\gamma^{ij}\f$,
 * i.e. \f$D = \gamma^{ik}\gamma^{jl}D_{ij}D_{kl}\f$.
 *
 * \note The Weyl Type D1 \f$D_{ij}\f$ is available via gr::weyl_type_D1.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
void weyl_type_D1_scalar(
    const gsl::not_null<Scalar<DataType>*> weyl_type_D1_scalar_result,
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_type_D1,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric);

template <typename DataType, size_t SpatialDim, typename Frame>
Scalar<DataType> weyl_type_D1_scalar(
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_type_D1,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric);

/// @}

namespace Tags {
/// Compute item for WeylTypeD1
/// Computed from WeylElectric, SpatialMetric, and InverseSpatialMetric
///
/// Can be retrieved using gr::Tags::WeylTypeD1
template <typename DataType, size_t SpatialDim, typename Frame>
struct WeylTypeD1Compute : WeylTypeD1<DataType, SpatialDim, Frame>,
                           db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::WeylElectric<DataType, SpatialDim, Frame>,
                 gr::Tags::SpatialMetric<DataType, SpatialDim, Frame>,
                 gr::Tags::InverseSpatialMetric<DataType, SpatialDim, Frame>>;

  using return_type = tnsr::ii<DataType, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*>,
      const tnsr::ii<DataType, SpatialDim, Frame>&,
      const tnsr::ii<DataType, SpatialDim, Frame>&,
      const tnsr::II<DataType, SpatialDim, Frame>&)>(
      &weyl_type_D1<DataType, SpatialDim, Frame>);

  using base = WeylTypeD1<DataType, SpatialDim, Frame>;
};

/// Can be retrieved using gr::Tags::WeylTypeD1Scalar
template <typename DataType, size_t SpatialDim, typename Frame>
struct WeylTypeD1ScalarCompute : WeylTypeD1Scalar<DataType>, db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::WeylTypeD1Compute<DataType, SpatialDim, Frame>,
                 gr::Tags::InverseSpatialMetric<DataType, SpatialDim, Frame>>;

  using return_type = Scalar<DataType>;

  static constexpr auto function =
      static_cast<void (*)(gsl::not_null<Scalar<DataType>*>,
                           const tnsr::ii<DataType, SpatialDim, Frame>&,
                           const tnsr::II<DataType, SpatialDim, Frame>&)>(
          &gr::weyl_type_D1_scalar<DataType, SpatialDim, Frame>);

  using base = WeylTypeD1Scalar<DataType>;
};
}  // namespace Tags
}  // namespace gr
