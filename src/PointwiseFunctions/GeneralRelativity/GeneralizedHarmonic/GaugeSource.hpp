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
 * \brief  Computes generalized harmonic gauge source function.
 * \details If \f$N, N^i, g_{ij}, \Gamma_{ijk}, K\f$ are the lapse, shift,
 * spatial metric, spatial Christoffel symbols, and trace of the extrinsic
 * curvature, then we compute
 * \f{align}
 * H_l &= N^{-2} g_{il}(\partial_t N^i - N^k \partial_k N^i)
 * + N^{-1} \partial_l N - g^{km}\Gamma_{lkm} \\
 * H_0 &= -N^{-1} \partial_t N + N^{-1} N^k\partial_k N + N^k H_k - N K
 * \f}
 * See Eqs. 8 and 9 of \cite Lindblom2005qh
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void gauge_source(
    gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> gauge_source_h,
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const Scalar<DataType>& trace_extrinsic_curvature,
    const tnsr::i<DataType, SpatialDim, Frame>& trace_christoffel_last_indices);

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> gauge_source(
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const Scalar<DataType>& trace_extrinsic_curvature,
    const tnsr::i<DataType, SpatialDim, Frame>& trace_christoffel_last_indices);
/// @}

namespace Tags {
/*!
 * \brief  Compute item to get the implicit gauge source function from 3 + 1
 * quantities.
 *
 * \details See `gauge_source()`. Can be retrieved using
 * `GeneralizedHarmonic::Tags::GaugeH`.
 */
template <size_t SpatialDim, typename Frame>
struct GaugeHImplicitFrom3p1QuantitiesCompute : GaugeH<SpatialDim, Frame>,
                                                db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>,
                 ::Tags::dt<gr::Tags::Lapse<DataVector>>,
                 ::Tags::deriv<gr::Tags::Lapse<DataVector>,
                               tmpl::size_t<SpatialDim>, Frame>,
                 gr::Tags::Shift<SpatialDim, Frame, DataVector>,
                 ::Tags::dt<gr::Tags::Shift<SpatialDim, Frame, DataVector>>,
                 ::Tags::deriv<gr::Tags::Shift<SpatialDim, Frame, DataVector>,
                               tmpl::size_t<SpatialDim>, Frame>,
                 gr::Tags::SpatialMetric<SpatialDim, Frame, DataVector>,
                 gr::Tags::TraceExtrinsicCurvature<DataVector>,
                 gr::Tags::TraceSpatialChristoffelFirstKind<SpatialDim, Frame,
                                                            DataVector>>;

  using return_type = tnsr::a<DataVector, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::a<DataVector, SpatialDim, Frame>*>,
      const Scalar<DataVector>&, const Scalar<DataVector>&,
      const tnsr::i<DataVector, SpatialDim, Frame>&,
      const tnsr::I<DataVector, SpatialDim, Frame>&,
      const tnsr::I<DataVector, SpatialDim, Frame>&,
      const tnsr::iJ<DataVector, SpatialDim, Frame>&,
      const tnsr::ii<DataVector, SpatialDim, Frame>&, const Scalar<DataVector>&,
      const tnsr::i<DataVector, SpatialDim, Frame>&)>(
      &gauge_source<SpatialDim, Frame, DataVector>);

  using base = GaugeH<SpatialDim, Frame>;
};

/*!
 * \brief  Compute item to get spacetime derivative of the gauge source function
 * from its spatial and time derivatives.
 *
 * \details Can be retrieved using
 * `GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH`.
 */
template <size_t SpatialDim, typename Frame>
struct SpacetimeDerivGaugeHCompute : SpacetimeDerivGaugeH<SpatialDim, Frame>,
                                     db::ComputeTag {
  using argument_tags = tmpl::list<
      ::Tags::dt<GeneralizedHarmonic::Tags::GaugeH<SpatialDim, Frame>>,
      ::Tags::deriv<GeneralizedHarmonic::Tags::GaugeH<SpatialDim, Frame>,
                    tmpl::size_t<SpatialDim>, Frame>>;

  using return_type = tnsr::ab<DataVector, SpatialDim, Frame>;

  static constexpr void function(
      const gsl::not_null<tnsr::ab<DataVector, SpatialDim, Frame>*>
          spacetime_deriv_gauge_source,
      const tnsr::a<DataVector, SpatialDim, Frame>& time_deriv_gauge_source,
      const tnsr::ia<DataVector, SpatialDim, Frame>& deriv_gauge_source) {
    destructive_resize_components(spacetime_deriv_gauge_source,
                                  get<0>(time_deriv_gauge_source).size());
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      spacetime_deriv_gauge_source->get(0, b) = time_deriv_gauge_source.get(b);
      for (size_t a = 1; a < SpatialDim + 1; ++a) {
        spacetime_deriv_gauge_source->get(a, b) =
            deriv_gauge_source.get(a - 1, b);
      }
    }
  }

  using base = SpacetimeDerivGaugeH<SpatialDim, Frame>;
};
}  // namespace Tags
}  // namespace GeneralizedHarmonic
