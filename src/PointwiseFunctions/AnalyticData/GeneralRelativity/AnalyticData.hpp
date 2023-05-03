// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"

namespace gr {
/// Base struct for properties common to all GR analytic data classes
template <size_t Dim>
struct AnalyticDataBase {
  static constexpr size_t volume_dim = Dim;
  template <typename DataType, typename Frame = ::Frame::Inertial>
  using DerivLapse =
      ::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<volume_dim>, Frame>;
  template <typename DataType, typename Frame = ::Frame::Inertial>
  using DerivShift = ::Tags::deriv<gr::Tags::Shift<DataType, volume_dim, Frame>,
                                   tmpl::size_t<volume_dim>, Frame>;
  template <typename DataType, typename Frame = ::Frame::Inertial>
  using DerivSpatialMetric =
      ::Tags::deriv<gr::Tags::SpatialMetric<DataType, volume_dim, Frame>,
                    tmpl::size_t<volume_dim>, Frame>;

  template <typename DataType, typename Frame = ::Frame::Inertial>
  using tags = tmpl::list<
      gr::Tags::Lapse<DataType>, ::Tags::dt<gr::Tags::Lapse<DataType>>,
      DerivLapse<DataType, Frame>, gr::Tags::Shift<DataType, volume_dim, Frame>,
      ::Tags::dt<gr::Tags::Shift<DataType, volume_dim, Frame>>,
      DerivShift<DataType, Frame>,
      gr::Tags::SpatialMetric<DataType, volume_dim, Frame>,
      ::Tags::dt<gr::Tags::SpatialMetric<DataType, volume_dim, Frame>>,
      DerivSpatialMetric<DataType, Frame>,
      gr::Tags::SqrtDetSpatialMetric<DataType>,
      gr::Tags::ExtrinsicCurvature<DataType, volume_dim, Frame>,
      gr::Tags::InverseSpatialMetric<DataType, volume_dim, Frame>>;
};

/*!
 * \ingroup AnalyticDataGroup
 * \brief Classes which implement analytic data for general relativity
 */
namespace AnalyticData {}
}  // namespace gr
