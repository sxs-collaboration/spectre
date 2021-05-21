// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"

namespace gr {
/// Base struct for properties common to all GR analytic solutions
template <size_t Dim>
struct AnalyticSolution {
  static constexpr size_t volume_dim = Dim;
  template <typename DataType, typename Frame = ::Frame::Inertial>
  using DerivLapse = ::Tags::deriv<gr::Tags::Lapse<DataType>,
                                   tmpl::size_t<volume_dim>, Frame>;
  template <typename DataType, typename Frame = ::Frame::Inertial>
  using DerivShift =
      ::Tags::deriv<gr::Tags::Shift<volume_dim, Frame, DataType>,
                    tmpl::size_t<volume_dim>, Frame>;
  template <typename DataType, typename Frame = ::Frame::Inertial>
  using DerivSpatialMetric = ::Tags::deriv<
      gr::Tags::SpatialMetric<volume_dim, Frame, DataType>,
      tmpl::size_t<volume_dim>, Frame>;

template <typename DataType, typename Frame = ::Frame::Inertial>
  using tags = tmpl::list<
      gr::Tags::Lapse<DataType>, ::Tags::dt<gr::Tags::Lapse<DataType>>,
      DerivLapse<DataType, Frame>, gr::Tags::Shift<volume_dim, Frame, DataType>,
      ::Tags::dt<gr::Tags::Shift<volume_dim, Frame, DataType>>,
      DerivShift<DataType, Frame>,
      gr::Tags::SpatialMetric<volume_dim, Frame, DataType>,
      ::Tags::dt<gr::Tags::SpatialMetric<volume_dim, Frame, DataType>>,
      DerivSpatialMetric<DataType, Frame>,
      gr::Tags::SqrtDetSpatialMetric<DataType>,
      gr::Tags::ExtrinsicCurvature<volume_dim, Frame, DataType>,
      gr::Tags::InverseSpatialMetric<volume_dim, Frame, DataType>>;
};

/*!
 * \ingroup AnalyticSolutionsGroup
 * \brief Classes which implement analytic solutions to Einstein's equations
 */
namespace Solutions {}
}  // namespace gr
