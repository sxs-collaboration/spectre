// Distributed under the MIT License.
// See LICENSE.txt for details.

/*!
 * \file
 * \brief Defines lists of `gr::Tags`.
 *
 * With this separate file we avoid having to include `Tags.hpp` where it's
 * unnecessary and we avoid including prefix tags in `TagsDeclarations.hpp`.
 */

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"

namespace gr {
namespace Tags {

/// All tags for spacetime quantities in the 3+1 decomposition
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
using all_spacetime_three_plus_one = tmpl::list<
    gr::Tags::Lapse<DataType>, ::Tags::dt<gr::Tags::Lapse<DataType>>,
    ::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<Dim>, Frame>,
    gr::Tags::Shift<Dim, Frame, DataType>,
    ::Tags::dt<gr::Tags::Shift<Dim, Frame, DataType>>,
    ::Tags::deriv<gr::Tags::Shift<Dim, Frame, DataType>, tmpl::size_t<Dim>,
                  Frame>,
    gr::Tags::SpatialMetric<Dim, Frame, DataType>,
    ::Tags::dt<gr::Tags::SpatialMetric<Dim, Frame, DataType>>,
    ::Tags::deriv<gr::Tags::SpatialMetric<Dim, Frame, DataType>,
                  tmpl::size_t<Dim>, Frame>,
    gr::Tags::SqrtDetSpatialMetric<DataType>,
    gr::Tags::ExtrinsicCurvature<Dim, Frame, DataType>,
    gr::Tags::InverseSpatialMetric<Dim, Frame, DataType>>;

/// All tags for source quantities in the 3+1 decomposition
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
using all_source_three_plus_one = tmpl::list<EnergyDensity<DataType>>;

}  // namespace Tags
}  // namespace gr
