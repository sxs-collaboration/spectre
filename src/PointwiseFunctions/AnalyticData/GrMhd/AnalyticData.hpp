// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Solutions.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"

namespace grmhd {
/// Base struct for properties common to all GRMHD analytic data classes
struct AnalyticDataBase {
  static constexpr size_t volume_dim = 3_st;

  template <typename DataType>
  using tags =
      tmpl::push_back<typename gr::AnalyticSolution<3>::template tags<DataType>,
                      hydro::Tags::RestMassDensity<DataType>,
                      hydro::Tags::SpecificInternalEnergy<DataType>,
                      hydro::Tags::Pressure<DataType>,
                      hydro::Tags::SpatialVelocity<DataType, 3>,
                      hydro::Tags::MagneticField<DataType, 3>,
                      hydro::Tags::DivergenceCleaningField<DataType>,
                      hydro::Tags::LorentzFactor<DataType>,
                      hydro::Tags::SpecificEnthalpy<DataType>>;
};

/*!
 * \ingroup AnalyticDataGroup
 * \brief Holds classes implementing analytic data for the GrMhd system.
 */
namespace AnalyticData {}
}  // namespace grmhd
