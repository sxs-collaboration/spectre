// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Solutions.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"

namespace grmhd {
/// Base struct for properties common to all GRMHD analytic solutions
struct AnalyticSolution {
  static constexpr size_t volume_dim = 3_st;

  template <typename DataType>
  using tags = tmpl::push_back<
      typename gr::AnalyticSolution<3>::template tags<DataType>,
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
 * \ingroup AnalyticSolutionsGroup
 * \brief Holds classes implementing a solution to the GrMhd system.
 */
namespace Solutions {}
}  // namespace grmhd
