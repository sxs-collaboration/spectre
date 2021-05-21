// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Solutions.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"

namespace RelativisticEuler {
/// Base struct for properties common to all Relativistic Euler analytic
/// solutions
template <size_t Dim>
struct AnalyticSolution {
  static constexpr size_t volume_dim = Dim;

  template <typename DataType>
  using tags = tmpl::push_back<
      typename gr::AnalyticSolution<Dim>::template tags<DataType>,
      hydro::Tags::RestMassDensity<DataType>,
      hydro::Tags::SpecificInternalEnergy<DataType>,
      hydro::Tags::Pressure<DataType>,
      hydro::Tags::SpatialVelocity<DataType, Dim>,
      hydro::Tags::MagneticField<DataType, Dim>,
      hydro::Tags::DivergenceCleaningField<DataType>,
      hydro::Tags::LorentzFactor<DataType>,
      hydro::Tags::SpecificEnthalpy<DataType>>;
};

/*!
 * \ingroup AnalyticSolutionsGroup
 * \brief Holds classes implementing a solution to the relativistic Euler
 * system.
 */
namespace Solutions {}
}  // namespace RelativisticEuler
