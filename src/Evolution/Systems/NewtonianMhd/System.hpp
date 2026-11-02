// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/NewtonianMhd/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianMhd/Characteristics.hpp"
#include "Evolution/Systems/NewtonianMhd/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianMhd/PrimitiveFromConservative.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "Evolution/Systems/NewtonianMhd/TimeDerivativeTerms.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup EvolutionSystemsGroup
/// \brief Items related to evolving the Newtonian Euler system
namespace NewtonianMhd {

namespace detail {
/// Asserts on use rather than on instantiation of `System`.
template <bool EvolveMagneticField>
struct MagneticFieldNotSupportedYet {
  static_assert(not EvolveMagneticField,
                "Evolving the magnetic field is not supported yet. "
                "Instantiate the system with EvolveMagneticField = false.");
};
}  // namespace detail

template <size_t Dim, bool EvolveMagneticField = true>
struct System {
  static_assert(
      Dim == 3 or not EvolveMagneticField,
      "Magnetic field evolution is only supported at Dim == 3 since the vector "
      "dimensionality is tied to that of the system. "
      "Currently, lower-dimensional MHD problems still need to be run in a 3D "
      "domain. Users may set ignored directions to be thin and periodic.");

  static constexpr bool evolve_magnetic_field = EvolveMagneticField;

  static std::string name() {
    return EvolveMagneticField ? "NewtonianMhd" : "NewtonianEuler";
  }

  static constexpr bool is_in_flux_conservative_form = true;
  static constexpr bool has_primitive_and_conservative_vars = true;
  static constexpr size_t volume_dim = Dim;

  using boundary_conditions_base = BoundaryConditions::BoundaryCondition<Dim>;

  using flux_variables = tmpl::append<
      tmpl::list<Tags::MassDensityCons, Tags::MomentumDensity<Dim>,
                 Tags::EnergyDensity>,
      tmpl::conditional_t<EvolveMagneticField,
                          tmpl::list<Tags::MagneticFieldCons<>,
                                     Tags::DivergenceCleaningFieldCons>,
                          tmpl::list<>>>;
  using variables_tag = ::Tags::Variables<flux_variables>;
  using non_conservative_variables = tmpl::list<>;
  using gradient_variables = tmpl::list<>;
  // Compute item for pressure not currently implemented in SpECTRE,
  // so its simple tag is passed along with the primitive variables.
  using primitive_variables_tag = ::Tags::Variables<tmpl::append<
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, Dim>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::Pressure<DataVector>>,
      tmpl::conditional_t<
          EvolveMagneticField,
          tmpl::list<hydro::Tags::MagneticField<DataVector, 3>,
                     hydro::Tags::DivergenceCleaningField<DataVector>>,
          tmpl::list<>>>>;

  using compute_volume_time_derivative_terms = tmpl::conditional_t<
      EvolveMagneticField,
      detail::MagneticFieldNotSupportedYet<EvolveMagneticField>,
      TimeDerivativeTerms<Dim>>;

  using conservative_from_primitive = ConservativeFromPrimitive<Dim>;
  using primitive_from_conservative = PrimitiveFromConservative<Dim>;

  using compute_largest_characteristic_speed =
      Tags::ComputeLargestCharacteristicSpeed<Dim>;
};

}  // namespace NewtonianMhd
