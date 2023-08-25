// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "Evolution/Systems/NewtonianEuler/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryCorrections/Hllc.hpp"
#include "Evolution/Systems/NewtonianEuler/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Range.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct ConvertPolytropic {
  using unpacked_container = bool;
  using packed_container = EquationsOfState::PolytropicFluid<false>;
  using packed_type = bool;

  static inline unpacked_container unpack(const packed_container& /*packed*/,
                                          const size_t /*grid_point_index*/) {
    return true;
  }

  [[noreturn]] static inline void pack(
      const gsl::not_null<packed_container*> /*packed*/,
      const unpacked_container& /*unpacked*/,
      const size_t /*grid_point_index*/) {
    ERROR("Should not be converting an EOS from an unpacked to a packed type");
  }

  static inline size_t get_size(const packed_container& /*packed*/) {
    return 1;
  }
};

struct ConvertIdeal {
  using unpacked_container = bool;
  using packed_container = EquationsOfState::IdealFluid<false>;
  using packed_type = bool;

  static inline unpacked_container unpack(const packed_container& /*packed*/,
                                          const size_t /*grid_point_index*/) {
    return false;
  }

  [[noreturn]] static inline void pack(
      const gsl::not_null<packed_container*> /*packed*/,
      const unpacked_container& /*unpacked*/,
      const size_t /*grid_point_index*/) {
    ERROR("Should not be converting an EOS from an unpacked to a packed type");
  }

  static inline size_t get_size(const packed_container& /*packed*/) {
    return 1;
  }
};

struct DummyInitialData {
  using argument_tags = tmpl::list<>;
  struct source_term_type {
    using sourced_variables = tmpl::list<>;
    using argument_tags = tmpl::list<>;
  };
};

namespace helpers = TestHelpers::evolution::dg;

template <size_t Dim, typename EosType>
void test(const gsl::not_null<std::mt19937*> gen, const size_t num_pts,
          const EosType& equation_of_state) {
  const tuples::TaggedTuple<hydro::Tags::EquationOfState<EosType>> volume_data{
      equation_of_state};
  const tuples::TaggedTuple<
      helpers::Tags::Range<NewtonianEuler::Tags::MassDensityCons>,
      helpers::Tags::Range<
          NewtonianEuler::Tags::SpecificInternalEnergy<DataVector>>>
      ranges{std::array{1.0e-30, 1.0}, std::array{1.0e-30, 1.0}};

  helpers::test_boundary_correction_conservation<
      NewtonianEuler::System<Dim, DummyInitialData>>(
      gen, NewtonianEuler::BoundaryCorrections::Hllc<Dim>{},
      Mesh<Dim - 1>{num_pts, SpatialDiscretization::Basis::Legendre,
                    SpatialDiscretization::Quadrature::Gauss},
      volume_data, ranges);

  helpers::test_boundary_correction_with_python<
      NewtonianEuler::System<Dim, DummyInitialData>,
      tmpl::list<ConvertPolytropic, ConvertIdeal>>(
      gen, "Hllc",
      {{"dg_package_data_mass_density", "dg_package_data_momentum_density",
        "dg_package_data_energy_density", "dg_package_data_pressure",
        "dg_package_data_normal_dot_flux_mass_density",
        "dg_package_data_normal_dot_flux_momentum_density",
        "dg_package_data_normal_dot_flux_energy_density",
        "dg_package_data_interface_unit_normal",
        "dg_package_data_normal_dot_velocity",
        "dg_package_data_largest_outgoing_char_speed",
        "dg_package_data_largest_ingoing_char_speed"}},
      {{"dg_boundary_terms_mass_density", "dg_boundary_terms_momentum_density",
        "dg_boundary_terms_energy_density"}},
      NewtonianEuler::BoundaryCorrections::Hllc<Dim>{},
      Mesh<Dim - 1>{num_pts, SpatialDiscretization::Basis::Legendre,
                    SpatialDiscretization::Quadrature::Gauss},
      volume_data, ranges);

  const auto hllc = TestHelpers::test_creation<std::unique_ptr<
      NewtonianEuler::BoundaryCorrections::BoundaryCorrection<Dim>>>("Hllc:");

  helpers::test_boundary_correction_with_python<
      NewtonianEuler::System<Dim, DummyInitialData>,
      tmpl::list<ConvertPolytropic, ConvertIdeal>>(
      gen, "Hllc",
      {{"dg_package_data_mass_density", "dg_package_data_momentum_density",
        "dg_package_data_energy_density", "dg_package_data_pressure",
        "dg_package_data_normal_dot_flux_mass_density",
        "dg_package_data_normal_dot_flux_momentum_density",
        "dg_package_data_normal_dot_flux_energy_density",
        "dg_package_data_interface_unit_normal",
        "dg_package_data_normal_dot_velocity",
        "dg_package_data_largest_outgoing_char_speed",
        "dg_package_data_largest_ingoing_char_speed"}},
      {{"dg_boundary_terms_mass_density", "dg_boundary_terms_momentum_density",
        "dg_boundary_terms_energy_density"}},
      dynamic_cast<const NewtonianEuler::BoundaryCorrections::Hllc<Dim>&>(
          *hllc),
      Mesh<Dim - 1>{num_pts, SpatialDiscretization::Basis::Legendre,
                    SpatialDiscretization::Quadrature::Gauss},
      volume_data, ranges);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NewtonianEuler.BoundaryCorrections.Hllc",
                  "[Unit][Evolution]") {
  PUPable_reg(NewtonianEuler::BoundaryCorrections::Hllc<1>);
  PUPable_reg(NewtonianEuler::BoundaryCorrections::Hllc<2>);
  PUPable_reg(NewtonianEuler::BoundaryCorrections::Hllc<3>);

  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/NewtonianEuler/BoundaryCorrections"};
  MAKE_GENERATOR(gen);

  test<1>(make_not_null(&gen), 1,
          EquationsOfState::PolytropicFluid<false>{1.0e-3, 2.0});
  test<2>(make_not_null(&gen), 5,
          EquationsOfState::PolytropicFluid<false>{1.0e-3, 2.0});
  test<3>(make_not_null(&gen), 5,
          EquationsOfState::PolytropicFluid<false>{1.0e-3, 2.0});

  test<1>(make_not_null(&gen), 1, EquationsOfState::IdealFluid<false>{1.3});
  test<2>(make_not_null(&gen), 5, EquationsOfState::IdealFluid<false>{1.3});
  test<3>(make_not_null(&gen), 5, EquationsOfState::IdealFluid<false>{1.3});
}
