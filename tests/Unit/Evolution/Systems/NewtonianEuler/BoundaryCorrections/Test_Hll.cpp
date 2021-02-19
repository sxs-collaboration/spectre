// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "Evolution/Systems/NewtonianEuler/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/NewtonianEuler/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
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

  static inline unpacked_container unpack(
      const packed_container& /*packed*/,
      const size_t /*grid_point_index*/) noexcept {
    return true;
  }

  [[noreturn]] static inline void pack(
      const gsl::not_null<packed_container*> /*packed*/,
      const unpacked_container& /*unpacked*/,
      const size_t /*grid_point_index*/) {
    ERROR("Should not be converting an EOS from an unpacked to a packed type");
  }

  static inline size_t get_size(const packed_container& /*packed*/) noexcept {
    return 1;
  }
};

struct ConvertIdeal {
  using unpacked_container = bool;
  using packed_container = EquationsOfState::IdealFluid<false>;
  using packed_type = bool;

  static inline unpacked_container unpack(
      const packed_container& /*packed*/,
      const size_t /*grid_point_index*/) noexcept {
    return false;
  }

  [[noreturn]] static inline void pack(
      const gsl::not_null<packed_container*> /*packed*/,
      const unpacked_container& /*unpacked*/,
      const size_t /*grid_point_index*/) {
    ERROR("Should not be converting an EOS from an unpacked to a packed type");
  }

  static inline size_t get_size(const packed_container& /*packed*/) noexcept {
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

template <size_t Dim, typename EosTag>
void test(const size_t num_pts,
          const tuples::TaggedTuple<EosTag>& volume_data) {
  TestHelpers::evolution::dg::test_boundary_correction_conservation<
      NewtonianEuler::System<Dim, typename EosTag::type, DummyInitialData>>(
      NewtonianEuler::BoundaryCorrections::Hll<Dim>{},
      Mesh<Dim - 1>{num_pts, Spectral::Basis::Legendre,
                    Spectral::Quadrature::Gauss},
      volume_data);

  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      NewtonianEuler::System<Dim, typename EosTag::type, DummyInitialData>,
      tmpl::list<ConvertPolytropic, ConvertIdeal>>(
      "Hll",
      {{"dg_package_data_mass_density", "dg_package_data_momentum_density",
        "dg_package_data_energy_density",
        "dg_package_data_normal_dot_flux_mass_density",
        "dg_package_data_normal_dot_flux_momentum_density",
        "dg_package_data_normal_dot_flux_energy_density",
        "dg_package_data_largest_outgoing_char_speed",
        "dg_package_data_largest_ingoing_char_speed"}},
      {{"dg_boundary_terms_mass_density", "dg_boundary_terms_momentum_density",
        "dg_boundary_terms_energy_density"}},
      NewtonianEuler::BoundaryCorrections::Hll<Dim>{},
      Mesh<Dim - 1>{num_pts, Spectral::Basis::Legendre,
                    Spectral::Quadrature::Gauss},
      volume_data);

  const auto hll = TestHelpers::test_factory_creation<
      NewtonianEuler::BoundaryCorrections::BoundaryCorrection<Dim>>("Hll:");

  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      NewtonianEuler::System<Dim, typename EosTag::type, DummyInitialData>,
      tmpl::list<ConvertPolytropic, ConvertIdeal>>(
      "Hll",
      {{"dg_package_data_mass_density", "dg_package_data_momentum_density",
        "dg_package_data_energy_density",
        "dg_package_data_normal_dot_flux_mass_density",
        "dg_package_data_normal_dot_flux_momentum_density",
        "dg_package_data_normal_dot_flux_energy_density",
        "dg_package_data_largest_outgoing_char_speed",
        "dg_package_data_largest_ingoing_char_speed"}},
      {{"dg_boundary_terms_mass_density", "dg_boundary_terms_momentum_density",
        "dg_boundary_terms_energy_density"}},
      dynamic_cast<const NewtonianEuler::BoundaryCorrections::Hll<Dim>&>(
          *hll),
      Mesh<Dim - 1>{num_pts, Spectral::Basis::Legendre,
                    Spectral::Quadrature::Gauss},
      volume_data);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NewtonianEuler.Hll", "[Unit][Evolution]") {
  PUPable_reg(NewtonianEuler::BoundaryCorrections::Hll<1>);
  PUPable_reg(NewtonianEuler::BoundaryCorrections::Hll<2>);
  PUPable_reg(NewtonianEuler::BoundaryCorrections::Hll<3>);
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/NewtonianEuler/BoundaryCorrections"};
  test<1>(1, tuples::TaggedTuple<hydro::Tags::EquationOfState<
                 EquationsOfState::PolytropicFluid<false>>>{
                 EquationsOfState::PolytropicFluid<false>{1.0e-3, 2.0}});
  test<2>(5, tuples::TaggedTuple<hydro::Tags::EquationOfState<
                 EquationsOfState::PolytropicFluid<false>>>{
                 EquationsOfState::PolytropicFluid<false>{1.0e-3, 2.0}});
  test<3>(5, tuples::TaggedTuple<hydro::Tags::EquationOfState<
                 EquationsOfState::PolytropicFluid<false>>>{
                 EquationsOfState::PolytropicFluid<false>{1.0e-3, 2.0}});

  test<1>(
      1, tuples::TaggedTuple<
             hydro::Tags::EquationOfState<EquationsOfState::IdealFluid<false>>>{
             EquationsOfState::IdealFluid<false>{1.3}});
  test<2>(
      5, tuples::TaggedTuple<
             hydro::Tags::EquationOfState<EquationsOfState::IdealFluid<false>>>{
             EquationsOfState::IdealFluid<false>{1.3}});
  test<3>(
      5, tuples::TaggedTuple<
             hydro::Tags::EquationOfState<EquationsOfState::IdealFluid<false>>>{
             EquationsOfState::IdealFluid<false>{1.3}});
}
