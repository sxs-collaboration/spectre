// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/BondiMichel.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/PointwiseFunctions/AnalyticSolutions/GrMhd/VerifyGrMhdSolution.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_forward_declare Tags::dt

// IWYU pragma: no_include <vector>

namespace {

struct BondiMichelProxy : grmhd::Solutions::BondiMichel {
  using grmhd::Solutions::BondiMichel::BondiMichel;

  template <typename DataType>
  using hydro_variables_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataType>,
                 hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>,
                 hydro::Tags::SpecificInternalEnergy<DataType>,
                 hydro::Tags::Pressure<DataType>,
                 hydro::Tags::LorentzFactor<DataType>,
                 hydro::Tags::SpecificEnthalpy<DataType>>;

  template <typename DataType>
  using grmhd_variables_tags =
      tmpl::push_back<hydro_variables_tags<DataType>,
                      hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>,
                      hydro::Tags::DivergenceCleaningField<DataType>>;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<hydro_variables_tags<DataType>>
  hydro_variables(const tnsr::I<DataType, 3>& x) const noexcept {
    return variables(x, 0.0, hydro_variables_tags<DataType>{});
  }

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<grmhd_variables_tags<DataType>>
  grmhd_variables(const tnsr::I<DataType, 3>& x) const noexcept {
    return variables(x, 0.0, grmhd_variables_tags<DataType>{});
  }
};

void test_create_from_options() noexcept {
  const auto flow = test_creation<grmhd::Solutions::BondiMichel>(
      "  Mass: 1.2\n"
      "  SonicRadius: 5.0\n"
      "  SonicDensity: 0.05\n"
      "  PolytropicExponent: 1.4\n"
      "  MagFieldStrength: 2.0");
  CHECK(flow == grmhd::Solutions::BondiMichel(1.2, 5.0, 0.05, 1.4, 2.0));
}

void test_move() noexcept {
  grmhd::Solutions::BondiMichel flow(2.0, 3000.0, 1.3, 1.5, 0.24);
  grmhd::Solutions::BondiMichel flow_copy(2.0, 3000.0, 1.3, 1.5, 0.24);
  test_move_semantics(std::move(flow), flow_copy);  //  NOLINT
}

void test_serialize() noexcept {
  grmhd::Solutions::BondiMichel flow(1.0, 3500.0, 1.3, 1.5, 0.24);
  test_serialization(flow);
}

template <typename DataType>
void test_variables(const DataType& used_for_size) {
  const double mass = 1.6;
  const double sonic_radius = 4.0;
  const double sonic_density = 0.4;
  const double polytropic_exponent = 4. / 3.;
  const double mag_field_strength = 2.3;

  pypp::check_with_random_values<
      1, BondiMichelProxy::hydro_variables_tags<DataType>>(
      &BondiMichelProxy::hydro_variables<DataType>,
      BondiMichelProxy(mass, sonic_radius, sonic_density, polytropic_exponent,
                       mag_field_strength),
      "TestFunctions",
      {"bondi_michel_rest_mass_density", "bondi_michel_spatial_velocity",
       "bondi_michel_specific_internal_energy", "bondi_michel_pressure",
       "bondi_michel_lorentz_factor", "bondi_michel_specific_enthalpy"},
      {{{1.0, 20.0}}},
      std::make_tuple(mass, sonic_radius, sonic_density, polytropic_exponent,
                      mag_field_strength),
      used_for_size);

  pypp::check_with_random_values<
      1, BondiMichelProxy::grmhd_variables_tags<DataType>>(
      &BondiMichelProxy::grmhd_variables<DataType>,
      BondiMichelProxy(mass, sonic_radius, sonic_density, polytropic_exponent,
                       mag_field_strength),
      "TestFunctions",
      {"bondi_michel_rest_mass_density", "bondi_michel_spatial_velocity",
       "bondi_michel_specific_internal_energy", "bondi_michel_pressure",
       "bondi_michel_lorentz_factor", "bondi_michel_specific_enthalpy",
       "bondi_michel_magnetic_field", "bondi_michel_divergence_cleaning_field"},
      {{{1.0, 20.0}}},
      std::make_tuple(mass, sonic_radius, sonic_density, polytropic_exponent,
                      mag_field_strength),
      used_for_size);
}

void test_solution() noexcept {
  grmhd::Solutions::BondiMichel solution(1.0, 50.0, 1.3, 1.5, 0.24);
  const std::array<double, 3> x{{4.0, 4.0, 4.0}};
  const std::array<double, 3> dx{{1.e-3, 1.e-3, 1.e-3}};

  domain::creators::Brick<Frame::Inertial> brick(
      x - dx, x + dx, {{false, false, false}}, {{0, 0, 0}}, {{4, 4, 4}});
  Mesh<3> mesh{brick.initial_extents()[0], Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto};
  const auto domain = brick.create_domain();
  verify_grmhd_solution(solution, domain.blocks()[0], mesh, 1.e-10, 1.234,
                        1.e-4);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.GrMhd.BondiMichel",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/GrMhd"};

  test_create_from_options();
  test_serialize();
  test_move();

  test_variables(std::numeric_limits<double>::signaling_NaN());
  test_variables(DataVector(5));

  test_solution();
}
