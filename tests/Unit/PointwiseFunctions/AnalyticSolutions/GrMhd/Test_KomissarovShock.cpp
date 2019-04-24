// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/Options.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/KomissarovShock.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/PointwiseFunctions/AnalyticSolutions/GrMhd/VerifyGrMhdSolution.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_forward_declare Tensor

// IWYU pragma: no_include <vector>

namespace {

struct KomissarovShockProxy : grmhd::Solutions::KomissarovShock {
  using grmhd::Solutions::KomissarovShock::KomissarovShock;

  template <typename DataType>
  using hydro_variables_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataType>,
                 hydro::Tags::SpatialVelocity<DataType, 3>,
                 hydro::Tags::SpecificInternalEnergy<DataType>,
                 hydro::Tags::Pressure<DataType>,
                 hydro::Tags::LorentzFactor<DataType>,
                 hydro::Tags::SpecificEnthalpy<DataType>>;

  template <typename DataType>
  using grmhd_variables_tags =
      tmpl::push_back<hydro_variables_tags<DataType>,
                      hydro::Tags::MagneticField<DataType, 3>,
                      hydro::Tags::DivergenceCleaningField<DataType>>;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<hydro_variables_tags<DataType>>
  hydro_variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                  const double t) const noexcept {
    return variables(x, t, hydro_variables_tags<DataType>{});
  }

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<grmhd_variables_tags<DataType>>
  grmhd_variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                  const double t) const noexcept {
    return variables(x, t, grmhd_variables_tags<DataType>{});
  }
};

void test_create_from_options() noexcept {
  const auto komissarov_shock =
      test_creation<grmhd::Solutions::KomissarovShock>(
          "  AdiabaticIndex: 1.33\n"
          "  LeftDensity: 1.\n"
          "  RightDensity: 3.323\n"
          "  LeftPressure: 10.\n"
          "  RightPressure: 55.36\n"
          "  LeftVelocity: [0.83, 0., 0.]\n"
          "  RightVelocity: [0.62, -0.44, 0.]\n"
          "  LeftMagneticField: [10., 18.28, 0.]\n"
          "  RightMagneticField: [10., 14.49, 0.]\n"
          "  ShockSpeed: 0.5\n");
  CHECK(komissarov_shock == grmhd::Solutions::KomissarovShock(
                                1.33, 1., 3.323, 10., 55.36,
                                std::array<double, 3>{{0.83, 0., 0.}},
                                std::array<double, 3>{{0.62, -0.44, 0.}},
                                std::array<double, 3>{{10., 18.28, 0.}},
                                std::array<double, 3>{{10., 14.49, 0.}}, 0.5));
}

void test_move() noexcept {
  grmhd::Solutions::KomissarovShock komissarov_shock(
      4. / 3., 1., 3.323, 10., 55.36,
      std::array<double, 3>{{0.8370659816473115, 0., 0.}},
      std::array<double, 3>{{0.6202085442748952, -0.44207111995019704, 0.}},
      std::array<double, 3>{{10., 18.28, 0.}},
      std::array<double, 3>{{10., 14.49, 0.}}, 0.5);
  grmhd::Solutions::KomissarovShock komissarov_shock_copy(
      4. / 3., 1., 3.323, 10., 55.36,
      std::array<double, 3>{{0.8370659816473115, 0., 0.}},
      std::array<double, 3>{{0.6202085442748952, -0.44207111995019704, 0.}},
      std::array<double, 3>{{10., 18.28, 0.}},
      std::array<double, 3>{{10., 14.49, 0.}}, 0.5);
  test_move_semantics(std::move(komissarov_shock),
                      komissarov_shock_copy);  //  NOLINT
}

void test_serialize() noexcept {
  grmhd::Solutions::KomissarovShock komissarov_shock(
      4. / 3., 1., 3.323, 10., 55.36,
      std::array<double, 3>{{0.8370659816473115, 0., 0.}},
      std::array<double, 3>{{0.6202085442748952, -0.44207111995019704, 0.}},
      std::array<double, 3>{{10., 18.28, 0.}},
      std::array<double, 3>{{10., 14.49, 0.}}, 0.5);
  test_serialization(komissarov_shock);
}

void test_left_and_right_variables() noexcept {
  grmhd::Solutions::KomissarovShock komissarov_shock(
      4. / 3., 1., 3.323, 10., 55.36,
      std::array<double, 3>{{0.8370659816473115, 0., 0.}},
      std::array<double, 3>{{0.6202085442748952, -0.44207111995019704, 0.}},
      std::array<double, 3>{{10., 18.28, 0.}},
      std::array<double, 3>{{10., 14.49, 0.}}, 0.5);

  // Test that the fluid variables are set somewhere to the right and to the
  // left of the shock interface
  tnsr::I<double, 3, Frame::Inertial> left_x{0.};
  get<0>(left_x) = -1.;
  tnsr::I<double, 3, Frame::Inertial> right_x{0.};
  get<0>(right_x) = 1.;
  CHECK(
      get(get<hydro::Tags::RestMassDensity<double>>(komissarov_shock.variables(
          left_x, 0., tmpl::list<hydro::Tags::RestMassDensity<double>>{}))) ==
      1.);
  CHECK(
      get(get<hydro::Tags::RestMassDensity<double>>(komissarov_shock.variables(
          right_x, 0., tmpl::list<hydro::Tags::RestMassDensity<double>>{}))) ==
      3.323);
  CHECK(get(get<hydro::Tags::Pressure<double>>(komissarov_shock.variables(
            left_x, 0., tmpl::list<hydro::Tags::Pressure<double>>{}))) == 10.);
  CHECK(get(get<hydro::Tags::Pressure<double>>(komissarov_shock.variables(
            right_x, 0., tmpl::list<hydro::Tags::Pressure<double>>{}))) ==
        55.36);
  CHECK(get<0>(get<hydro::Tags::SpatialVelocity<double, 3, Frame::Inertial>>(
            komissarov_shock.variables(left_x, 0.,
                                       tmpl::list<hydro::Tags::SpatialVelocity<
                                           double, 3, Frame::Inertial>>{}))) ==
        0.8370659816473115);
  CHECK(get<0>(get<hydro::Tags::SpatialVelocity<double, 3, Frame::Inertial>>(
            komissarov_shock.variables(right_x, 0.,
                                       tmpl::list<hydro::Tags::SpatialVelocity<
                                           double, 3, Frame::Inertial>>{}))) ==
        0.6202085442748952);
  CHECK(get(get<hydro::Tags::LorentzFactor<double>>(komissarov_shock.variables(
            left_x, 0., tmpl::list<hydro::Tags::LorentzFactor<double>>{}))) ==
        approx(1.827812900709479));
  CHECK(get(get<hydro::Tags::LorentzFactor<double>>(komissarov_shock.variables(
            right_x, 0., tmpl::list<hydro::Tags::LorentzFactor<double>>{}))) ==
        approx(1.5431906071513006));
}

template <typename DataType>
void test_variables(const DataType& used_for_size) noexcept {
  KomissarovShockProxy komissarov_shock(
      4. / 3., 1., 3.323, 10., 55.36,
      std::array<double, 3>{{0.8370659816473115, 0., 0.}},
      std::array<double, 3>{{0.6202085442748952, -0.44207111995019704, 0.}},
      std::array<double, 3>{{10., 18.28, 0.}},
      std::array<double, 3>{{10., 14.49, 0.}}, 0.5);
  const auto member_variables = std::make_tuple(
      4. / 3., 1., 3.323, 10., 55.36,
      std::array<double, 3>{{0.8370659816473115, 0., 0.}},
      std::array<double, 3>{{0.6202085442748952, -0.44207111995019704, 0.}},
      std::array<double, 3>{{10., 18.28, 0.}},
      std::array<double, 3>{{10., 14.49, 0.}}, 0.5);

  pypp::check_with_random_values<
      1, KomissarovShockProxy::hydro_variables_tags<DataType>>(
      &KomissarovShockProxy::hydro_variables<DataType>, komissarov_shock,
      "KomissarovShock",
      {"rest_mass_density", "spatial_velocity", "specific_internal_energy",
       "pressure", "lorentz_factor", "specific_enthalpy"},
      {{{-1., 1.}}}, member_variables, used_for_size);

  pypp::check_with_random_values<
      1, KomissarovShockProxy::grmhd_variables_tags<DataType>>(
      &KomissarovShockProxy::grmhd_variables<DataType>, komissarov_shock,
      "KomissarovShock",
      {"rest_mass_density", "spatial_velocity", "specific_internal_energy",
       "pressure", "lorentz_factor", "specific_enthalpy", "magnetic_field",
       "divergence_cleaning_field"},
      {{{-1., 1.}}}, member_variables, used_for_size);
}

void test_solution() noexcept {
  grmhd::Solutions::KomissarovShock solution(
      1.33, 1., 3.323, 10., 55.36, std::array<double, 3>{{0.83, 0., 0.}},
      std::array<double, 3>{{0.62, -0.44, 0.}},
      std::array<double, 3>{{10., 18.28, 0.}},
      std::array<double, 3>{{10., 14.49, 0.}}, 0.5);
  const std::array<double, 3> x{{1.0, 2.3, -0.4}};
  const std::array<double, 3> dx{{1.e-1, 1.e-1, 1.e-1}};

  domain::creators::Brick<Frame::Inertial> brick(
      x - dx, x + dx, {{false, false, false}}, {{0, 0, 0}}, {{6, 6, 6}});
  Mesh<3> mesh{brick.initial_extents()[0], Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto};
  const auto domain = brick.create_domain();
  verify_grmhd_solution(solution, domain.blocks()[0], mesh, 1.e-10, 1.234,
                        1.e-1);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Solutions.GrMhd.KomissarovShock",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/GrMhd"};

  test_create_from_options();
  test_serialize();
  test_move();

  test_left_and_right_variables();
  test_variables(std::numeric_limits<double>::signaling_NaN());
  test_variables(DataVector(5));

  test_solution();
}
