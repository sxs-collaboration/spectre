// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "Evolution/Systems/RelativisticEuler/Valencia/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct ConvertPolytropic {
  using unpacked_container = bool;
  using packed_container = EquationsOfState::PolytropicFluid<true>;
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
  using packed_container = EquationsOfState::IdealFluid<true>;
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

namespace helpers = TestHelpers::evolution::dg;

template <size_t Dim, typename EosType>
void test(const gsl::not_null<std::mt19937*> gen, const size_t num_pts,
          const EosType& equation_of_state) {
  tuples::TaggedTuple<hydro::Tags::EquationOfState<EosType>> volume_data{
      equation_of_state};
  tuples::TaggedTuple<
      helpers::Tags::Range<hydro::Tags::RestMassDensity<DataVector>>,
      helpers::Tags::Range<hydro::Tags::SpecificInternalEnergy<DataVector>>,
      helpers::Tags::Range<hydro::Tags::SpecificEnthalpy<DataVector>>,
      helpers::Tags::Range<hydro::Tags::SpatialVelocity<DataVector, Dim>>>
      ranges{std::array<double, 2>{{1.0e-30, 1.0}},
             std::array<double, 2>{{1.0e-30, 1.0}},
             std::array<double, 2>{{1.0, 2.0}},  // relativistic h = 1 + stuff
             std::array<double, 2>{{-0.1, 0.1}}};

  helpers::test_boundary_correction_conservation<
      RelativisticEuler::Valencia::System<Dim, EosType>>(
      gen, RelativisticEuler::Valencia::BoundaryCorrections::Rusanov<Dim>{},
      Mesh<Dim - 1>{num_pts, Spectral::Basis::Legendre,
                    Spectral::Quadrature::Gauss},
      volume_data, ranges);

  helpers::test_boundary_correction_with_python<
      RelativisticEuler::Valencia::System<Dim, EosType>,
      tmpl::list<ConvertPolytropic, ConvertIdeal>>(
      gen,
      "Evolution.Systems.RelativisticEuler.Valencia.BoundaryCorrections."
      "Rusanov",
      {{"dg_package_data_tilde_d", "dg_package_data_tilde_tau",
        "dg_package_data_tilde_s", "dg_package_data_normal_dot_flux_tilde_d",
        "dg_package_data_normal_dot_flux_tilde_tau",
        "dg_package_data_normal_dot_flux_tilde_s",
        "dg_package_data_abs_char_speed"}},
      {{"dg_boundary_terms_tilde_d", "dg_boundary_terms_tilde_tau",
        "dg_boundary_terms_tilde_s"}},
      RelativisticEuler::Valencia::BoundaryCorrections::Rusanov<Dim>{},
      Mesh<Dim - 1>{num_pts, Spectral::Basis::Legendre,
                    Spectral::Quadrature::Gauss},
      volume_data, ranges);

  const auto rusanov = TestHelpers::test_factory_creation<
      RelativisticEuler::Valencia::BoundaryCorrections::BoundaryCorrection<
          Dim>>("Rusanov:");

  helpers::test_boundary_correction_with_python<
      RelativisticEuler::Valencia::System<Dim, EosType>,
      tmpl::list<ConvertPolytropic, ConvertIdeal>>(
      gen,
      "Evolution.Systems.RelativisticEuler.Valencia.BoundaryCorrections."
      "Rusanov",
      {{"dg_package_data_tilde_d", "dg_package_data_tilde_tau",
        "dg_package_data_tilde_s", "dg_package_data_normal_dot_flux_tilde_d",
        "dg_package_data_normal_dot_flux_tilde_tau",
        "dg_package_data_normal_dot_flux_tilde_s",
        "dg_package_data_abs_char_speed"}},
      {{"dg_boundary_terms_tilde_d", "dg_boundary_terms_tilde_tau",
        "dg_boundary_terms_tilde_s"}},
      dynamic_cast<const RelativisticEuler::Valencia::BoundaryCorrections::
                       Rusanov<Dim>&>(*rusanov),
      Mesh<Dim - 1>{num_pts, Spectral::Basis::Legendre,
                    Spectral::Quadrature::Gauss},
      volume_data, ranges);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.RelativisticEuler.Valencia.Rusanov",
                  "[Unit][Evolution]") {
  PUPable_reg(RelativisticEuler::Valencia::BoundaryCorrections::Rusanov<1>);
  PUPable_reg(RelativisticEuler::Valencia::BoundaryCorrections::Rusanov<2>);
  PUPable_reg(RelativisticEuler::Valencia::BoundaryCorrections::Rusanov<3>);

  pypp::SetupLocalPythonEnvironment local_python_env{""};
  MAKE_GENERATOR(gen);

  test<1>(make_not_null(&gen), 1,
          EquationsOfState::PolytropicFluid<true>{1.0e-3, 2.0});
  test<2>(make_not_null(&gen), 5,
          EquationsOfState::PolytropicFluid<true>{1.0e-3, 2.0});
  test<3>(make_not_null(&gen), 5,
          EquationsOfState::PolytropicFluid<true>{1.0e-3, 2.0});

  test<1>(make_not_null(&gen), 1, EquationsOfState::IdealFluid<true>{1.3});
  test<2>(make_not_null(&gen), 5, EquationsOfState::IdealFluid<true>{1.3});
  test<3>(make_not_null(&gen), 5, EquationsOfState::IdealFluid<true>{1.3});
}
