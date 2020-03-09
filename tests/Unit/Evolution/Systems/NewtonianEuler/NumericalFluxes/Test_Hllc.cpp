// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/NewtonianEuler/Characteristics.hpp"
#include "Evolution/Systems/NewtonianEuler/Fluxes.hpp"
#include "Evolution/Systems/NewtonianEuler/NumericalFluxes/Hllc.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/TestHelpers.hpp"
#include "Helpers/Utilities/ProtocolTestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Protocols.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {

// This test will check the correct implementation of the numerical flux,
// as well as the interface expected to be used by the flux communication code.
// Note: for these checks, the interface normal does not require be normalized.
template <size_t Dim, typename Frame>
void apply_hllc(
    const gsl::not_null<Scalar<DataVector>*> n_dot_num_f_mass_density,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame>*>
        n_dot_num_f_momentum_density,
    const gsl::not_null<Scalar<DataVector>*> n_dot_num_f_energy_density,
    const Scalar<DataVector>& mass_density_int,
    const tnsr::I<DataVector, Dim, Frame>& momentum_density_int,
    const Scalar<DataVector>& energy_density_int,
    const Scalar<DataVector>& pressure_int,
    const Scalar<DataVector>& mass_density_ext,
    const tnsr::I<DataVector, Dim, Frame>& momentum_density_ext,
    const Scalar<DataVector>& energy_density_ext,
    const Scalar<DataVector>& pressure_ext,
    const tnsr::i<DataVector, Dim, Frame>& interface_normal) noexcept {
  const size_t number_of_points = get(mass_density_int).size();

  // make consistent velocity from conservatives
  tnsr::I<DataVector, Dim, Frame> velocity_int(number_of_points);
  tnsr::I<DataVector, Dim, Frame> velocity_ext(number_of_points);
  for (size_t i = 0; i < Dim; ++i) {
    velocity_int.get(i) = momentum_density_int.get(i) / get(mass_density_int);
    velocity_ext.get(i) = momentum_density_ext.get(i) / get(mass_density_ext);
  }

  // make consistent volume fluxes from primitives and conservatives
  tnsr::I<DataVector, Dim, Frame> flux_mass_density_int(number_of_points);
  tnsr::IJ<DataVector, Dim, Frame> flux_momentum_density_int(number_of_points);
  tnsr::I<DataVector, Dim, Frame> flux_energy_density_int(number_of_points);
  NewtonianEuler::ComputeFluxes<Dim>::apply(
      make_not_null(&flux_mass_density_int),
      make_not_null(&flux_momentum_density_int),
      make_not_null(&flux_energy_density_int), momentum_density_int,
      energy_density_int, velocity_int, pressure_int);

  tnsr::I<DataVector, Dim, Frame> flux_mass_density_ext(number_of_points);
  tnsr::IJ<DataVector, Dim, Frame> flux_momentum_density_ext(number_of_points);
  tnsr::I<DataVector, Dim, Frame> flux_energy_density_ext(number_of_points);
  NewtonianEuler::ComputeFluxes<Dim>::apply(
      make_not_null(&flux_mass_density_ext),
      make_not_null(&flux_momentum_density_ext),
      make_not_null(&flux_energy_density_ext), momentum_density_ext,
      energy_density_ext, velocity_ext, pressure_ext);

  // Given the general interface required by the (generic) evolution code,
  // the computation of the numerical flux will work for a general case where
  // the exterior interface normal does not equal minus the interior one. Here,
  // we test n_ext = n_int as it is the only case expected to be used.
  tnsr::i<DataVector, Dim, Frame> minus_interface_normal = interface_normal;
  for (size_t i = 0; i < Dim; ++i) {
    minus_interface_normal.get(i) *= -1.0;
  }

  // In an actual run, a generic action will perform the computation of the
  // normal fluxes. For simplicity, here we compute them manually.
  const Scalar<DataVector> n_dot_f_mass_density_int =
      dot_product(flux_mass_density_int, interface_normal);
  const Scalar<DataVector> n_dot_f_energy_density_int =
      dot_product(flux_energy_density_int, interface_normal);
  tnsr::I<DataVector, Dim, Frame> n_dot_f_momentum_density_int(
      number_of_points);
  tnsr::I<DataVector, Dim, Frame> minus_n_dot_f_momentum_density_ext(
      number_of_points);
  for (size_t i = 0; i < Dim; ++i) {
    n_dot_f_momentum_density_int.get(i) = 0.0;
    minus_n_dot_f_momentum_density_ext.get(i) = 0.0;
    for (size_t j = 0; j < Dim; ++j) {
      n_dot_f_momentum_density_int.get(i) +=
          flux_momentum_density_int.get(i, j) * interface_normal.get(j);
      minus_n_dot_f_momentum_density_ext.get(i) +=
          flux_momentum_density_ext.get(i, j) * minus_interface_normal.get(j);
    }
  }

  // make consistent sound speed for representative EoS (ideal gas)
  const double adiabatic_index = 1.3333333333333333;
  const Scalar<DataVector> sound_speed_int{
      sqrt(adiabatic_index * get(pressure_int) / get(mass_density_int))};
  const Scalar<DataVector> sound_speed_ext{
      sqrt(adiabatic_index * get(pressure_ext) / get(mass_density_ext))};

  using hllc_flux = NewtonianEuler::NumericalFluxes::Hllc<Dim, Frame>;
  const hllc_flux flux_computer{};
  const auto& used_for_size = get(mass_density_int);

  // Package interior data
  auto packaged_data_int = TestHelpers::NumericalFluxes::get_packaged_data(
      flux_computer, used_for_size, n_dot_f_mass_density_int,
      n_dot_f_momentum_density_int, n_dot_f_energy_density_int,
      mass_density_int, momentum_density_int, energy_density_int, velocity_int,
      pressure_int,
      NewtonianEuler::characteristic_speeds(velocity_int, sound_speed_int,
                                            interface_normal),
      interface_normal);

  // First, let exterior data equal interior data in order to test consistency.
  // We need to use minus interface normal to package exterior data, though.
  tnsr::I<DataVector, Dim, Frame> minus_n_dot_f_momentum_density_int(
      number_of_points);
  for (size_t i = 0; i < Dim; ++i) {
    minus_n_dot_f_momentum_density_int.get(i) =
        -n_dot_f_momentum_density_int.get(i);
  }
  auto packaged_data_ext = TestHelpers::NumericalFluxes::get_packaged_data(
      flux_computer, used_for_size,
      dot_product(flux_mass_density_int, minus_interface_normal),
      minus_n_dot_f_momentum_density_int,
      dot_product(flux_energy_density_int, minus_interface_normal),
      mass_density_int, momentum_density_int, energy_density_int, velocity_int,
      pressure_int,
      NewtonianEuler::characteristic_speeds(velocity_int, sound_speed_int,
                                            minus_interface_normal),
      minus_interface_normal);

  // Test consistency: F_num(U, U) = F(U)
  dg::NumericalFluxes::normal_dot_numerical_fluxes(
      hllc_flux{}, packaged_data_int, packaged_data_ext,
      n_dot_num_f_mass_density, n_dot_num_f_momentum_density,
      n_dot_num_f_energy_density);
  CHECK(*n_dot_num_f_mass_density == n_dot_f_mass_density_int);
  CHECK(*n_dot_num_f_momentum_density == n_dot_f_momentum_density_int);
  CHECK(*n_dot_num_f_energy_density == n_dot_f_energy_density_int);

  // Now package different exterior data.
  packaged_data_ext = TestHelpers::NumericalFluxes::get_packaged_data(
      flux_computer, used_for_size,
      dot_product(flux_mass_density_ext, minus_interface_normal),
      minus_n_dot_f_momentum_density_ext,
      dot_product(flux_energy_density_ext, minus_interface_normal),
      mass_density_ext, momentum_density_ext, energy_density_ext, velocity_ext,
      pressure_ext,
      NewtonianEuler::characteristic_speeds(velocity_ext, sound_speed_ext,
                                            minus_interface_normal),
      minus_interface_normal);

  // These numerical fluxes will be compared with those obtained with pypp.
  dg::NumericalFluxes::normal_dot_numerical_fluxes(
      hllc_flux{}, packaged_data_int, packaged_data_ext,
      n_dot_num_f_mass_density, n_dot_num_f_momentum_density,
      n_dot_num_f_energy_density);

  // Before exiting, check that flux is conservative by swapping int/ext data
  Scalar<DataVector> minus_n_dot_num_f_mass_density(number_of_points);
  tnsr::I<DataVector, Dim, Frame> minus_n_dot_num_f_momentum_density(
      number_of_points);
  Scalar<DataVector> minus_n_dot_num_f_energy_density(number_of_points);
  dg::NumericalFluxes::normal_dot_numerical_fluxes(
      hllc_flux{}, packaged_data_ext, packaged_data_int,
      make_not_null(&minus_n_dot_num_f_mass_density),
      make_not_null(&minus_n_dot_num_f_momentum_density),
      make_not_null(&minus_n_dot_num_f_energy_density));
  CHECK(get(*n_dot_num_f_mass_density) == -get(minus_n_dot_num_f_mass_density));
  CHECK(get(*n_dot_num_f_energy_density) ==
        -get(minus_n_dot_num_f_energy_density));
  for (size_t i = 0; i < Dim; ++i) {
    CHECK(n_dot_num_f_momentum_density->get(i) ==
          -minus_n_dot_num_f_momentum_density.get(i));
  }
}

template <size_t Dim, typename Frame>
void test_flux(const DataVector& used_for_size) noexcept {
  static_assert(test_protocol_conformance<
                    NewtonianEuler::NumericalFluxes::Hllc<Dim, Frame>,
                    dg::protocols::NumericalFlux>,
                "Failed testing protocol conformance");

  pypp::check_with_random_values<9>(
      &apply_hllc<Dim, Frame>,
      "Evolution.Systems.NewtonianEuler.NumericalFluxes.Hllc",
      {"n_dot_num_f_mass_density", "n_dot_num_f_momentum_density",
       "n_dot_num_f_energy_density"},
      {{{0.0, 3.0},
        {-4.0, 4.0},
        {0.0, 3.0},
        {0.0, 2.0},
        {0.0, 3.0},
        {-4.0, 4.0},
        {0.0, 3.0},
        {0.0, 2.0},
        {-0.5, 0.5}}},
      used_for_size);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.NumericalFluxes.Hllc",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};

  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_flux, (1, 2, 3), (Frame::Inertial));
}
