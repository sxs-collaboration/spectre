// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarWave/UpwindPenaltyCorrection.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim>
void upwind_penalty_flux(
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_pi,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        boundary_correction_phi,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_psi,

    const Scalar<DataVector>& constraint_gamma2,

    const Scalar<DataVector>& v_psi_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero_int,
    const Scalar<DataVector>& v_plus_int, const Scalar<DataVector>& v_minus_int,
    const tnsr::a<DataVector, 3, Frame::Inertial>& tensor_char_speeds_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_int,

    const Scalar<DataVector>& v_psi_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero_ext,
    const Scalar<DataVector>& v_plus_ext, const Scalar<DataVector>& v_minus_ext,
    const tnsr::a<DataVector, 3, Frame::Inertial>& tensor_char_speeds_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_ext) noexcept {
  const size_t num_pts = v_plus_int.begin()->size();
  const DataVector used_for_size{num_pts,
                                 std::numeric_limits<double>::signaling_NaN()};

  ScalarWave::UpwindPenaltyCorrection<Dim> correction_computer{};

  auto packaged_data_int = TestHelpers::NumericalFluxes::get_packaged_data(
      correction_computer, used_for_size, v_psi_int, v_zero_int, v_plus_int,
      v_minus_int,
      std::array<DataVector, 4>{
          {tensor_char_speeds_int[0], tensor_char_speeds_int[1],
           tensor_char_speeds_int[2], tensor_char_speeds_int[3]}},
      constraint_gamma2, unit_normal_int);
  auto packaged_data_ext = TestHelpers::NumericalFluxes::get_packaged_data(
      correction_computer, used_for_size, v_psi_ext, v_zero_ext, v_plus_ext,
      v_minus_ext,
      std::array<DataVector, 4>{
          {tensor_char_speeds_ext[0], tensor_char_speeds_ext[1],
           tensor_char_speeds_ext[2], tensor_char_speeds_ext[3]}},
      constraint_gamma2, unit_normal_ext);

  dg::NumericalFluxes::normal_dot_numerical_fluxes(
      correction_computer, packaged_data_int, packaged_data_ext,
      boundary_correction_pi, boundary_correction_phi,
      boundary_correction_psi);
}

template <size_t Dim>
void check_upwind_penalty_flux(const size_t num_pts_per_dim) noexcept {
  pypp::check_with_random_values<1>(
      &upwind_penalty_flux<Dim>, "UpwindPenaltyCorrection",
      {"pi_upwind_penalty_correction", "phi_upwind_penalty_correction",
       "psi_upwind_penalty_correction"},
      {{{-1.0, 1.0}}}, DataVector{pow<Dim>(num_pts_per_dim)});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarWave.UpwindPenaltyCorrection",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/ScalarWave"};

  constexpr size_t num_pts_per_dim = 5;
  check_upwind_penalty_flux<1>(num_pts_per_dim);
  check_upwind_penalty_flux<2>(num_pts_per_dim);
  check_upwind_penalty_flux<3>(num_pts_per_dim);
}
