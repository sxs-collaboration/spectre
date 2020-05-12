// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/UpwindPenaltyCorrection.hpp"
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
void upwind_penalty_boundary_correction(
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        boundary_correction_spacetime_metric,
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        boundary_correction_pi,
    const gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
        boundary_correction_phi,
    const Scalar<DataVector>& constraint_gamma2,

    const tnsr::aa<DataVector, Dim, Frame::Inertial>& v_spacetime_metric_int,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& v_zero_int,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& v_plus_int,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& v_minus_int,
    const tnsr::a<DataVector, 3, Frame::Inertial>& tensor_char_speeds_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_int,

    const tnsr::aa<DataVector, Dim, Frame::Inertial>& v_spacetime_metric_ext,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& v_zero_ext,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& v_plus_ext,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& v_minus_ext,
    const tnsr::a<DataVector, 3, Frame::Inertial>& tensor_char_speeds_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_ext) noexcept {
  const size_t num_pts = v_plus_int.begin()->size();
  const DataVector used_for_size{num_pts,
                                 std::numeric_limits<double>::signaling_NaN()};

  GeneralizedHarmonic::UpwindPenaltyCorrection<Dim> correction_computer{};

  auto packaged_data_int = TestHelpers::NumericalFluxes::get_packaged_data(
      correction_computer, used_for_size, v_spacetime_metric_int, v_zero_int,
      v_plus_int, v_minus_int,
      std::array<DataVector, 4>{
          {tensor_char_speeds_int[0], tensor_char_speeds_int[1],
           tensor_char_speeds_int[2], tensor_char_speeds_int[3]}},
      constraint_gamma2, unit_normal_int);
  auto packaged_data_ext = TestHelpers::NumericalFluxes::get_packaged_data(
      correction_computer, used_for_size, v_spacetime_metric_ext, v_zero_ext,
      v_plus_ext, v_minus_ext,
      std::array<DataVector, 4>{
          {tensor_char_speeds_ext[0], tensor_char_speeds_ext[1],
           tensor_char_speeds_ext[2], tensor_char_speeds_ext[3]}},
      constraint_gamma2, unit_normal_ext);

  dg::NumericalFluxes::normal_dot_numerical_fluxes(
      correction_computer, packaged_data_int, packaged_data_ext,
      boundary_correction_spacetime_metric, boundary_correction_pi,
      boundary_correction_phi);
}

template <size_t Dim>
void check_upwind_penalty_boundary_correction(
    const size_t num_pts_per_dim) noexcept {
  pypp::check_with_random_values<1>(
      &upwind_penalty_boundary_correction<Dim>, "UpwindPenaltyCorrection",
      {"spacetime_metric_upwind_penalty_correction",
       "pi_upwind_penalty_correction", "phi_upwind_penalty_correction"},
      {{{-1.0, 1.0}}}, DataVector{pow<Dim>(num_pts_per_dim)});
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.UpwindPenaltyCorrection",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GeneralizedHarmonic"};

  constexpr size_t num_pts_per_dim = 5;
  check_upwind_penalty_boundary_correction<1>(num_pts_per_dim);
  check_upwind_penalty_boundary_correction<2>(num_pts_per_dim);
  check_upwind_penalty_boundary_correction<3>(num_pts_per_dim);
}
