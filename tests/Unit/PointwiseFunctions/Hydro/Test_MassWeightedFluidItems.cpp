// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/MassWeightedFluidItems.hpp"

namespace hydro {

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.MassWeightedFluidItems",
                  "[Unit][Hydro]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/Hydro/");
  const DataVector used_for_size(5);
  pypp::check_with_random_values<1>(
      static_cast<void (*)(gsl::not_null<Scalar<DataVector>*>,
                           const Scalar<DataVector>&,
                           const Scalar<DataVector>&)>(
          &mass_weighted_internal_energy<DataVector>),
      "TestFunctions", {"mass_weighted_internal_energy"}, {{{0.0, 1.0}}},
      used_for_size);
  pypp::check_with_random_values<1>(
      static_cast<Scalar<DataVector> (*)(const Scalar<DataVector>&,
                                         const Scalar<DataVector>&)>(
          &mass_weighted_internal_energy<DataVector>),
      "TestFunctions", "mass_weighted_internal_energy", {{{0.0, 1.0}}},
      used_for_size);
  pypp::check_with_random_values<1>(
      static_cast<void (*)(gsl::not_null<Scalar<DataVector>*>,
                           const Scalar<DataVector>&,
                           const Scalar<DataVector>&)>(
          &mass_weighted_kinetic_energy<DataVector>),
      "TestFunctions", {"mass_weighted_kinetic_energy"}, {{{0.0, 1.0}}},
      used_for_size);
  pypp::check_with_random_values<1>(
      static_cast<Scalar<DataVector> (*)(const Scalar<DataVector>&,
                                         const Scalar<DataVector>&)>(
          &mass_weighted_kinetic_energy<DataVector>),
      "TestFunctions", "mass_weighted_kinetic_energy", {{{0.0, 1.0}}},
      used_for_size);
  pypp::check_with_random_values<1>(
      static_cast<void (*)(gsl::not_null<Scalar<DataVector>*>,
                           const Scalar<DataVector>&, const Scalar<DataVector>&,
                           const tnsr::I<DataVector, 1, Frame::Inertial>&,
                           const tnsr::ii<DataVector, 1, Frame::Inertial>&,
                           const Scalar<DataVector>& lapse,
                           const tnsr::I<DataVector, 1, Frame::Inertial>&)>(
          &tilde_d_unbound_ut_criterion<DataVector, 1, Frame::Inertial>),
      "TestFunctions", {"tilde_d_unbound_ut_criterion"}, {{{0.0, 1.0}}},
      used_for_size);
  pypp::check_with_random_values<1>(
      static_cast<Scalar<DataVector> (*)(
          const Scalar<DataVector>&, const Scalar<DataVector>&,
          const tnsr::I<DataVector, 1, Frame::Inertial>&,
          const tnsr::ii<DataVector, 1, Frame::Inertial>&,
          const Scalar<DataVector>& lapse,
          const tnsr::I<DataVector, 1, Frame::Inertial>&)>(
          &tilde_d_unbound_ut_criterion<DataVector, 1, Frame::Inertial>),
      "TestFunctions", "tilde_d_unbound_ut_criterion", {{{0.0, 1.0}}},
      used_for_size);
  pypp::check_with_random_values<1>(
      static_cast<tnsr::I<DataVector, 1, Frame::Inertial> (*)(
          const Scalar<DataVector>& tilde_d,
          const tnsr::I<DataVector, 1, Frame::Grid>& grid_coords,
          const tnsr::I<DataVector, 1, Frame::Inertial>& compute_coords)>(
          &mass_weighted_coords<::hydro::HalfPlaneIntegralMask::None,
                                DataVector, 1, Frame::Inertial>),
      "TestFunctions", "mass_weighted_coords_none", {{{0.0, 1.0}}},
      used_for_size);
  pypp::check_with_random_values<3>(
      static_cast<tnsr::I<DataVector, 3, Frame::Inertial> (*)(
          const Scalar<DataVector>& tilde_d,
          const tnsr::I<DataVector, 3, Frame::Grid>& grid_coords,
          const tnsr::I<DataVector, 3, Frame::Inertial>& compute_coords)>(
          &mass_weighted_coords<::hydro::HalfPlaneIntegralMask::None,
                                DataVector, 3, Frame::Inertial>),
      "TestFunctions", "mass_weighted_coords_none", {{{0.0, 1.0}}},
      used_for_size);
  pypp::check_with_random_values<1>(
      static_cast<tnsr::I<DataVector, 1, Frame::Inertial> (*)(
          const Scalar<DataVector>& tilde_d,
          const tnsr::I<DataVector, 1, Frame::Grid>& grid_coords,
          const tnsr::I<DataVector, 1, Frame::Inertial>& compute_coords)>(
          &mass_weighted_coords<::hydro::HalfPlaneIntegralMask::PositiveXOnly,
                                DataVector, 1, Frame::Inertial>),
      "TestFunctions", "mass_weighted_coords_a", {{{-1.0, 1.0}}},
      used_for_size);
  pypp::check_with_random_values<3>(
      static_cast<tnsr::I<DataVector, 3, Frame::Inertial> (*)(
          const Scalar<DataVector>& tilde_d,
          const tnsr::I<DataVector, 3, Frame::Grid>& grid_coords,
          const tnsr::I<DataVector, 3, Frame::Inertial>& compute_coords)>(
          &mass_weighted_coords<::hydro::HalfPlaneIntegralMask::PositiveXOnly,
                                DataVector, 3, Frame::Inertial>),
      "TestFunctions", "mass_weighted_coords_a", {{{-1.0, 1.0}}},
      used_for_size);
  pypp::check_with_random_values<1>(
      static_cast<tnsr::I<DataVector, 1, Frame::Inertial> (*)(
          const Scalar<DataVector>& tilde_d,
          const tnsr::I<DataVector, 1, Frame::Grid>& grid_coords,
          const tnsr::I<DataVector, 1, Frame::Inertial>& compute_coords)>(
          &mass_weighted_coords<::hydro::HalfPlaneIntegralMask::NegativeXOnly,
                                DataVector, 1, Frame::Inertial>),
      "TestFunctions", "mass_weighted_coords_b", {{{-1.0, 1.0}}},
      used_for_size);
  pypp::check_with_random_values<3>(
      static_cast<tnsr::I<DataVector, 3, Frame::Inertial> (*)(
          const Scalar<DataVector>& tilde_d,
          const tnsr::I<DataVector, 3, Frame::Grid>& grid_coords,
          const tnsr::I<DataVector, 3, Frame::Inertial>& compute_coords)>(
          &mass_weighted_coords<::hydro::HalfPlaneIntegralMask::NegativeXOnly,
                                DataVector, 3, Frame::Inertial>),
      "TestFunctions", "mass_weighted_coords_b", {{{-1.0, 1.0}}},
      used_for_size);
}

}  // namespace hydro
