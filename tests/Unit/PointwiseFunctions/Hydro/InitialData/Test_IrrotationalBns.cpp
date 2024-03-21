// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "PointwiseFunctions/Hydro/InitialData/IrrotationalBns.hpp"

namespace hydro::initial_data::irrotational_bns {
SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.InitialData.IrrotationalBns",
                  "[Unit][Hydro][Elliptic]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/Hydro/InitialData");
  const DataVector used_for_size(5);
  pypp::check_with_random_values<1>(
      static_cast<tnsr::I<DataVector, 3> (*)(const tnsr::I<DataVector, 3>&,
                                             const tnsr::I<DataVector, 3>&)>(
          &rotational_shift),
      "IrrotationalBns", "rotational_shift", {{{0.0, 1.0}}}, used_for_size);
  pypp::check_with_random_values<1>(
      static_cast<tnsr::II<DataVector, 3> (*)(const tnsr::I<DataVector, 3>&,
                                              const Scalar<DataVector>&)>(
          &rotational_shift_stress),
      "IrrotationalBns", "rotational_shift_stress", {{{0.0, 1.0}}},
      used_for_size);
  pypp::check_with_random_values<1>(
      static_cast<tnsr::iJ<DataVector, 3> (*)(
          const tnsr::I<DataVector, 3>&, const tnsr::iJ<DataVector, 3>&,
          const Scalar<DataVector>&, const tnsr::i<DataVector, 3>&,
          const tnsr::iJ<DataVector, 3>&)>(
          &derivative_rotational_shift_over_lapse_squared),
      "IrrotationalBns", "derivative_rotational_shift_over_lapse_squared",
      {{{0.0, 1.0}}}, used_for_size);
  pypp::check_with_random_values<1>(
      static_cast<Scalar<DataVector> (*)(
          const tnsr::I<DataVector, 3>&, const Scalar<DataVector>&,
          const tnsr::i<DataVector, 3>&, const tnsr::II<DataVector, 3>&,
          double)>(&specific_enthalpy_squared),
      "IrrotationalBns", "specific_enthalpy_squared", {{{0.0, 1.0}}},
      used_for_size);
  pypp::check_with_random_values<1>(
      static_cast<tnsr::I<DataVector, 3> (*)(
          const tnsr::I<DataVector, 3>&, double, const Scalar<DataVector>&)>(
          &spatial_rotational_killing_vector),
      "IrrotationalBns", "spatial_rotational_killing_vector", {{{0.0, 1.0}}},
      used_for_size);
  pypp::check_with_random_values<1>(
      static_cast<Scalar<DataVector> (*)(const tnsr::I<DataVector, 3>&, double,
                                         const Scalar<DataVector>&)>(
          &divergence_spatial_rotational_killing_vector),
      "IrrotationalBns", "divergence_spatial_rotational_killing_vector",
      {{{0.0, 1.0}}}, used_for_size);
}
}  // namespace hydro::initial_data::irrotational_bns
