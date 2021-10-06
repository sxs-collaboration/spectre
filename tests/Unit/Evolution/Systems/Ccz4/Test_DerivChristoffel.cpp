// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/DerivChristoffel.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <size_t Dim, typename DataType>
void test_compute_deriv_conformal_christoffel_second_kind(
    const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::iJkk<DataType, Dim, Frame::Inertial> (*)(
          const tnsr::II<DataType, Dim, Frame::Inertial>&,
          const tnsr::ijj<DataType, Dim, Frame::Inertial>&,
          const tnsr::ijkk<DataType, Dim, Frame::Inertial>&,
          const tnsr::iJJ<DataType, Dim, Frame::Inertial>&)>(
          &Ccz4::deriv_conformal_christoffel_second_kind<Dim, Frame::Inertial,
                                                         DataType>),
      "DerivChristoffel", "deriv_conformal_christoffel_second_kind",
      {{{-1., 1.}}}, used_for_size);
}

template <size_t Dim, typename DataType>
void test_compute_deriv_contracted_conformal_christoffel_second_kind(
    const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::iJ<DataType, Dim, Frame::Inertial> (*)(
          const tnsr::II<DataType, Dim, Frame::Inertial>&,
          const tnsr::iJJ<DataType, Dim, Frame::Inertial>&,
          const tnsr::Ijj<DataType, Dim, Frame::Inertial>&,
          const tnsr::iJkk<DataType, Dim, Frame::Inertial>&)>(
          &Ccz4::deriv_contracted_conformal_christoffel_second_kind<
              Dim, Frame::Inertial, DataType>),
      "DerivChristoffel", "deriv_contracted_conformal_christoffel_second_kind",
      {{{-1., 1.}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.DerivChristoffel",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env("Evolution/Systems/Ccz4/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(
      test_compute_deriv_conformal_christoffel_second_kind, (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(
      test_compute_deriv_contracted_conformal_christoffel_second_kind,
      (1, 2, 3));
}
