// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Ccz4/DerivLapse.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <size_t Dim, typename DataType>
void test_grad_grad_lapse(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::ij<DataType, Dim, Frame::Inertial> (*)(
          const Scalar<DataType>&,
          const tnsr::Ijj<DataType, Dim, Frame::Inertial>&,
          const tnsr::i<DataType, Dim, Frame::Inertial>&,
          const tnsr::ij<DataType, Dim, Frame::Inertial>&)>(
          &::Ccz4::grad_grad_lapse<Dim, Frame::Inertial, DataType>),
      "DerivLapse", "grad_grad_lapse", {{{-1., 1.}}}, used_for_size);
}

template <size_t Dim, typename DataType>
void test_divergence_lapse(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<Scalar<DataType> (*)(
          const Scalar<DataType>&,
          const tnsr::II<DataType, Dim, Frame::Inertial>&,
          const tnsr::ij<DataType, Dim, Frame::Inertial>&)>(
          &::Ccz4::divergence_lapse<Dim, Frame::Inertial, DataType>),
      "DerivLapse", "divergence_lapse", {{{-1., 1.}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.DerivLapse",
                  "[Evolution][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env("Evolution/Systems/Ccz4/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_grad_grad_lapse, (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_divergence_lapse, (1, 2, 3));
}
