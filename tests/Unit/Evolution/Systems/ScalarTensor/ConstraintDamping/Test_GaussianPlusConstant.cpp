// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/ScalarTensor/ConstraintDamping/DampingFunction.hpp"
#include "Evolution/Systems/ScalarTensor/ConstraintDamping/GaussianPlusConstant.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ScalarTensor.ConstraintDamp.GaussPlusConst",
    "[PointwiseFunctions][Unit]") {
  const DataVector dv{5};

  TestHelpers::test_creation<std::unique_ptr<
      ScalarTensor::ConstraintDamping::DampingFunction<1, Frame::Inertial>>>(
      "GaussianPlusConstant:\n"
      "  Constant: 4.0\n"
      "  Amplitude: 3.0\n"
      "  Width: 2.0\n"
      "  Center: [-9.0]");

  const double constant_3d{5.0};
  const double amplitude_3d{4.0};
  const double width_3d{1.5};
  const std::array<double, 3> center_3d{{1.1, -2.2, 3.3}};
  const ScalarTensor::ConstraintDamping::GaussianPlusConstant<3,
                                                              Frame::Inertial>
      gauss_plus_const_3d{constant_3d, amplitude_3d, width_3d, center_3d};
  const auto created_gauss_plus_const =
      TestHelpers::test_creation<ScalarTensor::ConstraintDamping::
                                     GaussianPlusConstant<3, Frame::Inertial>>(
          "Constant: 5.0\n"
          "Amplitude: 4.0\n"
          "Width: 1.5\n"
          "Center: [1.1, -2.2, 3.3]");
  CHECK(created_gauss_plus_const == gauss_plus_const_3d);
  const auto created_gauss_gh_damping_function =
      TestHelpers::test_creation<std::unique_ptr<
        ScalarTensor::ConstraintDamping::DampingFunction<3, Frame::Inertial>>>(
          "GaussianPlusConstant:\n"
          "  Constant: 5.0\n"
          "  Amplitude: 4.0\n"
          "  Width: 1.5\n"
          "  Center: [1.1, -2.2, 3.3]");

  test_serialization(gauss_plus_const_3d);
}
