// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/ConstraintDamping/TestHelpers.hpp"
#include "PointwiseFunctions/ConstraintDamping/DampingFunction.hpp"
#include "PointwiseFunctions/ConstraintDamping/TimeDependentTripleGaussian.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace {
template <typename DataType>
void test_triple_gaussian_random(const DataType& used_for_size) {
  register_derived_classes_with_charm<
      ConstraintDamping::TimeDependentTripleGaussian>();

  // Generate the amplitude and width
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> real_dis(-1.0, 1.0);
  std::uniform_real_distribution<> positive_dis(0.0, 1.0);

  const double constant = real_dis(gen);

  const double amplitude_1{positive_dis(gen)};
  const double amplitude_2{positive_dis(gen)};
  const double amplitude_3{positive_dis(gen)};

  const double width_1{positive_dis(gen) + 0.5};
  const double width_2{positive_dis(gen) + 0.5};
  const double width_3{positive_dis(gen) + 0.5};

  // Generate the center
  std::array<double, 3> center_1{};
  std::array<double, 3> center_2{};
  std::array<double, 3> center_3{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(center_1, i) = real_dis(gen);
    gsl::at(center_2, i) = real_dis(gen);
    gsl::at(center_3, i) = real_dis(gen);
  }

  // Name of FunctionOfTime to read. This must match up with the hard coded name
  // in the TimeDependentTripleGaussian
  const std::string function_of_time_for_scaling{"Expansion"s};

  const ConstraintDamping::TimeDependentTripleGaussian triple_gauss{
      constant, amplitude_1, width_1, center_1, amplitude_2,      width_2,
      center_2, amplitude_3, width_3, center_3, "ExpansionFactor"};

  TestHelpers::ConstraintDamping::check(
      triple_gauss, "TimeDependentTripleGaussian", used_for_size,
      {{{-1.0, 1.0}}}, {function_of_time_for_scaling}, constant, amplitude_1,
      width_1, center_1, amplitude_2, width_2, center_2, amplitude_3, width_3,
      center_3);

  const std::unique_ptr<ConstraintDamping::TimeDependentTripleGaussian>
      triple_gauss_unique_ptr =
          std::make_unique<ConstraintDamping::TimeDependentTripleGaussian>(
              constant, amplitude_1, width_1, center_1, amplitude_2, width_2,
              center_2, amplitude_3, width_3, center_3, "ExpansionFactor");

  TestHelpers::ConstraintDamping::check(
      triple_gauss_unique_ptr->get_clone(), "TimeDependentTripleGaussian",
      used_for_size, {{{-1.0, 1.0}}}, {function_of_time_for_scaling}, constant,
      amplitude_1, width_1, center_1, amplitude_2, width_2, center_2,
      amplitude_3, width_3, center_3);

  const std::unique_ptr<ConstraintDamping::TimeDependentTripleGaussian>
      triple_gauss_object_centers_unique_ptr =
          std::make_unique<ConstraintDamping::TimeDependentTripleGaussian>(
              constant, amplitude_1, width_1, std::nullopt, amplitude_2,
              width_2, std::nullopt, amplitude_3, width_3, center_3,
              "ObjectCenters");

  REQUIRE(dynamic_cast<const ConstraintDamping::TimeDependentTripleGaussian&>(
              *triple_gauss_object_centers_unique_ptr->get_clone()) ==
          *triple_gauss_object_centers_unique_ptr);
  TestHelpers::ConstraintDamping::check(
      triple_gauss_object_centers_unique_ptr->get_clone(),
      "TimeDependentTripleGaussianObjectCenters", used_for_size,
      {{{-1.0, 1.0}}}, {"GridCenters"}, constant, amplitude_1, width_1,
      center_1, amplitude_2, width_2, center_2, amplitude_3, width_3, center_3);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.ConstraintDamp.TimeDep3Gauss",
                  "[PointwiseFunctions][Unit]") {
  const DataVector dv{5};

  pypp::SetupLocalPythonEnvironment{
      "PointwiseFunctions/ConstraintDamping/Python"};

  test_triple_gaussian_random<DataVector>(dv);
  test_triple_gaussian_random<double>(
      std::numeric_limits<double>::signaling_NaN());

  const double constant_3d{5.0};
  const double amplitude_1_3d{4.0};
  const double width_1_3d{1.5};
  const std::array<double, 3> center_1_3d{{1.1, -2.2, 3.3}};
  const double amplitude_2_3d{3.0};
  const double width_2_3d{2.0};
  const std::array<double, 3> center_2_3d{{4.4, -5.5, 6.6}};
  const double amplitude_3_3d{5.0};
  const double width_3_3d{1.0};
  const std::array<double, 3> center_3_3d{{7.7, -8.8, 9.9}};

  const ConstraintDamping::TimeDependentTripleGaussian triple_gauss_3d{
      constant_3d,    amplitude_1_3d, width_1_3d,       center_1_3d,
      amplitude_2_3d, width_2_3d,     center_2_3d,      amplitude_3_3d,
      width_3_3d,     center_3_3d,    "ExpansionFactor"};
  const auto created_triple_gauss = TestHelpers::test_creation<
      ConstraintDamping::TimeDependentTripleGaussian>(
      "Constant: 5.0\n"
      "Gaussian1:\n"
      "  Amplitude: 4.0\n"
      "  Width: 1.5\n"
      "  Center: [1.1, -2.2, 3.3]\n"
      "Gaussian2:\n"
      "  Amplitude: 3.0\n"
      "  Width: 2.0\n"
      "  Center: [4.4, -5.5, 6.6]\n"
      "Gaussian3:\n"
      "  Amplitude: 5.0\n"
      "  Width: 1.0\n"
      "  Center: [7.7, -8.8, 9.9]\n"
      "MovementMethod: ExpansionFactor\n");
  CHECK(created_triple_gauss == triple_gauss_3d);
  CHECK_FALSE(created_triple_gauss != triple_gauss_3d);
  const auto created_triple_gauss_gh_damping_function =
      TestHelpers::test_creation<
          std::unique_ptr<ConstraintDamping::DampingFunction<3, Frame::Grid>>>(
          "TimeDependentTripleGaussian:\n"
          "  Constant: 5.0\n"
          "  Gaussian1:\n"
          "    Amplitude: 4.0\n"
          "    Width: 1.5\n"
          "    Center: [1.1, -2.2, 3.3]\n"
          "  Gaussian2:\n"
          "    Amplitude: 3.0\n"
          "    Width: 2.0\n"
          "    Center: [4.4, -5.5, 6.6]\n"
          "  Gaussian3:\n"
          "    Amplitude: 5.0\n"
          "    Width: 1.0\n"
          "    Center: [7.7, -8.8, 9.9]\n"
          "  MovementMethod: ExpansionFactor\n");

  test_serialization(triple_gauss_3d);
}
