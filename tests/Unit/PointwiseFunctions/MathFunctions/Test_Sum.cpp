// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <random>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/MathFunctions/TestHelpers.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "PointwiseFunctions/MathFunctions/Sum.hpp"
#include "Utilities/Gsl.hpp"

template <size_t VolumeDim, typename Fr>
class MathFunction;

namespace Frame {
struct Grid;
struct Inertial;
}  // namespace Frame

namespace {
template <size_t VolumeDim, typename DataType, typename Fr>
void test_sum_random(const DataType& used_for_size) noexcept {
  Parallel::register_derived_classes_with_charm<
      MathFunctions::Gaussian<VolumeDim, Fr>>();
  Parallel::register_derived_classes_with_charm<
      MathFunctions::Sum<VolumeDim, Fr>>();

  // Generate the amplitude and width
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> real_dis(-1, 1);
  std::uniform_real_distribution<> positive_dis(0, 1);

  const double amplitude_A = positive_dis(gen);
  const double amplitude_B = positive_dis(gen);

  // If the width is too small then the terms in the second derivative
  // can become very large and fail the test due to rounding errors.
  const double width_A = positive_dis(gen) + 0.5;
  const double width_B = positive_dis(gen) + 0.5;

  // Generate the center
  std::array<double, VolumeDim> center_A{};
  std::array<double, VolumeDim> center_B{};
  for (size_t i = 0; i < VolumeDim; ++i) {
    gsl::at(center_A, i) = real_dis(gen);
    gsl::at(center_B, i) = real_dis(gen);
  }

  MathFunctions::Gaussian<VolumeDim, Fr> gauss_A{amplitude_A, width_A,
                                                 center_A};
  MathFunctions::Gaussian<VolumeDim, Fr> gauss_B{amplitude_B, width_B,
                                                 center_B};
  MathFunctions::Sum<VolumeDim, Fr> sum{
      std::make_unique<MathFunctions::Gaussian<VolumeDim, Fr>>(
          std::move(gauss_A)),
      std::make_unique<MathFunctions::Gaussian<VolumeDim, Fr>>(
          std::move(gauss_B))};

  TestHelpers::MathFunctions::check(std::move(sum), "sum", used_for_size,
                                    {{{-1.0, 1.0}}}, amplitude_A, width_A,
                                    center_A, amplitude_B, width_B, center_B);

  const double amplitude_C = positive_dis(gen);
  const double amplitude_D = positive_dis(gen);
  const double amplitude_E = positive_dis(gen);
  const double width_C = positive_dis(gen) + 0.5;
  const double width_D = positive_dis(gen) + 0.5;
  const double width_E = positive_dis(gen) + 0.5;
  std::array<double, VolumeDim> center_C{};
  std::array<double, VolumeDim> center_D{};
  std::array<double, VolumeDim> center_E{};
  for (size_t i = 0; i < VolumeDim; ++i) {
    gsl::at(center_C, i) = real_dis(gen);
    gsl::at(center_D, i) = real_dis(gen);
    gsl::at(center_E, i) = real_dis(gen);
  }
  MathFunctions::Gaussian<VolumeDim, Fr> gauss_C{amplitude_C, width_C,
                                                 center_C};
  MathFunctions::Gaussian<VolumeDim, Fr> gauss_D{amplitude_D, width_D,
                                                 center_D};
  MathFunctions::Gaussian<VolumeDim, Fr> gauss_E{amplitude_E, width_E,
                                                 center_E};

  MathFunctions::Sum<VolumeDim, Fr> sum_cd{
      std::make_unique<MathFunctions::Gaussian<VolumeDim, Fr>>(
          std::move(gauss_C)),
      std::make_unique<MathFunctions::Gaussian<VolumeDim, Fr>>(
          std::move(gauss_D))};

  MathFunctions::Sum<VolumeDim, Fr> sum_of_sum{
      std::make_unique<MathFunctions::Gaussian<VolumeDim, Fr>>(
          std::move(gauss_E)),
      std::make_unique<MathFunctions::Sum<VolumeDim, Fr>>(std::move(sum_cd))};

  TestHelpers::MathFunctions::check(std::move(sum_of_sum), "sum_of_sum",
                                    used_for_size, {{{-1.0, 1.0}}}, amplitude_C,
                                    width_C, center_C, amplitude_D, width_D,
                                    center_D, amplitude_E, width_E, center_E);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.MathFunctions.Sum",
                  "[PointwiseFunctions][Unit]") {
  const DataVector dv{{std::numeric_limits<double>::signaling_NaN(),
                       std::numeric_limits<double>::signaling_NaN()}};

  pypp::SetupLocalPythonEnvironment{"PointwiseFunctions/MathFunctions/Python"};

  using VolumeDims = tmpl::integral_list<size_t, 1, 2, 3>;
  using Frames = tmpl::list<Frame::Grid, Frame::Inertial>;

  tmpl::for_each<VolumeDims>([&dv](auto dim_v) {
    using VolumeDim = typename decltype(dim_v)::type;
    tmpl::for_each<Frames>([&dv](auto frame_v) {
      using Fr = typename decltype(frame_v)::type;
      test_sum_random<VolumeDim::value, DataVector, Fr>(dv);
      test_sum_random<VolumeDim::value, double, Fr>(
          std::numeric_limits<double>::signaling_NaN());
    });
  });
}

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.MathFunctions.Sum.Factory",
                  "[PointwiseFunctions][Unit]") {
  // Tests must have at least one assertion to pass, but here just test
  // that factory creation works. Other MathFunctions test operator== here,
  // but Sum does not support operator== because Sum objects cannot
  // be copied (since their std::unique_ptr members) cannot be copied
  CHECK(true);

  TestHelpers::test_factory_creation<MathFunction<1, Frame::Inertial>>(
      "Sum:\n"
      "  MathFunctionA:\n"
      "    Sinusoid:\n"
      "      Amplitude: 3\n"
      "      Wavenumber: 2\n"
      "      Phase: -9\n"
      "  MathFunctionB:\n"
      "    PowX:\n"
      "      Power: 4\n");

  const auto created_sum =
      TestHelpers::test_creation<MathFunctions::Sum<3, Frame::Inertial>>(
          "MathFunctionA:\n"
          "  Gaussian:\n"
          "    Amplitude: 4.0\n"
          "    Width: 1.5\n"
          "    Center: [1.1, 2.2, -3.3]\n"
          "MathFunctionB:\n"
          "  Gaussian:\n"
          "    Amplitude: 4.4\n"
          "    Width: 1.4\n"
          "    Center: [1.4, 2.4, -3.4]");
  const auto created_sum_mathfunction =
      TestHelpers::test_factory_creation<MathFunction<3, Frame::Inertial>>(
          "Sum:\n"
          "  MathFunctionA:\n"
          "    Gaussian:\n"
          "      Amplitude: 1.1\n"
          "      Width: 2.2\n"
          "      Center: [1.1, 2.2, -3.3]\n"
          "  MathFunctionB:\n"
          "    Gaussian:\n"
          "      Amplitude: 1.4\n"
          "      Width: 2.4\n"
          "      Center: [1.2, 2.3, -3.4]\n");

  // Test creating a Sum where MathFunctionB is itself a Sum
  const auto created_sum_mathfunction_recursive =
      TestHelpers::test_factory_creation<MathFunction<3, Frame::Inertial>>(
          "Sum:\n"
          "  MathFunctionA:\n"
          "    Sum:\n"
          "      MathFunctionA:\n"
          "        Gaussian:\n"
          "          Amplitude: 1.1\n"
          "          Width: 2.2\n"
          "          Center: [1.2, 2.3, -3.4]\n"
          "      MathFunctionB:\n"
          "        Gaussian:\n"
          "          Amplitude: 1.2\n"
          "          Width: 2.1\n"
          "          Center: [1.1, 2.2, -3.3]\n"
          "  MathFunctionB:\n"
          "    Sum:\n"
          "      MathFunctionA:\n"
          "        Gaussian:\n"
          "          Amplitude: 1.5\n"
          "          Width: 2.1\n"
          "          Center: [1.5, 2.5, -3.5]\n"
          "      MathFunctionB:\n"
          "        Gaussian:\n"
          "          Amplitude: 1.6\n"
          "          Width: 2.6\n"
          "          Center: [1.6, 2.6, -3.6]\n");
}
