// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/MathFunctions/TestHelpers.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Utilities/Gsl.hpp"

template <size_t VolumeDim, typename Fr>
class MathFunction;

namespace Frame {
struct Grid;
struct Inertial;
}  // namespace Frame

namespace {
template <size_t VolumeDim, typename DataType, typename Fr>
void test_gaussian_random(const DataType& used_for_size) {
  Parallel::register_classes_with_charm<
      MathFunctions::Gaussian<VolumeDim, Fr>>();

  // Generate the amplitude and width
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> real_dis(-1, 1);
  std::uniform_real_distribution<> positive_dis(0, 1);

  const double amplitude = positive_dis(gen);
  // If the width is too small then the terms in the second derivative
  // can become very large and fail the test due to rounding errors.
  const double width = positive_dis(gen) + 0.5;

  // Generate the center
  std::array<double, VolumeDim> center{};
  for (size_t i = 0; i < VolumeDim; ++i) {
    gsl::at(center, i) = real_dis(gen);
  }

  MathFunctions::Gaussian<VolumeDim, Fr> gauss{amplitude, width, center};

  TestHelpers::MathFunctions::check(std::move(gauss), "gaussian", used_for_size,
                                    {{{-1.0, 1.0}}}, amplitude, width, center);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.MathFunctions.Gaussian",
                  "[PointwiseFunctions][Unit]") {
  const DataVector dv{5};

  pypp::SetupLocalPythonEnvironment{"PointwiseFunctions/MathFunctions/Python"};

  using VolumeDims = tmpl::integral_list<size_t, 1, 2, 3>;
  using Frames = tmpl::list<Frame::Grid, Frame::Inertial>;

  tmpl::for_each<VolumeDims>([&dv](auto dim_v) {
    using VolumeDim = typename decltype(dim_v)::type;
    tmpl::for_each<Frames>([&dv](auto frame_v) {
      using Fr = typename decltype(frame_v)::type;
      test_gaussian_random<VolumeDim::value, DataVector, Fr>(dv);
      test_gaussian_random<VolumeDim::value, double, Fr>(
          std::numeric_limits<double>::signaling_NaN());
    });
  });
}

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.MathFunctions.Gaussian.Factory",
                  "[PointwiseFunctions][Unit]") {
  TestHelpers::test_factory_creation<
      MathFunction<1, Frame::Inertial>,
      MathFunctions::Gaussian<1, Frame::Inertial>>(
      "Gaussian:\n"
      "  Amplitude: 3\n"
      "  Width: 2\n"
      "  Center: -9");

  const double amplitude_3d{4.0};
  const double width_3d{1.5};
  const std::array<double, 3> center_3d{{1.1, -2.2, 3.3}};
  const MathFunctions::Gaussian<3, Frame::Inertial> gauss_3d{
      amplitude_3d, width_3d, center_3d};
  const auto created_gauss =
      TestHelpers::test_creation<MathFunctions::Gaussian<3, Frame::Inertial>>(
          "Amplitude: 4.0\n"
          "Width: 1.5\n"
          "Center: [1.1, -2.2, 3.3]");
  CHECK(created_gauss == gauss_3d);
  const auto created_gauss_mathfunction = TestHelpers::test_factory_creation<
      MathFunction<3, Frame::Inertial>,
      MathFunctions::Gaussian<3, Frame::Inertial>>(
      "Gaussian:\n"
      "  Amplitude: 4.0\n"
      "  Width: 1.5\n"
      "  Center: [1.1, -2.2, 3.3]");
}
