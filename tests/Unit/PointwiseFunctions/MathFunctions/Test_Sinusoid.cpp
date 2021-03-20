// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/MathFunctions/TestHelpers.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "PointwiseFunctions/MathFunctions/Sinusoid.hpp"

template <size_t VolumeDim, typename Fr>
class MathFunction;

namespace Frame {
struct Grid;
struct Inertial;
}  // namespace Frame

namespace {
template <size_t VolumeDim, typename DataType, typename Fr>
void test_sinusoid_random(const DataType& used_for_size) noexcept {
  Parallel::register_derived_classes_with_charm<
      MathFunctions::Sinusoid<VolumeDim, Fr>>();

  // Generate the amplitude and width
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> real_dis(-1, 1);

  const double amplitude = real_dis(gen);
  // If the width is too small then the terms in the second derivative
  // can become very large and fail the test due to rounding errors.
  const double wavenumber = real_dis(gen);

  // Generate the center
  const double phase = real_dis(gen);

  MathFunctions::Sinusoid<VolumeDim, Fr> sinusoid{amplitude, wavenumber, phase};

  TestHelpers::MathFunctions::check(std::move(sinusoid), "sinusoid",
                                    used_for_size, {{{-1.0, 1.0}}}, amplitude,
                                    wavenumber, phase);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.MathFunctions.Sinusoid",
                  "[PointwiseFunctions][Unit]") {
  const DataVector dv{5};

  pypp::SetupLocalPythonEnvironment{"PointwiseFunctions/MathFunctions/Python"};

  using Frames = tmpl::list<Frame::Grid, Frame::Inertial>;
  tmpl::for_each<Frames>([&dv](auto frame_v) {
    using Fr = typename decltype(frame_v)::type;
    test_sinusoid_random<1, DataVector, Fr>(dv);
    test_sinusoid_random<1, double, Fr>(
        std::numeric_limits<double>::signaling_NaN());
  });
}

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.MathFunctions.Sinusoid.Factory",
                  "[PointwiseFunctions][Unit]") {
  TestHelpers::test_creation<std::unique_ptr<MathFunction<1, Frame::Inertial>>>(
      "Sinusoid:\n"
      "  Amplitude: 3\n"
      "  Wavenumber: 2\n"
      "  Phase: -9");
  TestHelpers::test_creation<std::unique_ptr<MathFunction<1, Frame::Inertial>>>(
      "Sinusoid:\n"
      "  Amplitude: 3\n"
      "  Wavenumber: 2\n"
      "  Phase: -9");
  // Catch requires us to have at least one CHECK in each test
  // The Unit.PointwiseFunctions.MathFunctions.Sinusoid.Factory does not need to
  // check anything
  CHECK(true);
}
