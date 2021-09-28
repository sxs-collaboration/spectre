// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
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
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "Utilities/TMPL.hpp"

template <size_t VolumeDim, typename Fr>
class MathFunction;

namespace Frame {
struct Grid;
struct Inertial;
}  // namespace Frame

namespace {
template <size_t VolumeDim, typename DataType, typename Fr>
void test_pow_x_random(const DataType& used_for_size) {
  Parallel::register_classes_with_charm<MathFunctions::PowX<VolumeDim, Fr>>();

  for (int power = -5; power < 6; ++power) {
    MathFunctions::PowX<VolumeDim, Fr> pow_x{power};
    TestHelpers::MathFunctions::check(std::move(pow_x), "pow_x", used_for_size,
                                      {{{-5.0, 5.0}}},
                                      static_cast<double>(power));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.MathFunctions.PowX",
                  "[PointwiseFunctions][Unit]") {
  const DataVector dv{5};

  pypp::SetupLocalPythonEnvironment{"PointwiseFunctions/MathFunctions/Python"};

  using Frames = tmpl::list<Frame::Grid, Frame::Inertial>;
  tmpl::for_each<Frames>([&dv](auto frame_v) {
    using Fr = typename decltype(frame_v)::type;
    test_pow_x_random<1, DataVector, Fr>(dv);
    test_pow_x_random<1, double, Fr>(
        std::numeric_limits<double>::signaling_NaN());
  });
}

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.MathFunctions.PowX.Factory",
                  "[PointwiseFunctions][Unit]") {
  TestHelpers::test_factory_creation<MathFunction<1, Frame::Inertial>,
                                     MathFunctions::PowX<1, Frame::Inertial>>(
      "PowX:\n    Power: 3");
  TestHelpers::test_factory_creation<MathFunction<1, Frame::Inertial>,
                                     MathFunctions::PowX<1, Frame::Inertial>>(
      "PowX:\n    Power: 3");
  // Catch requires us to have at least one CHECK in each test
  // The Unit.PointwiseFunctions.MathFunctions.PowX.Factory does not need to
  // check anything
  CHECK(true);
}
