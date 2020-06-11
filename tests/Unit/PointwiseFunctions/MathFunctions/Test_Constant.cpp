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
#include "PointwiseFunctions/MathFunctions/Constant.hpp"
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
void test_constant_random(const DataType& used_for_size) noexcept {
  Parallel::register_derived_classes_with_charm<
      MathFunctions::Constant<VolumeDim, Fr>>();

  // Generate the amplitude and width
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> real_dis(-5, 5);

  const double value = real_dis(gen);

  MathFunctions::Constant<VolumeDim, Fr> constant{value};

  TestHelpers::MathFunctions::check(std::move(constant), "constant",
                                    used_for_size, {{{-1.0, 1.0}}}, value);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.MathFunctions.Constant",
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
      test_constant_random<VolumeDim::value, DataVector, Fr>(dv);
      test_constant_random<VolumeDim::value, double, Fr>(
          std::numeric_limits<double>::signaling_NaN());
    });
  });
}

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.MathFunctions.Constant.Factory",
                  "[PointwiseFunctions][Unit]") {
  TestHelpers::test_factory_creation<MathFunction<1, Frame::Inertial>>(
      "Constant:\n"
      "  Value: -4.0\n");

  const double value{4.0};
  const MathFunctions::Constant<3, Frame::Inertial> constant{value};
  const auto created_constant =
      TestHelpers::test_creation<MathFunctions::Constant<3, Frame::Inertial>>(
          "Value: 4.0\n");
  CHECK(created_constant == constant);
  const auto created_constant_mathfunction =
      TestHelpers::test_factory_creation<MathFunction<3, Frame::Inertial>>(
          "Constant:\n"
          "  Value: 4.444\n");
}
