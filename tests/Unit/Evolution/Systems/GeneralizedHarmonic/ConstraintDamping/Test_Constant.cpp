// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Constant.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/DampingFunction.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace {
template <size_t VolumeDim, typename DataType, typename Fr>
void test_constant_random(const DataType& used_for_size) {
  register_derived_classes_with_charm<
      gh::ConstraintDamping::Constant<VolumeDim, Fr>>();

  // Generate the amplitude and width
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> real_dis(-1, 1);
  std::uniform_real_distribution<> positive_dis(0, 1);

  const double value = real_dis(gen);

  gh::ConstraintDamping::Constant<VolumeDim, Fr> val{value};

  TestHelpers::gh::ConstraintDamping::check(std::move(val), "constant",
                                            used_for_size, {{{-1.0, 1.0}}},
                                            {"IgnoredFunctionOfTime"}, value);

  std::unique_ptr<gh::ConstraintDamping::Constant<VolumeDim, Fr>>
      val_unique_ptr =
          std::make_unique<gh::ConstraintDamping::Constant<VolumeDim, Fr>>(
              value);

  TestHelpers::gh::ConstraintDamping::check(
      std::move(val_unique_ptr->get_clone()), "constant", used_for_size,
      {{{-1.0, 1.0}}}, {"IgnoredFunctionOfTime"}, value);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.ConstraintDamp.Const",
    "[PointwiseFunctions][Unit]") {
  const DataVector dv{5};

  pypp::SetupLocalPythonEnvironment{
      "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Python"};

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

  TestHelpers::test_creation<std::unique_ptr<
      gh::ConstraintDamping::DampingFunction<1, Frame::Inertial>>>(
      "Constant:\n"
      "  Value: 4.0\n");

  const double value_3d{5.0};
  const gh::ConstraintDamping::Constant<3, Frame::Inertial> val_3d{value_3d};
  const auto created_val = TestHelpers::test_creation<
      gh::ConstraintDamping::Constant<3, Frame::Inertial>>("Value: 5.0\n");
  CHECK(created_val == val_3d);
  const auto created_gh_damping_function =
      TestHelpers::test_creation<std::unique_ptr<
          gh::ConstraintDamping::DampingFunction<3, Frame::Inertial>>>(
          "Constant:\n"
          "  Value: 5.0\n");

  test_serialization(val_3d);
}
