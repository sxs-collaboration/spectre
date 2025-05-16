// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/DampingFunction.hpp"
#include "Evolution/Systems/ScalarTensor/ConstraintDamping/DampingFunction.hpp"
#include "Evolution/Systems/ScalarTensor/ConstraintDamping/TimeDependentTripleGaussian.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/Serialization/PupStlCpp11.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace TestHelpers::ScalarTensor::ConstraintDamping {
namespace detail {
template <size_t VolumeDim, typename Fr, class... MemberArgs, class T>
void check_impl(
    const std::unique_ptr<::ScalarTensor::ConstraintDamping::DampingFunction<
        VolumeDim, Fr>>& in_gh_damping_function,
    const std::string& python_function_prefix, const T& used_for_size,
    const std::array<std::pair<double, double>, 1> random_value_bounds,
    const std::vector<std::string>& function_of_time_names,
    const MemberArgs&... member_args) {
  using GhDampingFunc =
      ::ScalarTensor::ConstraintDamping::DampingFunction<VolumeDim, Fr>;

  const auto member_args_tuple = std::make_tuple(member_args...);
  const auto helper = [&python_function_prefix, &random_value_bounds,
                       &member_args_tuple, &function_of_time_names,
                       &used_for_size](const std::unique_ptr<GhDampingFunc>&
                                           gh_damping_function) {
    INFO("Testing call operator...");
    // Make a lambda that calls the damping function's call operator
    // with a hard-coded FunctionsOfTime, since check_with_random_values
    // cannot convert a FunctionsOfTime into a python type.
    // The FunctionsOfTime contains a single FunctionOfTime
    // \f$f(t) = a_0 + a_1 (t-t_0) + a_2 (t-t_0)^2 + a_3 (t-t_0)^3\f$, where
    // \f$a_0 = 1.0\f$, \f$a_1 = 0.2\f$, \f$a_2 = 0.03,\f$,
    // \f$a_3 = 0.004\f$, and \f$t_0\f$ is the smallest possible value
    // of the randomly selected time.
    //
    // The corresponding python function should use
    // the same hard-coded coefficients to evaluate \f$f(t)\f$ as well
    // as the same value of \f$t_0\f$.
    // However, here the PiecewisePolynomial must be initialized not
    // with the polynomial coefficients but with the values of \f$f(t)\f$
    // and its derivatives evaluated at \f$t=t_0\f$: these are,
    // respectively, \f$a_0,a_1,2 a_2,6 a_3\f$.
    //
    // Finally, note that the FunctionOfTime never expires.
    const auto damping_function_call_operator_helper =
        [&gh_damping_function, &random_value_bounds, &function_of_time_names](
            const tnsr::I<T, VolumeDim, Fr>& coordinates, const double time) {
          std::unordered_map<
              std::string,
              std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>
              functions_of_time{};
          for (const auto& function_of_time_name : function_of_time_names) {
            if (function_of_time_name == "GridCenters") {
              functions_of_time[function_of_time_name] = std::make_unique<
                  ::domain::FunctionsOfTime::PiecewisePolynomial<3>>(
                  std::min(gsl::at(random_value_bounds, 0).first,
                           gsl::at(random_value_bounds, 0).second),
                  std::array<DataVector, 4>{
                      {{16.0, 0.0, 0.0, -16.0, 0.0, 0.0},
                       {-0.001, 0.0, 0.0, 0.002, 0.0, 0.0},
                       {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                       {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}},
                  std::numeric_limits<double>::max());
            } else {
              // The randomly selected time will be between the
              // random_value_bounds, so set the earliest time of the
              // function_of_times to the lower bound in
              // random_value_bounds.
              functions_of_time[function_of_time_name] = std::make_unique<
                  ::domain::FunctionsOfTime::PiecewisePolynomial<3>>(
                  std::min(gsl::at(random_value_bounds, 0).first,
                           gsl::at(random_value_bounds, 0).second),
                  std::array<DataVector, 4>{{{1.0}, {0.2}, {0.06}, {0.024}}},
                  std::numeric_limits<double>::max());
            }
          }
          // Default-construct the scalar, to test that the damping
          // function's call operator correctly resizes it
          // (in the case T is a DataVector)
          // with set_number_of_grid_points()
          Scalar<T> value_at_coordinates{};
          gh_damping_function->operator()(make_not_null(&value_at_coordinates),
                                          coordinates, time, functions_of_time);
          return value_at_coordinates;
        };

    pypp::check_with_random_values<1>(
        &decltype(damping_function_call_operator_helper)::operator(),
        damping_function_call_operator_helper, "TestFunctions",
        python_function_prefix + "_call_operator", random_value_bounds,
        member_args_tuple, used_for_size);
    INFO("Done testing call operator...");
    INFO("Done\n\n");
  };

  helper(in_gh_damping_function);
  helper(serialize_and_deserialize(in_gh_damping_function));
}
}  // namespace detail

// \ingroup TestingFrameworkGroup
// \brief Test a DampingFunction by comparing to python functions

// The python functions must be added to TestFunctions.py in
// tests/Unit/Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Python.
// Each python function for a corresponding DampingFunction should begin
// with a prefix `python_function_prefix`. The prefix for each class of
// DampingFunction is arbitrary, but should generally be descriptive (e.g.
// 'gaussian_plus_constant') of the DampingFunction.

// The input parameter `function_of_time_name` is the name of the FunctionOfTime
// that will be included in the FunctionsOfTime passed to the DampingFunction's
// call operator. For time-dependent DampingFunctions, this parameter must be
// consistent with the FunctionOfTime name that the call operator of
// `in_gh_damping_function` expects. For time-independent DampingFunctions,
// `function_of_time_name` will be ignored.

// If a DampingFunction class has member variables set by its constructor, then
// these member variables must be passed in as the last arguments to the `check`
// function`. Each python function must take these same arguments as the
// trailing arguments.
template <class DampingFunctionType, class T, class... MemberArgs>
void check(std::unique_ptr<DampingFunctionType> in_gh_damping_function,
           const std::string& python_function_prefix, const T& used_for_size,
           const std::array<std::pair<double, double>, 1>& random_value_bounds,
           const std::vector<std::string>& function_of_time_names,
           const MemberArgs&... member_args) {
  detail::check_impl(
      std::unique_ptr<::ScalarTensor::ConstraintDamping::DampingFunction<
          DampingFunctionType::volume_dim,
          typename DampingFunctionType::frame>>(
          std::move(in_gh_damping_function)),
      python_function_prefix, used_for_size, random_value_bounds,
      function_of_time_names, member_args...);
}

template <class DampingFunctionType, class T, class... MemberArgs>
void check(DampingFunctionType in_gh_damping_function,
           const std::string& python_function_prefix, const T& used_for_size,
           const std::array<std::pair<double, double>, 1>& random_value_bounds,
           const std::vector<std::string>& function_of_time_names,
           const MemberArgs&... member_args) {
  detail::check_impl(
      std::unique_ptr<::ScalarTensor::ConstraintDamping::DampingFunction<
          DampingFunctionType::volume_dim,
          typename DampingFunctionType::frame>>(
          std::make_unique<DampingFunctionType>(
              std::move(in_gh_damping_function))),
      python_function_prefix, used_for_size, random_value_bounds,
      function_of_time_names, member_args...);
}

}  // namespace TestHelpers::ScalarTensor::ConstraintDamping

namespace {
template <typename DataType>
void test_triple_gaussian_random(const DataType& used_for_size) {
  register_derived_classes_with_charm<
      ScalarTensor::ConstraintDamping::TimeDependentTripleGaussian>();

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

  const ScalarTensor::ConstraintDamping::TimeDependentTripleGaussian
      triple_gauss{constant,    amplitude_1, width_1,          center_1,
                   amplitude_2, width_2,     center_2,         amplitude_3,
                   width_3,     center_3,    "ExpansionFactor"};

  TestHelpers::ScalarTensor::ConstraintDamping::check(
      std::move(triple_gauss), "time_dependent_triple_gaussian", used_for_size,
      {{{-1.0, 1.0}}}, {function_of_time_for_scaling}, constant, amplitude_1,
      width_1, center_1, amplitude_2, width_2, center_2, amplitude_3, width_3,
      center_3);

  const std::unique_ptr<
      ScalarTensor::ConstraintDamping::TimeDependentTripleGaussian>
      triple_gauss_unique_ptr = std::make_unique<
          ScalarTensor::ConstraintDamping::TimeDependentTripleGaussian>(
          constant, amplitude_1, width_1, center_1, amplitude_2, width_2,
          center_2, amplitude_3, width_3, center_3, "ExpansionFactor");

  TestHelpers::ScalarTensor::ConstraintDamping::check(
      triple_gauss_unique_ptr->get_clone(), "time_dependent_triple_gaussian",
      used_for_size, {{{-1.0, 1.0}}}, {function_of_time_for_scaling}, constant,
      amplitude_1, width_1, center_1, amplitude_2, width_2, center_2,
      amplitude_3, width_3, center_3);

  const std::unique_ptr<
      ScalarTensor::ConstraintDamping::TimeDependentTripleGaussian>
      triple_gauss_object_centers_unique_ptr = std::make_unique<
          ScalarTensor::ConstraintDamping::TimeDependentTripleGaussian>(
          constant, amplitude_1, width_1, std::nullopt, amplitude_2, width_2,
          std::nullopt, amplitude_3, width_3, center_3, "ObjectCenters");

  REQUIRE(
      dynamic_cast<
          const ScalarTensor::ConstraintDamping::TimeDependentTripleGaussian&>(
          *triple_gauss_object_centers_unique_ptr->get_clone()) ==
      *triple_gauss_object_centers_unique_ptr);
  TestHelpers::ScalarTensor::ConstraintDamping::check(
      triple_gauss_object_centers_unique_ptr->get_clone(),
      "time_dependent_triple_gaussian_object_centers", used_for_size,
      {{{-1.0, 1.0}}}, {"GridCenters"}, constant, amplitude_1, width_1,
      center_1, amplitude_2, width_2, center_2, amplitude_3, width_3, center_3);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ScalarTensor.ConstraintDamp.TimeDep3Gauss",
    "[PointwiseFunctions][Unit]") {
  const DataVector dv{5};

  pypp::SetupLocalPythonEnvironment{
      "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Python"};

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

  const ScalarTensor::ConstraintDamping::TimeDependentTripleGaussian
      triple_gauss_3d{constant_3d, amplitude_1_3d,   width_1_3d,
                      center_1_3d, amplitude_2_3d,   width_2_3d,
                      center_2_3d, amplitude_3_3d,   width_3_3d,
                      center_3_3d, "ExpansionFactor"};
  const auto created_triple_gauss = TestHelpers::test_creation<
      ScalarTensor::ConstraintDamping::TimeDependentTripleGaussian>(
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
      TestHelpers::test_creation<std::unique_ptr<
          ScalarTensor::ConstraintDamping::DampingFunction<3, Frame::Grid>>>(
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
