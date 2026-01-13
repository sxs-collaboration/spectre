// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <sstream>
#include <string>
#include <variant>

#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/TimeDependentOptions/FromVolumeFile.hpp"
#include "Domain/Creators/TimeDependentOptions/SkewMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Domain/Creators/TimeDependent/TestHelpers.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace domain::creators::time_dependent_options {
namespace {
void test_skew_map_options() {
  {
    INFO("None");
    const auto skew_map_options =
        TestHelpers::test_option_tag<SkewMapOptions>("None");

    CHECK(not skew_map_options.has_value());
  }
  {
    INFO("Hardcoded options");
    const auto skew_map_options = TestHelpers::test_option_tag<
        domain::creators::time_dependent_options::SkewMapOptions>(
        "InitialValuesY: [0.1, -0.2, 0.3]\n"
        "InitialValuesZ: [-0.4, 0.5, -0.6]\n");

    REQUIRE(skew_map_options.has_value());
    CHECK(std::holds_alternative<SkewMapOptions>(skew_map_options.value()));
    const auto& hard_coded_options =
        std::get<SkewMapOptions>(skew_map_options.value());
    CHECK(hard_coded_options.initial_angles_y == std::array{0.1, -0.2, 0.3});
    CHECK(hard_coded_options.initial_angles_z == std::array{-0.4, 0.5, -0.6});

    const auto skew_ptr = get_skew(skew_map_options.value(), 0.1, 65.8);

    const auto* skew =
        dynamic_cast<domain::FunctionsOfTime::PiecewisePolynomial<2>*>(
            skew_ptr.get());

    CHECK(skew != nullptr);

    CHECK(skew->time_bounds() == std::array{0.1, 65.8});
    CHECK(skew->func_and_2_derivs(0.1) == std::array{
                                              DataVector{0.1, -0.4},
                                              DataVector{-0.2, 0.5},
                                              DataVector{0.3, -0.6},
                                          });
  }
  {
    INFO("FromVolumeFile");
    std::unordered_map<std::string,
                       std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        functions_of_time{};
    functions_of_time["Skew"] =
        std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
            0.0,
            std::array{DataVector{2, 1.0}, DataVector{2, 2.0},
                       DataVector{2, 3.0}},
            100.0);
    const std::string filename{"Wildfires.h5"};
    const std::string subfile_name{"VolumeData"};
    if (file_system::check_if_file_exists(filename)) {
      file_system::rm(filename, true);
    }

    TestHelpers::domain::creators::write_volume_data(filename, subfile_name,
                                                     0.1, functions_of_time);

    const auto skew_map_options = TestHelpers::test_option_tag<
        domain::creators::time_dependent_options::SkewMapOptions>(
        "H5Filename: " + filename + "\nSubfileName: " + subfile_name);

    REQUIRE(skew_map_options.has_value());
    CHECK(std::holds_alternative<FromVolumeFile>(skew_map_options.value()));

    const auto skew_ptr = get_skew(skew_map_options.value(), 0.1, 65.8);

    const auto* skew =
        dynamic_cast<domain::FunctionsOfTime::PiecewisePolynomial<2>*>(
            skew_ptr.get());

    CHECK(skew != nullptr);

    CHECK(skew->time_bounds() == std::array{0.1, 65.8});
    CHECK_ITERABLE_APPROX(skew->func_and_2_derivs(0.3),
                          functions_of_time.at("Skew")->func_and_2_derivs(0.3));

    if (file_system::check_if_file_exists(filename)) {
      file_system::rm(filename, true);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.TimeDependentOptions.SkewMap",
                  "[Domain][Unit]") {
  domain::FunctionsOfTime::register_derived_with_charm();
  test_skew_map_options();
}
}  // namespace domain::creators::time_dependent_options
