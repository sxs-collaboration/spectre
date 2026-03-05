// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <sstream>
#include <string>
#include <variant>

#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/TimeDependentOptions/FromVolumeFile.hpp"
#include "Domain/Creators/TimeDependentOptions/RotationMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/SettleToConstantQuaternion.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Domain/Creators/TimeDependent/TestHelpers.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace domain::creators::time_dependent_options {
namespace {
constexpr double infinity = std::numeric_limits<double>::infinity();
void test_rotation_map_options() {
  {
    INFO("None");
    const auto rotation_map_options =
        TestHelpers::test_option_tag<RotationMapOptions<false>>("None");

    CHECK(not rotation_map_options.has_value());
  }
  const auto non_settle_fots = []<bool AllowSettleFoTs>() {
    INFO("Hardcoded AllowSettleFots = " + std::to_string(AllowSettleFoTs) +
         ", Non-Settle Fots");
    const auto rotation_map_options =
        TestHelpers::test_option_tag<RotationMapOptions<AllowSettleFoTs>>(
            "InitialAngularVelocity: [0.0, 0.4, 0.1]\n");

    REQUIRE(rotation_map_options.has_value());
    CHECK(std::holds_alternative<RotationMapOptions<AllowSettleFoTs>>(
        rotation_map_options.value()));

    const auto& hardcoded_options =
        std::get<RotationMapOptions<AllowSettleFoTs>>(
            rotation_map_options.value());

    CHECK(hardcoded_options.quaternions ==
          std::array{DataVector{1.0, 0.0, 0.0, 0.0}, DataVector{4, 0.0},
                     DataVector{4, 0.0}});
    CHECK(hardcoded_options.angles ==
          std::array{DataVector{3, 0.0}, DataVector{0.0, 0.4, 0.1},
                     DataVector{3, 0.0}, DataVector{3, 0.0}});
    CHECK_FALSE(hardcoded_options.decay_timescale.has_value());

    const auto rotation_ptr =
        get_rotation(rotation_map_options.value(), 0.3, 2.9);

    const auto* rotation =
        dynamic_cast<domain::FunctionsOfTime::QuaternionFunctionOfTime<3>*>(
            rotation_ptr.get());

    REQUIRE(rotation != nullptr);

    CHECK(rotation->time_bounds() == std::array{0.3, 2.9});
    CHECK(rotation->func(0.3)[0] == DataVector{1.0, 0.0, 0.0, 0.0});
  };

  non_settle_fots.template operator()<false>();
  non_settle_fots.template operator()<true>();

  {
    INFO("Hardcoded AllowSettleFots = true, Settle Fots");
    const auto rotation_map_options =
        TestHelpers::test_option_tag<RotationMapOptions<true>>(
            "InitialQuaternions: [[1.0, 0.0, 0.0, 0.0]]\n"
            "DecayTimescale: 40\n");

    REQUIRE(rotation_map_options.has_value());
    CHECK(std::holds_alternative<RotationMapOptions<true>>(
        rotation_map_options.value()));

    const auto& hardcoded_options =
        std::get<RotationMapOptions<true>>(rotation_map_options.value());

    const std::array expected_values{DataVector{1.0}, DataVector{2.0},
                                     DataVector{3.0}};
    CHECK(hardcoded_options.quaternions ==
          std::array{DataVector{1.0, 0.0, 0.0, 0.0}, DataVector{4, 0.0},
                     DataVector{4, 0.0}});
    CHECK(hardcoded_options.angles == make_array<4>(DataVector{3, 0.0}));
    CHECK(hardcoded_options.decay_timescale == std::optional{40.0});

    const auto rotation_ptr =
        get_rotation(rotation_map_options.value(), 0.3, 2.9);

    const auto* rotation =
        dynamic_cast<domain::FunctionsOfTime::SettleToConstantQuaternion*>(
            rotation_ptr.get());

    REQUIRE(rotation != nullptr);

    CHECK(rotation->time_bounds() == std::array{0.3, infinity});
    CHECK(rotation->func(0.3)[0] == DataVector{1.0, 0.0, 0.0, 0.0});
  }

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time["Rotation"] =
      std::make_unique<domain::FunctionsOfTime::QuaternionFunctionOfTime<3>>(
          0.0, std::array{DataVector{1.0, 0.0, 0.0, 0.0}},
          std::array{DataVector{3, 0.1}, DataVector{3, 0.2}, DataVector{3, 0.3},
                     DataVector{3, 0.4}},
          100.0);

  const std::string filename{"GoatCheese.h5"};
  const std::string subfile_name{"VolumeData"};

  {
    INFO("FromVolumeFile non-Settle FoTs");

    if (file_system::check_if_file_exists(filename)) {
      file_system::rm(filename, true);
    }

    TestHelpers::domain::creators::write_volume_data(filename, subfile_name,
                                                     0.3, functions_of_time);

    const auto rotation_map_options =
        TestHelpers::test_option_tag<RotationMapOptions<false>>(
            "H5Filename: GoatCheese.h5\n"
            "SubfileName: VolumeData");

    REQUIRE(rotation_map_options.has_value());
    CHECK(std::holds_alternative<FromVolumeFile>(rotation_map_options.value()));

    const auto rotation_ptr =
        get_rotation(rotation_map_options.value(), 0.3, 100.0);

    const auto* rotation =
        dynamic_cast<domain::FunctionsOfTime::QuaternionFunctionOfTime<3>*>(
            rotation_ptr.get());

    REQUIRE(rotation != nullptr);

    CHECK(rotation->time_bounds() == std::array{0.3, 100.0});
    CHECK(rotation->func_and_2_derivs(0.3) ==
          functions_of_time.at("Rotation")->func_and_2_derivs(0.3));
  }
  {
    INFO("FromVolumeFile Settle FoTs");

    if (file_system::check_if_file_exists(filename)) {
      file_system::rm(filename, true);
    }

    functions_of_time["Rotation"] =
        std::make_unique<domain::FunctionsOfTime::SettleToConstantQuaternion>(
            std::array{DataVector{1.0, 0.0, 0.0, 0.0}, DataVector{4, 2.0},
                       DataVector{4, 3.0}},
            0.0, 100.0);

    TestHelpers::domain::creators::write_volume_data(filename, subfile_name,
                                                     0.3, functions_of_time);

    const auto rotation_map_options =
        TestHelpers::test_option_tag<RotationMapOptions<true>>(
            "H5Filename: GoatCheese.h5\n"
            "SubfileName: VolumeData");

    REQUIRE(rotation_map_options.has_value());
    CHECK(std::holds_alternative<FromVolumeFile>(rotation_map_options.value()));

    const auto rotation_ptr =
        get_rotation(rotation_map_options.value(), 0.3, 100.0);

    const auto* rotation =
        dynamic_cast<domain::FunctionsOfTime::SettleToConstantQuaternion*>(
            rotation_ptr.get());

    REQUIRE(rotation != nullptr);

    CHECK(rotation->time_bounds() == std::array{0.0, infinity});
    CHECK(rotation->func_and_2_derivs(0.3) ==
          functions_of_time.at("Rotation")->func_and_2_derivs(0.3));
  }

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.TimeDependentOptions.RotationMap",
                  "[Domain][Unit]") {
  domain::FunctionsOfTime::register_derived_with_charm();
  test_rotation_map_options();
}
}  // namespace domain::creators::time_dependent_options
