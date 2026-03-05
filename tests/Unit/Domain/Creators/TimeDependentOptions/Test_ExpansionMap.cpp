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
#include "Domain/Creators/TimeDependentOptions/ExpansionMap.hpp"
#include "Domain/Creators/TimeDependentOptions/FromVolumeFile.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/SettleToConstant.hpp"
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
void test_expansion_map_options() {
  {
    INFO("None");
    const auto expansion_map_options =
        TestHelpers::test_option_tag<ExpansionMapOptions<false>>("None");

    CHECK(not expansion_map_options.has_value());
  }

  const auto non_settle_fots = []<bool AllowSettleFoTs>() {
    INFO("Hardcoded AllowSettleFots = " + std::to_string(AllowSettleFoTs) +
         ", Non-Settle Fots");
    const auto expansion_map_options =
        TestHelpers::test_option_tag<ExpansionMapOptions<AllowSettleFoTs>>(
            "InitialValues: [1.0, 2.0, 3.0]\n"
            "DecayTimescaleOuterBoundary: 50\n"
            "AsymptoticVelocityOuterBoundary: -1e-5");

    REQUIRE(expansion_map_options.has_value());
    CHECK(std::holds_alternative<ExpansionMapOptions<AllowSettleFoTs>>(
        expansion_map_options.value()));

    const auto& hardcoded_options =
        std::get<ExpansionMapOptions<AllowSettleFoTs>>(
            expansion_map_options.value());

    const std::array expected_values{DataVector{1.0}, DataVector{2.0},
                                     DataVector{3.0}};
    CHECK(hardcoded_options.initial_values == expected_values);
    CHECK(hardcoded_options.initial_values_outer_boundary ==
          std::array{DataVector{1.0}, DataVector{0.0}, DataVector{0.0}});
    CHECK(hardcoded_options.decay_timescale_outer_boundary == 50.0);
    CHECK_FALSE(hardcoded_options.decay_timescale.has_value());
    CHECK(hardcoded_options.asymptotic_velocity_outer_boundary ==
          std::optional{-1.e-5});

    const auto expansion_fots =
        get_expansion(expansion_map_options.value(), 0.3, 2.9);
    const auto& expansion_ptr = expansion_fots.at("Expansion");
    const auto& expansion_outer_boundary_ptr =
        expansion_fots.at("ExpansionOuterBoundary");

    const auto* expansion =
        dynamic_cast<domain::FunctionsOfTime::PiecewisePolynomial<2>*>(
            expansion_ptr.get());
    const auto* expansion_outer_boundary =
        dynamic_cast<domain::FunctionsOfTime::FixedSpeedCubic*>(
            expansion_outer_boundary_ptr.get());

    REQUIRE(expansion != nullptr);
    REQUIRE(expansion_outer_boundary != nullptr);

    CHECK(expansion->time_bounds() == std::array{0.3, 2.9});
    CHECK(expansion->func_and_2_derivs(0.3) == expected_values);
    CHECK(expansion_outer_boundary->time_bounds() == std::array{0.3, infinity});
    CHECK(expansion_outer_boundary->decay_timescale() == 50.0);
    CHECK(expansion_outer_boundary->velocity() == -1e-5);
  };

  non_settle_fots.template operator()<false>();
  non_settle_fots.template operator()<true>();

  {
    INFO("Hardcoded AllowSettleFots = true, Settle Fots");
    const auto expansion_map_options =
        TestHelpers::test_option_tag<ExpansionMapOptions<true>>(
            "InitialValues: [1.0, 2.0, 3.0]\n"
            "InitialValuesOuterBoundary: [4.0, 5.0, 6.0]\n"
            "DecayTimescaleOuterBoundary: 50\n"
            "DecayTimescale: 40\n");

    REQUIRE(expansion_map_options.has_value());
    CHECK(std::holds_alternative<ExpansionMapOptions<true>>(
        expansion_map_options.value()));

    const auto& hardcoded_options =
        std::get<ExpansionMapOptions<true>>(expansion_map_options.value());

    const std::array expected_values{DataVector{1.0}, DataVector{2.0},
                                     DataVector{3.0}};
    CHECK(hardcoded_options.initial_values == expected_values);
    CHECK(hardcoded_options.initial_values_outer_boundary ==
          std::array{DataVector{4.0}, DataVector{5.0}, DataVector{6.0}});
    CHECK(hardcoded_options.decay_timescale_outer_boundary == 50.0);
    CHECK(hardcoded_options.decay_timescale == std::optional{40.0});
    CHECK_FALSE(
        hardcoded_options.asymptotic_velocity_outer_boundary.has_value());

    const auto expansion_fots =
        get_expansion(expansion_map_options.value(), 0.3, 2.9);
    const auto& expansion_ptr = expansion_fots.at("Expansion");
    const auto& expansion_outer_boundary_ptr =
        expansion_fots.at("ExpansionOuterBoundary");

    const auto* expansion =
        dynamic_cast<domain::FunctionsOfTime::SettleToConstant*>(
            expansion_ptr.get());
    const auto* expansion_outer_boundary =
        dynamic_cast<domain::FunctionsOfTime::SettleToConstant*>(
            expansion_outer_boundary_ptr.get());

    REQUIRE(expansion != nullptr);
    REQUIRE(expansion_outer_boundary != nullptr);

    CHECK(expansion->time_bounds() == std::array{0.3, infinity});
    CHECK(expansion->func_and_2_derivs(0.3) == expected_values);
    CHECK(expansion_outer_boundary->time_bounds() == std::array{0.3, infinity});
  }

  const std::string filename{"Commencement.h5"};
  const std::string subfile_name{"VolumeData"};

  domain::FunctionsOfTimeMap functions_of_time{};
  functions_of_time["Expansion"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          0.0, std::array{DataVector{1.0}, DataVector{2.0}, DataVector{3.0}},
          100.0);
  functions_of_time["ExpansionOuterBoundary"] =
      std::make_unique<domain::FunctionsOfTime::FixedSpeedCubic>(0.0, 0.0,
                                                                 -1e-5, 50.0);

  {
    INFO("FromVolumeFile non-Settle FoTs");

    if (file_system::check_if_file_exists(filename)) {
      file_system::rm(filename, true);
    }

    TestHelpers::domain::creators::write_volume_data(filename, subfile_name,
                                                     0.3, functions_of_time);

    const auto expansion_map_options =
        TestHelpers::test_option_tag<ExpansionMapOptions<false>>(
            "H5Filename: Commencement.h5\n"
            "SubfileName: VolumeData");

    REQUIRE(expansion_map_options.has_value());
    CHECK(
        std::holds_alternative<FromVolumeFile>(expansion_map_options.value()));

    const auto expansion_fots =
        get_expansion(expansion_map_options.value(), 0.3, 100.0);
    const auto& expansion_ptr = expansion_fots.at("Expansion");
    const auto& expansion_outer_boundary_ptr =
        expansion_fots.at("ExpansionOuterBoundary");

    const auto* expansion =
        dynamic_cast<domain::FunctionsOfTime::PiecewisePolynomial<2>*>(
            expansion_ptr.get());
    const auto* expansion_outer_boundary =
        dynamic_cast<domain::FunctionsOfTime::FixedSpeedCubic*>(
            expansion_outer_boundary_ptr.get());

    REQUIRE(expansion != nullptr);
    REQUIRE(expansion_outer_boundary != nullptr);

    CHECK(expansion->time_bounds() == std::array{0.3, 100.0});
    CHECK(expansion->func_and_2_derivs(0.3) ==
          functions_of_time.at("Expansion")->func_and_2_derivs(0.3));
    CHECK(
        expansion_outer_boundary->func_and_2_derivs(0.3) ==
        functions_of_time.at("ExpansionOuterBoundary")->func_and_2_derivs(0.3));
    CHECK(expansion_outer_boundary->time_bounds() == std::array{0.0, infinity});
    CHECK(expansion_outer_boundary->decay_timescale() == 50.0);
    CHECK(expansion_outer_boundary->velocity() == -1e-5);
  }
  {
    INFO("FromVolumeFile Settle FoTs");

    if (file_system::check_if_file_exists(filename)) {
      file_system::rm(filename, true);
    }

    functions_of_time["Expansion"] =
        std::make_unique<domain::FunctionsOfTime::SettleToConstant>(
            std::array{DataVector{1.0}, DataVector{2.0}, DataVector{3.0}}, 0.0,
            100.0);
    functions_of_time["ExpansionOuterBoundary"] =
        std::make_unique<domain::FunctionsOfTime::SettleToConstant>(
            std::array{DataVector{3.0}, DataVector{4.0}, DataVector{6.0}}, 0.0,
            100.0);

    TestHelpers::domain::creators::write_volume_data(filename, subfile_name,
                                                     0.3, functions_of_time);

    const auto expansion_map_options =
        TestHelpers::test_option_tag<ExpansionMapOptions<true>>(
            "H5Filename: Commencement.h5\n"
            "SubfileName: VolumeData");

    REQUIRE(expansion_map_options.has_value());
    CHECK(
        std::holds_alternative<FromVolumeFile>(expansion_map_options.value()));

    const auto expansion_fots =
        get_expansion(expansion_map_options.value(), 0.3, 100.0);
    const auto& expansion_ptr = expansion_fots.at("Expansion");
    const auto& expansion_outer_boundary_ptr =
        expansion_fots.at("ExpansionOuterBoundary");

    const auto* expansion =
        dynamic_cast<domain::FunctionsOfTime::SettleToConstant*>(
            expansion_ptr.get());
    const auto* expansion_outer_boundary =
        dynamic_cast<domain::FunctionsOfTime::SettleToConstant*>(
            expansion_outer_boundary_ptr.get());

    REQUIRE(expansion != nullptr);
    REQUIRE(expansion_outer_boundary != nullptr);

    CHECK(expansion->time_bounds() == std::array{0.0, infinity});
    CHECK(expansion->func_and_2_derivs(0.3) ==
          functions_of_time.at("Expansion")->func_and_2_derivs(0.3));
    CHECK(
        expansion_outer_boundary->func_and_2_derivs(0.3) ==
        functions_of_time.at("ExpansionOuterBoundary")->func_and_2_derivs(0.3));
    CHECK(expansion_outer_boundary->time_bounds() == std::array{0.0, infinity});
  }

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.TimeDependentOptions.ExpansionMap",
                  "[Domain][Unit]") {
  domain::FunctionsOfTime::register_derived_with_charm();
  test_expansion_map_options();
}
}  // namespace domain::creators::time_dependent_options
