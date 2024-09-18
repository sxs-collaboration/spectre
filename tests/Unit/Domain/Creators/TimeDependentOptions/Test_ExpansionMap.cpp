// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
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
#include "Framework/TestCreation.hpp"
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

namespace {
void test_expansion_map_options() {
  {
    const auto expansion_map_options = TestHelpers::test_creation<
        domain::creators::time_dependent_options::ExpansionMapOptions>(
        "InitialValues: [1.0, 2.0, 3.0]\n"
        "InitialValuesOuterBoundary: [4.0, 5.0, 6.0]\n"
        "DecayTimescaleOuterBoundary: 50\n"
        "DecayTimescale: Auto\n"
        "AsymptoticVelocityOuterBoundary: -1e-5");

    CHECK(expansion_map_options.name() == "ExpansionMap");
    CHECK(expansion_map_options.initial_values ==
          std::array{DataVector{1.0}, DataVector{2.0}, DataVector{3.0}});
    CHECK(expansion_map_options.initial_values_outer_boundary ==
          std::array{DataVector{4.0}, DataVector{5.0}, DataVector{6.0}});
    CHECK(expansion_map_options.decay_timescale_outer_boundary == 50.0);
    CHECK_FALSE(expansion_map_options.decay_timescale.has_value());
    CHECK(expansion_map_options.asymptotic_velocity_outer_boundary.has_value());
    CHECK(expansion_map_options.asymptotic_velocity_outer_boundary.value() ==
          -1e-5);
  }
  {
    const auto expansion_map_options = TestHelpers::test_creation<
        domain::creators::time_dependent_options::ExpansionMapOptions>(
        "InitialValues: [1.0, 2.0, 3.0]\n"
        "InitialValuesOuterBoundary: [4.0, 5.0, 6.0]\n"
        "DecayTimescaleOuterBoundary: 50\n"
        "DecayTimescale: 40\n"
        "AsymptoticVelocityOuterBoundary: Auto");

    CHECK(expansion_map_options.name() == "ExpansionMap");
    CHECK(expansion_map_options.initial_values ==
          std::array{DataVector{1.0}, DataVector{2.0}, DataVector{3.0}});
    CHECK(expansion_map_options.initial_values_outer_boundary ==
          std::array{DataVector{4.0}, DataVector{5.0}, DataVector{6.0}});
    CHECK(expansion_map_options.decay_timescale_outer_boundary == 50.0);
    CHECK(expansion_map_options.decay_timescale.has_value());
    CHECK(expansion_map_options.decay_timescale.value() == 40.0);
    CHECK_FALSE(
        expansion_map_options.asymptotic_velocity_outer_boundary.has_value());
  }

  CHECK_THROWS_WITH(
      (TestHelpers::test_creation<
          domain::creators::time_dependent_options::ExpansionMapOptions>(
          "InitialValues: [1.0, 2.0, 3.0]\n"
          "InitialValuesOuterBoundary: [4.0, 5.0, 6.0]\n"
          "DecayTimescaleOuterBoundary: 50\n"
          "DecayTimescale: Auto\n"
          "AsymptoticVelocityOuterBoundary: Auto")),
      Catch::Matchers::ContainsSubstring(
          "must specify one of DecayTimescale or "
          "AsymptoticVelocityOuterBoundary, but not both."));
  CHECK_THROWS_WITH(
      (TestHelpers::test_creation<
          domain::creators::time_dependent_options::ExpansionMapOptions>(
          "InitialValues: [1.0, 2.0, 3.0]\n"
          "InitialValuesOuterBoundary: [4.0, 5.0, 6.0]\n"
          "DecayTimescaleOuterBoundary: 50\n"
          "DecayTimescale: 40\n"
          "AsymptoticVelocityOuterBoundary: -1e-5")),
      Catch::Matchers::ContainsSubstring(
          "must specify one of DecayTimescale or "
          "AsymptoticVelocityOuterBoundary, but not both."));
  CHECK_THROWS_WITH(
      (TestHelpers::test_creation<
          domain::creators::time_dependent_options::ExpansionMapOptions>(
          "InitialValues: [1.0, 2.0, 3.0]\n"
          "InitialValuesOuterBoundary: [4.0, 5.0, 6.0]\n"
          "DecayTimescaleOuterBoundary: Auto\n"
          "DecayTimescale: 40\n"
          "AsymptoticVelocityOuterBoundary: Auto")),
      Catch::Matchers::ContainsSubstring(
          "When specifying the ExpansionMap initial outer "
          "boundary values directly, you must also specify a "
          "'DecayTimescaleOuterBoundary'."));

  {
    std::unordered_map<std::string,
                       std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        functions_of_time{};
    functions_of_time["Expansion"] =
        std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
            0.0, std::array{DataVector{1.0}, DataVector{2.0}, DataVector{3.0}},
            100.0);
    functions_of_time["ExpansionOuterBoundary"] =
        std::make_unique<domain::FunctionsOfTime::FixedSpeedCubic>(0.0, 0.0,
                                                                   -1e-5, 50.0);
    const std::string filename{"Commencement.h5"};
    const std::string subfile_name{"VolumeData"};
    if (file_system::check_if_file_exists(filename)) {
      file_system::rm(filename, true);
    }

    {
      h5::H5File<h5::AccessType::ReadWrite> h5_file{filename};
      auto& vol_file = h5_file.insert<h5::VolumeData>(subfile_name);

      // We don't care about the volume data here, just the functions of time
      vol_file.write_volume_data(
          0, 0.0,
          {ElementVolumeData{
              "blah",
              {TensorComponent{"RandomTensor", DataVector{3, 0.0}}},
              {3},
              {Spectral::Basis::Legendre},
              {Spectral::Quadrature::GaussLobatto}}},
          std::nullopt, serialize(functions_of_time));
    }

    {
      const auto expansion_map_options = TestHelpers::test_creation<
          domain::creators::time_dependent_options::ExpansionMapOptions>(
          "DecayTimescaleOuterBoundary: 60\n"
          "DecayTimescale: Auto\n"
          "AsymptoticVelocityOuterBoundary: -2e-5\n"
          "InitialValues:\n"
          "  H5Filename: " +
          filename + "\n  SubfileName: " + subfile_name +
          "\n  Time: 0.0\n"
          "InitialValuesOuterBoundary:\n"
          "  H5Filename: " +
          filename + "\n  SubfileName: " + subfile_name + "\n  Time: 0.0");
      CHECK(expansion_map_options.name() == "ExpansionMap");
      CHECK(expansion_map_options.initial_values ==
            std::array{DataVector{1.0}, DataVector{2.0}, DataVector{3.0}});
      CHECK(expansion_map_options.initial_values_outer_boundary ==
            std::array{DataVector{0.0}, DataVector{0.0}, DataVector{0.0}});
      CHECK(expansion_map_options.decay_timescale_outer_boundary == 60.0);
      CHECK_FALSE(expansion_map_options.decay_timescale.has_value());
      CHECK(
          expansion_map_options.asymptotic_velocity_outer_boundary.has_value());
      CHECK(expansion_map_options.asymptotic_velocity_outer_boundary.value() ==
            -2e-5);
    }
    {
      const auto expansion_map_options = TestHelpers::test_creation<
          domain::creators::time_dependent_options::ExpansionMapOptions>(
          "DecayTimescaleOuterBoundary: Auto\n"
          "DecayTimescale: Auto\n"
          "AsymptoticVelocityOuterBoundary: Auto\n"
          "InitialValues:\n"
          "  H5Filename: " +
          filename + "\n  SubfileName: " + subfile_name +
          "\n  Time: 0.0\n"
          "InitialValuesOuterBoundary:\n"
          "  H5Filename: " +
          filename + "\n  SubfileName: " + subfile_name + "\n  Time: 0.0");
      CHECK(expansion_map_options.name() == "ExpansionMap");
      CHECK(expansion_map_options.initial_values ==
            std::array{DataVector{1.0}, DataVector{2.0}, DataVector{3.0}});
      CHECK(expansion_map_options.initial_values_outer_boundary ==
            std::array{DataVector{0.0}, DataVector{0.0}, DataVector{0.0}});
      CHECK(expansion_map_options.decay_timescale_outer_boundary == 50.0);
      CHECK_FALSE(expansion_map_options.decay_timescale.has_value());
      CHECK(
          expansion_map_options.asymptotic_velocity_outer_boundary.has_value());
      CHECK(expansion_map_options.asymptotic_velocity_outer_boundary.value() ==
            -1e-5);
    }

    if (file_system::check_if_file_exists(filename)) {
      file_system::rm(filename, true);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.TimeDependentOptions.ExpansionMap",
                  "[Domain][Unit]") {
  domain::FunctionsOfTime::register_derived_with_charm();
  test_expansion_map_options();
}
