// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/TimeDependentOptions/GridCenters.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Framework/TestCreation.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace domain::creators::time_dependent_options {

SPECTRE_TEST_CASE("Unit.Domain.Creators.TimeDependentOptions.GridCenters",
                  "[Domain][Unit]") {
  domain::FunctionsOfTime::register_derived_with_charm();

  {
    const auto grid_centers_options =
        TestHelpers::test_option_tag<GridCentersOptions>(
            "SpecEvolutionParametersPerlFile: " + unit_test_src_path() +
            "../InputFiles/GrMhd/GhValenciaDivClean/EvolutionParameters.perl\n"
            "ScaleInspiralRateBy: Auto\n");
    REQUIRE(grid_centers_options.has_value());
    CHECK(grid_centers_options->initial_values ==
          std::array{DataVector{16.1996, 3.59764e-5, 0.0, -16.2, 0.0, 0.0},
                     DataVector{-0.00008095, 0.0, 0.0, 0.00008095, 0.0, 0.0},
                     DataVector{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}});
    CHECK(not grid_centers_options->scale_inspiral_rate_by.has_value());
  }

  {
    const auto grid_centers_options =
        TestHelpers::test_option_tag<GridCentersOptions>(
            "SpecEvolutionParametersPerlFile: " + unit_test_src_path() +
            "../InputFiles/GrMhd/GhValenciaDivClean/EvolutionParameters.perl\n"
            "ScaleInspiralRateBy: 0.9\n");
    REQUIRE(grid_centers_options.has_value());
    CHECK(grid_centers_options->initial_values ==
          std::array{DataVector{16.1996, 3.59764e-5, 0.0, -16.2, 0.0, 0.0},
                     DataVector{-0.00008095 * 0.9, 0.0, 0.0, 0.00008095 * 0.9,
                                0.0, 0.0},
                     DataVector{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}});
    CHECK(grid_centers_options->scale_inspiral_rate_by.value() == 0.9);
  }
}
}  // namespace domain::creators::time_dependent_options
