// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>

#include "ControlSystem/DataVectorHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"

SPECTRE_TEST_CASE("Unit.ControlSystem.DataVectorHelpers",
                  "[ControlSystem][Unit]") {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<double> dist{-10.0, 10.0};
  const auto arr = make_with_random_values<std::array<double, 3>>(
      make_not_null(&generator), dist, std::array<double, 3>{});

  DataVector dv{arr.size(), 0.0};
  for (size_t i = 0; i < arr.size(); i++) {
    dv[i] = gsl::at(arr, i);
  }

  CHECK(dv == array_to_datavector(arr));
}
