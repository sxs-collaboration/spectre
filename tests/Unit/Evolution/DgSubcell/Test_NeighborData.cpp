// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/NeighborData.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell {
namespace {
void test() noexcept {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<double> dist{-100.0, 100.0};
  std::vector<double> slice_data(10);
  std::vector<double> max_vars(10);
  std::vector<double> min_vars(10);
  fill_with_random_values(make_not_null(&slice_data), make_not_null(&gen),
                          make_not_null(&dist));
  fill_with_random_values(make_not_null(&max_vars), make_not_null(&gen),
                          make_not_null(&dist));
  fill_with_random_values(make_not_null(&min_vars), make_not_null(&gen),
                          make_not_null(&dist));

  const NeighborData nhbr_data{slice_data, max_vars, min_vars};

  // test equivalence
  {
    // clang-tidy: intentional copy
    const auto nhbr_data2 = nhbr_data;  // NOLINT
    CHECK(nhbr_data == nhbr_data2);
    CHECK_FALSE(nhbr_data != nhbr_data2);
  }
  {
    auto nhbr_data2 = nhbr_data;
    nhbr_data2.data_for_reconstruction[0] = 200.0;
    CHECK_FALSE(nhbr_data2 == nhbr_data);
    CHECK(nhbr_data2 != nhbr_data);
  }
  {
    auto nhbr_data2 = nhbr_data;
    nhbr_data2.max_variables_values[0] = 200.0;
    CHECK_FALSE(nhbr_data2 == nhbr_data);
    CHECK(nhbr_data2 != nhbr_data);
  }
  {
    auto nhbr_data2 = nhbr_data;
    nhbr_data2.min_variables_values[0] = 200.0;
    CHECK_FALSE(nhbr_data2 == nhbr_data);
    CHECK(nhbr_data2 != nhbr_data);
  }

  // Test stream operator
  CHECK(get_output(nhbr_data) ==
        get_output(nhbr_data.data_for_reconstruction) + '\n' +
            get_output(nhbr_data.max_variables_values) + '\n' +
            get_output(nhbr_data.min_variables_values));

  // Test serialization
  test_serialization(nhbr_data);
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.NeighborData", "[Evolution][Unit]") {
  test();
}
}  // namespace
}  // namespace evolution::dg::subcell
