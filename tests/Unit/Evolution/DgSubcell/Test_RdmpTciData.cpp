// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell {
namespace {
void test() {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<double> dist{-100.0, 100.0};
  DataVector max_vars{10};
  DataVector min_vars{10};
  fill_with_random_values(make_not_null(&max_vars), make_not_null(&gen),
                          make_not_null(&dist));
  fill_with_random_values(make_not_null(&min_vars), make_not_null(&gen),
                          make_not_null(&dist));

  const RdmpTciData rdmp_tci_data{max_vars, min_vars};

  // test equivalence
  {
    // clang-tidy: intentional copy
    const auto rdmp_tci_data2 = rdmp_tci_data;  // NOLINT
    CHECK(rdmp_tci_data == rdmp_tci_data2);
    CHECK_FALSE(rdmp_tci_data != rdmp_tci_data2);
  }
  {
    auto rdmp_tci_data2 = rdmp_tci_data;
    rdmp_tci_data2.max_variables_values[0] = 200.0;
    CHECK_FALSE(rdmp_tci_data2 == rdmp_tci_data);
    CHECK(rdmp_tci_data2 != rdmp_tci_data);
  }
  {
    auto rdmp_tci_data2 = rdmp_tci_data;
    rdmp_tci_data2.min_variables_values[0] = 200.0;
    CHECK_FALSE(rdmp_tci_data2 == rdmp_tci_data);
    CHECK(rdmp_tci_data2 != rdmp_tci_data);
  }

  // Test stream operator
  CHECK(get_output(rdmp_tci_data) ==
        get_output(rdmp_tci_data.max_variables_values) + '\n' +
            get_output(rdmp_tci_data.min_variables_values));

  // Test serialization
  test_serialization(rdmp_tci_data);
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.RdmpTciData", "[Evolution][Unit]") {
  test();
}
}  // namespace
}  // namespace evolution::dg::subcell
