// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <string>
#include <unordered_map>

#include "Domain/FunctionsOfTime/CombineFunctionsOfTime.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Utilities/StdHelpers.hpp"

namespace {
using FoT = domain::FunctionsOfTime::FunctionOfTime;
using PP = domain::FunctionsOfTime::PiecewisePolynomial<0>;
using FoTMap = std::unordered_map<std::string, std::unique_ptr<FoT>>;

SPECTRE_TEST_CASE("Unit.Domain.FunctionsOfTime.CombineFunctionsOfTime",
                  "[Domain][Unit]") {
  const double initial_time = 0.0;
  const std::array<DataVector, 1> init_func{{{0.5}}};
  const std::string name_base{"FoT"};
  const double expiration_time = 1.0;
  PP pp0{initial_time, init_func, expiration_time};
  PP pp1{initial_time, init_func, expiration_time};
  PP pp2{initial_time, init_func, expiration_time};

  FoTMap map0{};
  map0[name_base + "0"] = std::make_unique<PP>(pp0);
  FoTMap map1{};
  map1[name_base + "1"] = std::make_unique<PP>(pp1);
  map1[name_base + "2"] = std::make_unique<PP>(pp2);

  FoTMap expected_map{};
  expected_map[name_base + "0"] = std::make_unique<PP>(pp0);
  expected_map[name_base + "1"] = std::make_unique<PP>(pp1);
  expected_map[name_base + "2"] = std::make_unique<PP>(pp2);

  FoTMap combined_map =
      domain::FunctionsOfTime::combine_functions_of_time(map0, map1);

  CHECK(combined_map.size() == expected_map.size());
  CHECK(keys_of(combined_map) == keys_of(expected_map));
  for (auto& [name, f_of_t] : expected_map) {
    CHECK(dynamic_cast<PP&>(*f_of_t) ==
          dynamic_cast<PP&>(*(combined_map.at(name))));
  }
}
}  // namespace
