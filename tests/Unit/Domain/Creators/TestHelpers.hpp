// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "tests/Unit/TestingFramework.hpp"

#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Utilities/Tuple.hpp"

namespace TestHelpers {
namespace domain {
namespace creators {
template <typename Creator, typename... ExpectedFunctionsOfTime>
void test_functions_of_time(
    const Creator& creator,
    const std::tuple<std::pair<std::string, ExpectedFunctionsOfTime>...>&
        expected_functions_of_time) {
  const std::unordered_map<
      std::string, std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>&
      functions_of_time = creator.functions_of_time();
  REQUIRE(functions_of_time.size() == sizeof...(ExpectedFunctionsOfTime));

  tuple_fold(
      expected_functions_of_time,
      [&functions_of_time](const auto& name_and_function_of_time) noexcept {
        const std::string& name = name_and_function_of_time.first;
        const auto& function_of_time = name_and_function_of_time.second;
        using FunctionOfTimeType = std::decay_t<decltype(function_of_time)>;
        const bool in_functions_of_time =
            functions_of_time.find(name) != functions_of_time.end();
        CHECK(in_functions_of_time);
        if (in_functions_of_time) {
          const auto* function_from_creator =
              dynamic_cast<const FunctionOfTimeType*>(
                  functions_of_time.at(name).get());
          REQUIRE(function_from_creator != nullptr);
          CHECK(*function_from_creator == function_of_time);
        }
      });
}
}  // namespace creators
}  // namespace domain
}  // namespace TestHelpers
