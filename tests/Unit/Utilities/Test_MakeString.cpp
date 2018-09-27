// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <string>

#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/StdHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.MakeString", "[Unit][Utilities]") {
  /// [make_string]
  std::array<int, 3> arr{{2, 3, 4}};
  const std::string t = MakeString{} << "Test" << 2 << arr << "Done";
  /// [make_string]
  const std::string expected = "Test2" + get_output(arr) + "Done";
  CHECK(t == expected);
  CHECK(get_output(MakeString{} << "Test" << 2 << arr << "Done") == expected);
}
