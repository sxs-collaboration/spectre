// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "Utilities/Base64.hpp"

namespace {
void check_encoding(const std::string& data, const std::string& encoded) {
  std::vector<std::byte> binary_data(data.size());
  std::transform(data.begin(), data.end(), binary_data.begin(),
                 [](const char c) { return static_cast<std::byte>(c); });
  CHECK(base64_encode(binary_data) == encoded);
  CHECK(base64_decode(encoded) == binary_data);
}

void test_errors() {
  CHECK_THROWS_WITH(
      base64_decode("x"),
      Catch::Matchers::ContainsSubstring(
          "base64 encoded data must have a multiple of 4 characters, not 1"));

  CHECK_THROWS_WITH(base64_decode("xx*x"),
                    Catch::Matchers::ContainsSubstring(
                        "Invalid character in base64 encoded string: '*'"));

  CHECK_THROWS_WITH(base64_decode("===="),
                    Catch::Matchers::ContainsSubstring(
                        "Misplaced padding character at position 0 of base64 "
                        "encoded string."));
  CHECK_THROWS_WITH(base64_decode("x==="),
                    Catch::Matchers::ContainsSubstring(
                        "Misplaced padding character at position 1 of base64 "
                        "encoded string."));
  CHECK_THROWS_WITH(base64_decode("x=xx"),
                    Catch::Matchers::ContainsSubstring(
                        "Misplaced padding character at position 1 of base64 "
                        "encoded string."));
  CHECK_THROWS_WITH(base64_decode("xx=x"),
                    Catch::Matchers::ContainsSubstring(
                        "Misplaced padding character at position 2 of base64 "
                        "encoded string."));
  CHECK_THROWS_WITH(base64_decode("xxx=xxxx"),
                    Catch::Matchers::ContainsSubstring(
                        "Misplaced padding character at position 3 of base64 "
                        "encoded string."));
  CHECK_THROWS_WITH(base64_decode("xxx====="),
                    Catch::Matchers::ContainsSubstring(
                        "Misplaced padding character at position 3 of base64 "
                        "encoded string."));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.Base64", "[Unit][Utilities]") {
  check_encoding("", "");
  check_encoding("f", "Zg==");
  check_encoding("fo", "Zm8=");
  check_encoding("foo", "Zm9v");
  check_encoding("foob", "Zm9vYg==");
  check_encoding("fooba", "Zm9vYmE=");
  check_encoding("foobar", "Zm9vYmFy");

  check_encoding({0}, "AA==");

  test_errors();
}
