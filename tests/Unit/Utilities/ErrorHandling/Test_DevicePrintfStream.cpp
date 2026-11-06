// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <ios>
#include <limits>
#include <vector>

#include "Utilities/ErrorHandling/DevicePrintfStream.hpp"

namespace {
enum class Color : uint8_t { Red = 2, Green = 5 };
}  // namespace

// The message is printed right away, so check the output.
// [[OutputRegex, Hello abc \(null\) 10 -12 345 -9223372036854775808
// 18446744073709551615 5 \(1,-2,3\) \(\) \(\(1,2\),\(3,4\)\) a<\?>b
// 1\.000000000000000000e\+00 -2\.500000000000000000e-01 nan -inf end]]
SPECTRE_TEST_CASE("Unit.ErrorHandling.DevicePrintfStream",
                  "[Unit][ErrorHandling]") {
  OUTPUT_TEST();
  const char* const c_string = "abc";
  DevicePrintfStream message{};
  message << "Hello" << ' ' << c_string << ' '
          << static_cast<const char*>(nullptr) << ' ' << true << false << ' '
          << -12 << ' ' << 345U << ' ' << std::numeric_limits<long long>::min()
          << ' ' << std::numeric_limits<unsigned long long>::max() << ' '
          << Color::Green << ' ' << std::array<int, 3>{{1, -2, 3}} << ' '
          << std::array<double, 0>{} << ' '
          << std::array<std::array<int, 2>, 2>{{{{1, 2}}, {{3, 4}}}}
          << ' '
          // Unsupported types and ignored manipulators
          << 'a' << std::vector<double>{1.0} << 'b' << ' '
          << std::scientific
          // Same format as `ERROR` on the host
          << 1.0 << ' ' << -0.25F << ' '
          << std::numeric_limits<double>::quiet_NaN() << ' '
          << -std::numeric_limits<double>::infinity() << " end\n";
  REQUIRE(std::fflush(stdout) == 0);
}
