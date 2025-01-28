// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/UtcTime.hpp"

#include <sstream>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/posix_time/posix_time_io.hpp>

std::string utc_time() {
  static const std::string datetime_format = "%Y-%m-%d %H:%M:%S UTC";
  // Use this needlessly complicated implementation with Boost because the
  // needed features from C++20 aren't available yet in all compilers.
  const auto now_utc = boost::posix_time::second_clock::universal_time();
  auto* time_facet = new boost::posix_time::time_facet(datetime_format.c_str());
  std::ostringstream oss;
  oss.imbue(std::locale(std::locale::classic(), time_facet));
  oss << now_utc;
  return oss.str();
}
