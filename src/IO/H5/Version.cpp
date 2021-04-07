// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/Version.hpp"

#include <algorithm>

#include "Helpers.hpp"

namespace h5 {
Version::Version(bool exists, detail::OpenGroup&& group, hid_t location,
                 std::string name, const uint32_t version)
    : version_([&exists, &location, &version, &name]() {
        if (exists) {
          return read_value_attribute<uint32_t>(location, name + extension());
        }
        write_to_attribute(location, name + extension(), version);
        return version;
      }()),
      group_(std::move(group)) {}

Version::Version(bool exists, detail::OpenGroup&& group, hid_t location,
                 std::string name, const std::string version)
    : version_([&exists, &location, &version, &name]() {
        if (exists) {
          return read_value_attribute<uint32_t>(location, name + extension());
        }
        write_to_attribute(location, name + extension(),
                           std::vector<std::string>{version});
        return static_cast<uint32_t>(0);
      }()),
      group_(std::move(group)) {}

}  // namespace h5
