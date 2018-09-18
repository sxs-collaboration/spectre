// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/Version.hpp"

#include <algorithm>

#include "Helpers.hpp"

namespace h5 {
/// \cond HIDDEN_SYMOLS
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
/// \endcond
}  // namespace h5
