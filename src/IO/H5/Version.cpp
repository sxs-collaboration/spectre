// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/Version.hpp"

#include "ErrorHandling/Error.hpp"
#include "Helpers.hpp"
#include "IO/H5/CheckH5.hpp"

namespace h5 {
/// \cond HIDDEN_SYMOLS
Version::Version(bool exists, detail::OpenGroup&& group, hid_t location,
                 std::string name, const uint32_t version)
    : version_([&exists, &location, &version, &name]() {
        if (exists) {
          return detail::read_value_from_attribute<uint32_t>(
              location, name + extension());
        }
        detail::write_value_to_attribute(location, name + extension(), version);
        return version;
      }()),
      group_(std::move(group)) {}
/// \endcond
}  // namespace h5
