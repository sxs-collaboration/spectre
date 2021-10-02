// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/SourceArchive.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "IO/H5/Helpers.hpp"
#include "Utilities/Formaline.hpp"

namespace h5 {
SourceArchive::SourceArchive(const bool exists, detail::OpenGroup&& group,
                             const hid_t location, const std::string& name)
    : group_(std::move(group)) {
  if (exists) {
    source_archive_ =
        read_data<1, std::vector<char>>(location, name + extension());
  } else {
    source_archive_ = formaline::get_archive();
    write_data(location, source_archive_, {source_archive_.size()},
               name + extension());
  }
}
}  // namespace h5
