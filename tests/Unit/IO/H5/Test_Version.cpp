// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstdint>
#include <hdf5.h>
#include <string>

#include "IO/H5/AccessType.hpp"
#include "IO/H5/CheckH5.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Version.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Utilities/FileSystem.hpp"

SPECTRE_TEST_CASE("Unit.IO.H5.Version", "[Unit][IO][H5]") {
  const std::string h5_file_name("Unit.IO.H5.Version.h5");
  const uint32_t version_number = 2;
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  {
    // [h5file_write_version]
    h5::H5File<h5::AccessType::ReadWrite> my_file(h5_file_name);
    my_file.insert<h5::Version>("/the_version", version_number);
    // [h5file_write_version]
  }

  h5::H5File<h5::AccessType::ReadOnly> my_file(h5_file_name);
  // [h5file_read_version]
  const auto& const_version = my_file.get<h5::Version>("/the_version");
  // [h5file_read_version]
  CHECK(const_version.subfile_path() == "/the_version");
  const auto const_version_val = const_version.get_version();
  my_file.close_current_object();
  CHECK(version_number == const_version_val);
  auto& version = my_file.get<h5::Version>("/the_version");
  const auto version_val = version.get_version();
  CHECK(version_number == version_val);
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}
