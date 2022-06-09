// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <hdf5.h>
#include <memory>
#include <string>
#include <utility>

#include "IO/H5/AccessType.hpp"
#include "IO/H5/CheckH5.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Helpers.hpp"
#include "IO/H5/OpenGroup.hpp"
#include "IO/H5/Wrappers.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Utilities/FileSystem.hpp"

namespace {
void test_errors() {
  CHECK_THROWS_WITH(
      []() {
        const std::string file_name("Unit.IO.H5.OpenGroup_read_access.h5");
        if (file_system::check_if_file_exists(file_name)) {
          file_system::rm(file_name, true);
        }

        const hid_t file_id = H5Fcreate(file_name.c_str(), h5::h5f_acc_trunc(),
                                        h5::h5p_default(), h5::h5p_default());
        CHECK_H5(file_id, "Failed to open file: " << file_name);
        {
          h5::detail::OpenGroup group(file_id, "/group",
                                      h5::AccessType::ReadOnly);
        }
        CHECK_H5(H5Fclose(file_id),
                 "Failed to close file: '" << file_name << "'");
      }(),
      Catch::Contains("Cannot create group 'group' in path: /group because the "
                      "access is ReadOnly"));
  if (file_system::check_if_file_exists(
          "Unit.IO.H5.OpenGroup_read_access.h5")) {
    file_system::rm("Unit.IO.H5.OpenGroup_read_access.h5", true);
  }

  CHECK_THROWS_WITH(
      []() {
        h5::detail::OpenGroup group(-1, "/group", h5::AccessType::ReadWrite);
      }(),
      Catch::Contains(
          "Failed to open the group 'group' because the file_id passed in is "
          "invalid, or because the group_id inside the OpenGroup constructor "
          "got corrupted. It is most likely that the file_id is invalid."));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.IO.H5.OpenGroupMove", "[Unit][IO][H5]") {
  const std::string file_name("Unit.IO.H5.OpenGroupMove.h5");
  const std::string file_name2("Unit.IO.H5.OpenGroupMove2.h5");

  {
    if (file_system::check_if_file_exists(file_name)) {
      file_system::rm(file_name, true);
    }
    const hid_t file_id = H5Fcreate(file_name.c_str(), h5::h5f_acc_trunc(),
                                    h5::h5p_default(), h5::h5p_default());
    CHECK_H5(file_id, "Failed to open file: " << file_name);
    auto group = std::make_unique<h5::detail::OpenGroup>(
        file_id, "/group/group2/group3", h5::AccessType::ReadWrite);
    h5::detail::OpenGroup group2(std::move(*group));
    group.reset();
    CHECK(group == nullptr);
    CHECK_H5(H5Fclose(file_id), "Failed to close file: '" << file_name << "'");
    if (file_system::check_if_file_exists(file_name)) {
      file_system::rm(file_name, true);
    }
  }

  {
    if (file_system::check_if_file_exists(file_name)) {
      file_system::rm(file_name, true);
    }
    if (file_system::check_if_file_exists(file_name2)) {
      file_system::rm(file_name2, true);
    }
    const hid_t file_id = H5Fcreate(file_name.c_str(), h5::h5f_acc_trunc(),
                                    h5::h5p_default(), h5::h5p_default());
    CHECK_H5(file_id, "Failed to open file: " << file_name);
    const hid_t file_id2 = H5Fcreate(file_name2.c_str(), h5::h5f_acc_trunc(),
                                     h5::h5p_default(), h5::h5p_default());
    CHECK_H5(file_id2, "Failed to open file: " << file_name);
    {
      h5::detail::OpenGroup group(file_id, "/group/group2/group3",
                                  h5::AccessType::ReadWrite);
      h5::detail::OpenGroup group2(file_id2, "/group/group2/group3",
                                   h5::AccessType::ReadWrite);
      group2 = std::move(group);
      h5::detail::OpenGroup group3(file_id2, "/group/group2/group3",
                                   h5::AccessType::ReadWrite);
    }
    CHECK_H5(H5Fclose(file_id), "Failed to close file: '" << file_name << "'");
    CHECK_H5(H5Fclose(file_id2),
             "Failed to close file: '" << file_name2 << "'");
    if (file_system::check_if_file_exists(file_name)) {
      file_system::rm(file_name, true);
    }
    if (file_system::check_if_file_exists(file_name2)) {
      file_system::rm(file_name2, true);
    }
  }
  {
    if (file_system::check_if_file_exists(file_name)) {
      file_system::rm(file_name, true);
    }
    const hid_t file_id = H5Fcreate(file_name.c_str(), h5::h5f_acc_trunc(),
                                    h5::h5p_default(), h5::h5p_default());
    h5::detail::OpenGroup group(file_id, "/", h5::AccessType::ReadWrite);
    CHECK(group.group_path_with_trailing_slash() == "/");
    h5::detail::OpenGroup group2(file_id, "/group/group2/group3",
                                 h5::AccessType::ReadWrite);
    CHECK(group2.group_path_with_trailing_slash() == "/group/group2/group3/");
    if (file_system::check_if_file_exists(file_name)) {
      file_system::rm(file_name, true);
    }
  }

  test_errors();
}
