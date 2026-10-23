// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Helpers/IO/H5Identifiers.hpp"

#include <hdf5.h>
#include <string>
#include <vector>

#include "IO/H5/CheckH5.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace TestHelpers::io {
namespace {
// Create and immediately close an object so its identifier marks the current
// position in the sequence of identifiers of its kind.
template <typename Create, typename Close>
hid_t probe(const std::string& kind, const Create& create, const Close& close) {
  const hid_t id = create();
  CHECK_H5(id, "Failed to create HDF5 " << kind << " probe");
  CHECK_H5(close(id), "Failed to close HDF5 " << kind << " probe");
  return id;
}

// Describe an open identifier to help locate the code that leaked it.
std::string describe_identifier(const hid_t id) {
  switch (H5Iget_type(id)) {
    case H5I_GENPROP_LST: {
      const hid_t class_id = H5Pget_class(id);
      if (class_id < 0) {
        return "property list of unknown class";
      }
      char* class_name = H5Pget_class_name(class_id);
      const std::string result =
          class_name == nullptr
              ? std::string{"property list of unknown class"}
              : "property list of class '" + std::string{class_name} + "'";
      H5free_memory(class_name);
      H5Pclose_class(class_id);
      return result;
    }
    case H5I_DATATYPE:
      return "datatype of class " +
             std::to_string(static_cast<int>(H5Tget_class(id)));
    case H5I_DATASPACE:
      return "dataspace of rank " +
             std::to_string(H5Sget_simple_extent_ndims(id));
    default:
      return "identifier of type " +
             std::to_string(static_cast<int>(H5Iget_type(id)));
  }
}

// Describe the identifiers strictly between two probes that are still open.
std::vector<std::string> open_identifiers_between(const std::string& kind,
                                                  const hid_t first_probe,
                                                  const hid_t second_probe) {
  // Guard against the sequential-identifier assumption being violated.
  constexpr hid_t max_identifiers_to_scan = 100'000'000;
  if (second_probe < first_probe or
      second_probe - first_probe > max_identifiers_to_scan) {
    ERROR("Unexpected HDF5 " << kind << " probe identifiers " << first_probe
                             << " and " << second_probe
                             << ". Identifiers are expected to be handed out "
                                "sequentially.");
  }
  std::vector<std::string> open_identifiers{};
  for (hid_t id = first_probe + 1; id < second_probe; ++id) {
    const htri_t is_open = H5Iis_valid(id);
    CHECK_H5(is_open, "Failed to query validity of HDF5 identifier " << id);
    if (is_open > 0) {
      open_identifiers.push_back(describe_identifier(id));
    }
  }
  return open_identifiers;
}

void check_no_leaked_identifiers_between(const std::string& kind,
                                         const hid_t first_probe,
                                         const hid_t second_probe) {
  const auto leaked = open_identifiers_between(kind, first_probe, second_probe);
  std::string description{};
  for (const auto& leaked_identifier : leaked) {
    description += "\n  " + leaked_identifier;
  }
  INFO("Leaked HDF5 " << kind << " identifiers:" << description);
  CHECK(leaked.empty());
}
}  // namespace

H5IdentifierSnapshot snapshot_h5_identifiers() {
  H5IdentifierSnapshot snapshot{};
  const ssize_t number_of_file_objects =
      H5Fget_obj_count(H5F_OBJ_ALL, H5F_OBJ_ALL);
  CHECK_H5(number_of_file_objects, "Failed to count open HDF5 file objects");
  snapshot.number_of_file_objects = static_cast<size_t>(number_of_file_objects);
  snapshot.datatype_probe =
      probe("datatype", []() { return H5Tcopy(H5T_NATIVE_INT); }, H5Tclose);
  snapshot.dataspace_probe =
      probe("dataspace", []() { return H5Screate(H5S_SCALAR); }, H5Sclose);
  snapshot.property_list_probe = probe(
      "property list", []() { return H5Pcreate(H5P_DATASET_CREATE); },
      H5Pclose);
  return snapshot;
}

void check_no_leaked_h5_identifiers(const H5IdentifierSnapshot& snapshot) {
  const H5IdentifierSnapshot now = snapshot_h5_identifiers();
  {
    INFO(
        "Open HDF5 files, groups, datasets, attributes and named datatypes "
        "when the snapshot was taken: "
        << snapshot.number_of_file_objects
        << ", now: " << now.number_of_file_objects);
    CHECK(now.number_of_file_objects == snapshot.number_of_file_objects);
  }
  check_no_leaked_identifiers_between("datatype", snapshot.datatype_probe,
                                      now.datatype_probe);
  check_no_leaked_identifiers_between("dataspace", snapshot.dataspace_probe,
                                      now.dataspace_probe);
  check_no_leaked_identifiers_between(
      "property list", snapshot.property_list_probe, now.property_list_probe);
}
}  // namespace TestHelpers::io
