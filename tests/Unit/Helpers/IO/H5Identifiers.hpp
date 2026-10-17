// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <hdf5.h>

namespace TestHelpers::io {
/*!
 * \brief Snapshot of the open HDF5 identifiers, used by
 * `check_no_leaked_h5_identifiers()` to detect leaked HDF5 handles.
 *
 * Files, groups, datasets, attributes and named datatypes are counted with
 * `H5Fget_obj_count`. Transient datatypes, dataspaces and property lists are
 * not attached to a file, so for those kinds the snapshot records the
 * identifier of a probe object that is created and closed immediately. HDF5
 * hands out the identifiers of each kind sequentially, so every identifier of
 * a kind that is created later lies between two consecutive probes and
 * `H5Iis_valid` tells whether it is still open.
 */
struct H5IdentifierSnapshot {
  size_t number_of_file_objects{0};
  hid_t datatype_probe{-1};
  hid_t dataspace_probe{-1};
  hid_t property_list_probe{-1};
};

/// Take a snapshot of the open HDF5 identifiers.
H5IdentifierSnapshot snapshot_h5_identifiers();

/*!
 * \brief Check that every HDF5 identifier created since `snapshot` was taken
 * has been closed again.
 *
 * \note Code that throws from inside the h5 library (e.g. `CHECK_THROWS_WITH`
 * tests of error messages) legitimately leaves handles open, so only wrap
 * code paths that complete successfully.
 */
void check_no_leaked_h5_identifiers(const H5IdentifierSnapshot& snapshot);
}  // namespace TestHelpers::io
