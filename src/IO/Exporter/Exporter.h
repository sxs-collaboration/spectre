// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// C interface to the functions in Exporter.hpp, so they can be called from C,
/// Fortran, Mathematica, or any other language with a C foreign-function
/// interface. This file is intended to be included in external programs, so it
/// intentionally has no dependencies on any other headers.

#pragma once

#ifdef __cplusplus
#include <cstddef>
#else
#include <stdbool.h>
#include <stddef.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Interpolate data in volume files to target points
 *
 * This is the C interface to `spectre::Exporter::interpolate_to_points`. See
 * Exporter.hpp for details on the arguments. The data is written directly into
 * the caller's `out` buffers, and the `target_points` are read in place, so no
 * memory proportional to the number of target points is copied.
 *
 * \param out Array of `num_tensor_components` pointers, each to a buffer of
 * `num_points` doubles that will be filled with the interpolated data of the
 * corresponding tensor component. The buffers can be separate arrays or point
 * into a single contiguous array of size `num_tensor_components * num_points`.
 * Points outside the domain are filled with NaNs (unless
 * `error_on_missing_points` is set).
 * \param volume_files Array of `num_volume_files` null-terminated paths to the
 * H5 volume data files.
 * \param num_volume_files Number of volume data files.
 * \param subfile_name Name of the subfile in the H5 files containing the volume
 * data.
 * \param observation_step Index of the observation in the volume files. A value
 * of 0 is the first observation, and a value of -1 is the last observation.
 * \param tensor_components Array of `num_tensor_components` null-terminated
 * names of the tensor components to interpolate, e.g. "Lapse" or "Shift_x".
 * \param num_tensor_components Number of tensor components.
 * \param target_points Array of `num_dimensions` pointers, each to a buffer of
 * `num_points` coordinates in the inertial frame (first the x-coordinates of
 * all points, then the y-coordinates, etc.). The buffers can be separate arrays
 * or point into a single contiguous array of size
 * `num_dimensions * num_points`.
 * \param num_dimensions Number of dimensions of the volume data (1, 2, or 3).
 * \param num_points Number of target points.
 * \param extrapolate_into_excisions Enables extrapolation into excision regions
 * of the domain.
 * \param error_on_missing_points If `true`, it's an error if any of the target
 * points are outside the domain.
 * \param num_threads The number of OpenMP threads to use. Zero or negative
 * values let OpenMP determine the number of threads.
 * \return 0 on success. On failure, a nonzero value is returned and
 * `spectre_exporter_last_error` returns the error message.
 */
int spectre_interpolate_to_points(
    double* const* out, const char* const* volume_files,
    size_t num_volume_files, const char* subfile_name, int observation_step,
    const char* const* tensor_components, size_t num_tensor_components,
    const double* const* target_points, size_t num_dimensions,
    size_t num_points, bool extrapolate_into_excisions,
    bool error_on_missing_points, int num_threads);

/*!
 * \brief The error message of the last failed call to the C interface on this
 * thread, or an empty string if the last call succeeded
 *
 * The returned string is owned by the library and stays valid until the next
 * call to the C interface on this thread.
 */
const char* spectre_exporter_last_error(void);

#ifdef __cplusplus
}
#endif
