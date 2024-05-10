// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

/// \cond
class DataVector;
namespace Spectral {
enum class Basis : uint8_t;
enum class Quadrature : uint8_t;
}  // namespace Spectral
/// \endcond

/// Functions for testing volume data output.
namespace TestHelpers::io::VolumeData {
/// Helper function that multiplies a tensor component by a double, which is
/// typically the observation time, to generate different tensor values at
/// different times for testing.
template <typename T>
T multiply(const double obs_value, const T& component) {
  T result = component;
  for (auto& t : result) {
    t *= obs_value;
  }
  return result;
}

/// Helper function to check that volume data was written correctly.
/// This function checks the following:
///    0. That the provided observation_id is present in the file
///    1. That the grid_names provided are present in the file
///    2. That the provided bases and quadratures agree with the bases
///       and quadratures in the file.
///    3. That the expected_components are present in the file.
///    4. That the expected_components, after rescaling them by a constant
///       factor_to_rescale_components, agree with the components in the file,
///       except for those that are invalid_components which should be nans in
///       the file.
///       Note: if components_comparison_precision is defined, then the
///       comparison is approximate, using *components_comparison_precision as
///       the tolerance.
template <typename DataType>
void check_volume_data(
    const std::string& h5_file_name, const uint32_t version_number,
    const std::string& group_name, const size_t observation_id,
    const double observation_value,
    const std::vector<DataType>& tensor_components_and_coords,
    const std::vector<std::string>& grid_names,
    const std::vector<std::vector<Spectral::Basis>>& bases,
    const std::vector<std::vector<Spectral::Quadrature>>& quadratures,
    const std::vector<std::vector<size_t>>& extents,
    const std::vector<std::string>& expected_components,
    const std::vector<std::vector<size_t>>& grid_data_orders,
    const std::optional<double>& components_comparison_precision,
    double factor_to_rescale_components = 1.0,
    const std::vector<std::string>& invalid_components = {});
}  // namespace TestHelpers::io::VolumeData
