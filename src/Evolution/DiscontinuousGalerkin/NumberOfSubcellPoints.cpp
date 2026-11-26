// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/NumberOfSubcellPoints.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/MinimumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace evolution::dg {
size_t number_of_subcell_points(
    const size_t num_dg_points,
    const std::array<int, 3>& number_of_points_coefficients) {
  ASSERT(num_dg_points > 0, "The number of DG grid points must be positive.");
  const auto n = static_cast<int64_t>(num_dg_points);
  const int64_t num_fd_points =
      static_cast<int64_t>(number_of_points_coefficients[0]) * n * n +
      static_cast<int64_t>(number_of_points_coefficients[1]) * n +
      static_cast<int64_t>(number_of_points_coefficients[2]);
  constexpr auto min_num_fd_points = static_cast<int64_t>(
      Spectral::minimum_number_of_points<Spectral::Basis::FiniteDifference,
                                         Spectral::Quadrature::CellCentered>);
  constexpr auto max_num_fd_points = static_cast<int64_t>(
      Spectral::maximum_number_of_points<Spectral::Basis::FiniteDifference,
                                         Spectral::Quadrature::CellCentered>);
  if (num_fd_points < n) {
    ERROR("The number of FD grid points ("
          << num_fd_points << " = " << number_of_points_coefficients[0] << " * "
          << n << "^2 + " << number_of_points_coefficients[1] << " * " << n
          << " + " << number_of_points_coefficients[2]
          << ") must be at least the number of DG grid points (" << n
          << "). Change NumberOfSubcellPoints.");
  }
  if (num_fd_points < min_num_fd_points or num_fd_points > max_num_fd_points) {
    ERROR("The number of FD grid points ("
          << num_fd_points << " = " << number_of_points_coefficients[0] << " * "
          << n << "^2 + " << number_of_points_coefficients[1] << " * " << n
          << " + " << number_of_points_coefficients[2] << ") for " << n
          << " DG grid points must be in [" << min_num_fd_points << ", "
          << max_num_fd_points << "]. Change NumberOfSubcellPoints.");
  }
  return static_cast<size_t>(num_fd_points);
}
}  // namespace evolution::dg
