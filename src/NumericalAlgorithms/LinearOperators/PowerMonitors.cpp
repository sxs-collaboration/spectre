// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/PowerMonitors.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "NumericalAlgorithms/Interpolation/LinearRegression.hpp"
#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctions/Fourier.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/ZernikeB2.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace PowerMonitors {

template <typename VectorType, size_t Dim>
void power_monitors(const gsl::not_null<std::array<DataVector, Dim>*> result,
                    const VectorType& u, const Mesh<Dim>& mesh) {
  ASSERT(u.size() == mesh.number_of_grid_points(),
         "The number of grid points per element ("
             << mesh.number_of_grid_points()
             << ") must match the size of the "
                "vector ("
             << u.size() << ").");
  if (mesh.basis(0) == Spectral::Basis::ZernikeB2) {
    if constexpr (std::is_same_v<VectorType, DataVector>) {
      if constexpr (Dim > 1) {
        Spectral::b2_power_monitor_radial(make_not_null(&(*result)[0]), u,
                                          mesh);
        Spectral::b2_power_monitor_angular(make_not_null(&(*result)[1]), u,
                                           mesh);
        if (Dim == 3) {
          // Compute the z power monitor directly: for each disk point, extract
          // and transform the z-strip using a 1D mesh for the z-dimension only.
          const size_t n_disk = mesh.extents(0) * mesh.extents(1);
          const size_t n_z = mesh.extents(2);
          const Mesh<1> z_mesh{n_z, mesh.basis(2), mesh.quadrature(2)};
          DataVector z_slice(n_z);
          gsl::at(*result, 2).destructive_resize(n_z);
          gsl::at(*result, 2) = 0.0;
          for (size_t i_disk = 0; i_disk < n_disk; ++i_disk) {
            for (size_t j_z = 0; j_z < n_z; ++j_z) {
              z_slice[j_z] = u[i_disk + n_disk * j_z];
            }
            const auto z_modal = to_modal_coefficients(z_slice, z_mesh);
            for (size_t k = 0; k < n_z; ++k) {
              gsl::at(*result, 2)[k] += square(z_modal[k]);
            }
          }
          gsl::at(*result, 2) =
              sqrt(gsl::at(*result, 2) / static_cast<double>(n_disk));
        }
        return;
      } else {
        ERROR("Passed mesh is using ZernikeB2 for Dim == 1");
      }
    } else {
      ERROR(
          "Support for complex numbers with ZernikeB2 power monitor has not "
          "been tested yet");
    }
  }

  // Get modal coefficients
  const auto modal_coefficients = to_modal_coefficients(u, mesh);

  double slice_sum = 0.0;
  size_t n_slice = 0;
  size_t n_stripe = 0;
  for (size_t sliced_dim = 0; sliced_dim < Dim; ++sliced_dim) {
    n_slice = mesh.extents().slice_away(sliced_dim).product();
    n_stripe = mesh.extents(sliced_dim);

    gsl::at(*result, sliced_dim).destructive_resize(n_stripe);

    for (size_t index = 0; index < n_stripe; ++index) {
      slice_sum = 0.0;
      for (SliceIterator si(mesh.extents(), sliced_dim, index); si; ++si) {
        const auto& mode = modal_coefficients[si.volume_offset()];
        if constexpr (std::is_same_v<VectorType, ComplexDataVector>) {
          slice_sum += square(abs(mode));
        } else {
          slice_sum += square(mode);
        }
      }
      slice_sum /= static_cast<double>(n_slice);
      slice_sum = sqrt(slice_sum);

      gsl::at(*result, sliced_dim)[index] = slice_sum;
    }

    // For Fourier dimensions, combine cos and sin power for each wavenumber.
    // Output size is N/2+1, one entry per distinct wavenumber 0..N/2.
    if (mesh.basis(sliced_dim) == Spectral::Basis::Fourier) {
      ASSERT(mesh.extents(sliced_dim) % 2 == 1,
             "We expect Fourier basis to have an odd number of collocation "
             "points, got "
                 << mesh.extents(sliced_dim));
      const DataVector& raw_pm = gsl::at(*result, sliced_dim);
      const size_t n_combined = n_stripe / 2 + 1;
      DataVector combined(n_combined);
      combined[0] = raw_pm[0];
      const size_t n_pairs = n_combined - 1;
      for (size_t m = 1; m <= n_pairs; ++m) {
        combined[m] = std::hypot(
            raw_pm[Spectral::Fourier::modal_storage_index(static_cast<int>(m))],
            raw_pm[Spectral::Fourier::modal_storage_index(
                -static_cast<int>(m))]);
      }
      gsl::at(*result, sliced_dim) = std::move(combined);
    }
  }
}

template <typename VectorType, size_t Dim>
std::array<DataVector, Dim> power_monitors(const VectorType& u,
                                           const Mesh<Dim>& mesh) {
  std::array<DataVector, Dim> result{};
  power_monitors(make_not_null(&result), u, mesh);
  return result;
}

void spherical_shell_radial_power_monitor(
    const gsl::not_null<DataVector*> result, const DataVector& tensor_component,
    const Mesh<3>& mesh) {
  if (mesh.basis(0) == Spectral::Basis::SphericalHarmonic or
      mesh.basis(1) != Spectral::Basis::SphericalHarmonic or
      mesh.basis(2) != Spectral::Basis::SphericalHarmonic) {
    ERROR(
        "Spherical-shell power monitors require the mesh dimensions to be "
        "ordered (radial, theta, phi), but the mesh is "
        << mesh);
  }
  if (tensor_component.size() != mesh.number_of_grid_points()) {
    ERROR("Expected a tensor component with "
          << mesh.number_of_grid_points() << " grid points, but got "
          << tensor_component.size() << ".");
  }

  const auto radial_mesh = mesh.slice_through(0);
  const size_t number_of_radial_points = mesh.extents(0);
  const size_t number_of_angular_points = mesh.extents(1) * mesh.extents(2);
  result->destructive_resize(number_of_radial_points);
  *result = 0.0;
  ModalVector radial_modes(number_of_radial_points, 0.0);
  const DataVector radial_slice{};
  for (size_t angular_index = 0; angular_index < number_of_angular_points;
       ++angular_index) {
    make_const_view(make_not_null(&radial_slice), tensor_component,
                    angular_index * number_of_radial_points,
                    number_of_radial_points);
    to_modal_coefficients(make_not_null(&radial_modes), radial_slice,
                          radial_mesh);
    *result += square(radial_modes);
  }
  *result = sqrt(*result / static_cast<double>(number_of_angular_points));
}

DataVector spherical_shell_radial_power_monitor(
    const DataVector& tensor_component, const Mesh<3>& mesh) {
  DataVector result{};
  spherical_shell_radial_power_monitor(make_not_null(&result), tensor_component,
                                       mesh);
  return result;
}

void spherical_shell_angular_power_monitor(
    const gsl::not_null<DataVector*> result,
    const DataVector& tensor_ylm_component, const Mesh<3>& mesh,
    const int spin_weight, const bool zero_m_is_real) {
  if (mesh.basis(0) == Spectral::Basis::SphericalHarmonic or
      mesh.basis(1) != Spectral::Basis::SphericalHarmonic or
      mesh.basis(2) != Spectral::Basis::SphericalHarmonic) {
    ERROR(
        "Spherical-shell power monitors require the mesh dimensions to be "
        "ordered (radial, theta, phi), but the mesh is "
        << mesh);
  }
  const size_t radial_extents = mesh.extents(0);
  const size_t ell_max = mesh.extents(1) - 1;
  const size_t m_max = (mesh.extents(2) - 1) / 2;
  if (ell_max != m_max) {
    ERROR(
        "Spherical-shell TensorYlm power monitors require l_max == m_max, "
        "but got l_max = "
        << ell_max << " and m_max = " << m_max << ".");
  }
  const ylm::SpherepackIterator iterator{ell_max, m_max, 1, zero_m_is_real};
  if (tensor_ylm_component.size() !=
      radial_extents * iterator.spherepack_array_size()) {
    ERROR("Expected a TensorYlm component with "
          << radial_extents * iterator.spherepack_array_size()
          << " coefficients, but got " << tensor_ylm_component.size() << ".");
  }

  result->destructive_resize(ell_max + 1);
  *result = 0.0;
  std::vector<size_t> counts(ell_max + 1, 0);
  const auto abs_spin_weight = static_cast<size_t>(std::abs(spin_weight));
  for (ylm::SpherepackIterator it{ell_max, m_max, 1, zero_m_is_real}; it;
       ++it) {
    if (it.l() < abs_spin_weight) {
      continue;
    }
    for (size_t r = 0; r < radial_extents; ++r) {
      (*result)[it.l()] +=
          square(tensor_ylm_component[it() * radial_extents + r]);
      ++counts[it.l()];
    }
  }
  for (size_t ell = 0; ell <= ell_max; ++ell) {
    (*result)[ell] =
        counts[ell] == 0
            ? 0.0
            : sqrt((*result)[ell] / static_cast<double>(counts[ell]));
  }
}

DataVector spherical_shell_angular_power_monitor(
    const DataVector& tensor_ylm_component, const Mesh<3>& mesh,
    const int spin_weight, const bool zero_m_is_real) {
  DataVector result{};
  spherical_shell_angular_power_monitor(make_not_null(&result),
                                        tensor_ylm_component, mesh, spin_weight,
                                        zero_m_is_real);
  return result;
}

size_t spherical_shell_number_of_angular_coefficients(
    const size_t ell, const int spin_weight, const bool zero_m_is_real,
    const size_t radial_extents) {
  // If l < |s|, spin weighted spherical harmonics vanish, so just return
  // 0. In accumulate_tensor_angular_power, this will cause such terms to
  // not contribute to accumulated tensor angular power.
  if (ell < static_cast<size_t>(std::abs(spin_weight))) {
    return 0;
  }
  // Spherepack stores real and imaginary parts separately. For real scalars,
  // m=0 has only a real coefficient, so there are 1 + 2*l coefficients at
  // each l. TensorYlm components are generally complex and retain both parts
  // at m=0, giving 2*(l+1) coefficients. zero_m_is_real says whether we
  // are dealing with a real scalar or not; coefficients_at_ell is set
  // accordingly. See the help text of `SpherepackIterator` for details on the
  // zero_m_is_real parameter.
  const size_t coefficients_at_ell =
      zero_m_is_real ? 2 * ell + 1 : 2 * (ell + 1);
  // Every angular coefficient occurs independently at each radial point.
  return radial_extents * coefficients_at_ell;
}

void normalize_spherical_shell_angular_power(
    const gsl::not_null<DataVector*> power, const std::vector<size_t>& counts) {
  if (power->size() != counts.size()) {
    ERROR(
        "The angular power and count buffers must have the same size, but "
        "got "
        << power->size() << " and " << counts.size() << ".");
  }
  for (size_t ell = 0; ell < power->size(); ++ell) {
    (*power)[ell] =
        counts[ell] == 0
            ? 0.0
            : sqrt((*power)[ell] / static_cast<double>(counts[ell]));
  }
}

// The power_monitor argument should be made of type ModalVector
// when pybindings for ModalVector are enabled
double relative_truncation_error(const DataVector& power_monitor,
                                 const size_t num_modes_to_use) {
  ASSERT(
      num_modes_to_use <= power_monitor.size(),
      "Number of modes needs less or equal than the number of power monitors");
  ASSERT(2_st <= num_modes_to_use,
         "Number of modes needs to be larger or equal than 2.");
  const size_t last_index = num_modes_to_use - 1;
  const double max_mode = blaze::max(power_monitor);
  const double cutoff =
      100. * std::numeric_limits<double>::epsilon() * max_mode;
  // If the last two or more modes are zero, assume that the function is
  // represented exactly and return a relative truncation error of zero.
  // Just one zero mode is not enough to make this assumption, as the function
  // could have zero modes by symmetry.
  if (num_modes_to_use >= 2 and power_monitor[last_index] < cutoff and
      power_monitor[last_index - 1] < cutoff) {
    return cutoff * 1.e-2;
  }
  // Compute weighted average and total sum in the current dimension
  double weighted_average = 0.0;
  double weight_sum = 0.0;
  double weight_value = 0.0;
  for (size_t index = 0; index <= last_index; ++index) {
    const double mode = power_monitor[index];
    if (mode < cutoff) {
      // Ignore modes below this cutoff, so modes that are zero (e.g. by
      // symmetry) don't make us underestimate the truncation error.
      continue;
    }
    // Compute current weight
    weight_value = exp(-square(static_cast<double>(last_index - index) - 0.5));
    // Add weighted power monitor
    weighted_average += weight_value * log10(mode);
    // Add term to weighted sum
    weight_sum += weight_value;
  }
  weighted_average /= weight_sum;

  // Maximum between the first two power monitors
  double leading_term = std::max(power_monitor[0], power_monitor[1]);
  ASSERT(not(leading_term == 0.0),
         "The leading power monitor term is zero bitwise.");

  return std::pow(10.0, weighted_average) / leading_term;
}

template <typename VectorType, size_t Dim>
std::array<double, Dim> relative_truncation_error(
    const VectorType& tensor_component, const Mesh<Dim>& mesh) {
  std::array<double, Dim> result{};
  const auto modes = power_monitors(tensor_component, mesh);
  for (size_t d = 0; d < Dim; ++d) {
    const auto& modes_d = gsl::at(modes, d);
    gsl::at(result, d) = relative_truncation_error(modes_d, modes_d.size());
  }
  return result;
}

template <typename VectorType, size_t Dim>
std::array<double, Dim> absolute_truncation_error(
    const VectorType& tensor_component, const Mesh<Dim>& mesh) {
  std::array<double, Dim> result{};
  const auto modes = power_monitors(tensor_component, mesh);
  // Use infinity norm to estimate the order of magnitude of the variable
  const double umax = max(abs(tensor_component));
  double relative_truncation_error_in_d = 0.0;
  for (size_t d = 0; d < Dim; ++d) {
    const auto& modes_d = gsl::at(modes, d);
    // Compute relative truncation error
    relative_truncation_error_in_d =
        relative_truncation_error(modes_d, modes_d.size());
    // Compute absolute truncation error estimate
    gsl::at(result, d) = umax * relative_truncation_error_in_d;
  }
  return result;
}

ConvergenceInfo convergence_rate_and_number_of_pile_up_modes(
    const DataVector& power_monitor, const size_t number_of_filtered_modes) {
  // Need enough unfiltered modes to compute the convergence rate. Here,
  // require at least 4 unfiltered modes.
  ASSERT(
      power_monitor.size() > number_of_filtered_modes + 3,
      "Power monitor needs at least 4 unfiltered modes to compute convergence "
      "rate and number of pile up modes, but power monitor has size "
          << power_monitor.size() << " with " << number_of_filtered_modes
          << " filtered modes");
  ConvergenceInfo result{};

  const size_t n_tilde = power_monitor.size() - number_of_filtered_modes;
  std::vector<double> mode_numbers_for_fit{};
  std::vector<double> mode_powers_for_fit{};
  mode_numbers_for_fit.reserve(n_tilde);
  mode_powers_for_fit.reserve(n_tilde);

  const size_t n_tilde_minus_one = n_tilde - 1;
  std::vector<double> slopes{};
  std::vector<double> delta_slopes{};

  // It turns out (as can be verified empirically) that the number of terms
  // in the loop to compute the convergence rate is 3 * (n_tilde - 5) for
  // n_tilde > 6, 4 for n_tilde == 5, and 3 for n_tilde == 4 or n_tilde == 3.
  // Here reserve the correct number of elements in terms of n_tilde, except
  // don't use an extra if to distinguish between the 3 and 4 special cases (no
  // harm in reserving space for a single extra double).
  const size_t max_slope_size = n_tilde > 6 ? 3 * (n_tilde - 5) : 4;
  slopes.reserve(max_slope_size);
  delta_slopes.reserve(max_slope_size);

  // Compute log of the power monitor only for unfiltered modes, and
  // ensure that log10 never causes a floating point exception here.
  constexpr double eps_for_log = 100.0 * std::numeric_limits<double>::min();
  const double log_floor = log10(eps_for_log);
  DataVector log_power = power_monitor;
  for (size_t i = 0; i < power_monitor.size() - number_of_filtered_modes; ++i) {
    if (abs(log_power[i]) < eps_for_log) {
      log_power[i] = log_floor;
    } else {
      log_power[i] = log10(abs(log_power[i]));
    }
  }

  // Compute convergence rate
  for (size_t k1 = 0; k1 < 3; ++k1) {
    for (size_t k2 = std::min(k1 + 4, n_tilde_minus_one);
         k2 <= n_tilde_minus_one; ++k2) {
      mode_numbers_for_fit.resize(k2 - k1 + 1);
      mode_powers_for_fit.resize(k2 - k1 + 1);
      for (size_t k = k1; k <= k2; ++k) {
        mode_numbers_for_fit[k - k1] = static_cast<double>(k);
        mode_powers_for_fit[k - k1] = log_power[k];
      }
      if (mode_numbers_for_fit.size() > 2) {
        const intrp::LinearRegressionResult regression_result =
            intrp::linear_regression(mode_numbers_for_fit, mode_powers_for_fit);
        slopes.push_back(regression_result.slope);
        delta_slopes.push_back(regression_result.delta_slope);
      } else if (mode_numbers_for_fit.size() == 2 and
                 mode_numbers_for_fit[0] != mode_numbers_for_fit[1]) {
        slopes.push_back((mode_powers_for_fit[1] - mode_powers_for_fit[0]) /
                         (mode_numbers_for_fit[1] - mode_numbers_for_fit[0]));
        delta_slopes.push_back(0.0);
      } else {
        // Cannot construct a slope; skip this term in sum
        continue;
      }
    }
  }
  const auto max_delta =
      std::max_element(delta_slopes.begin(), delta_slopes.end());
  // Scale the small factor to be a few orders of magnitude smaller than
  // the max slope delta, as SpEC does. But in case the fit happens to have
  // identically zero errors, make sure eps is still nonzero.
  const double eps = std::max(*max_delta * 1.e-3, 1.e-15);
  double num = 0.0;
  double denom = 0.0;
  for (size_t i = 0; i < slopes.size(); ++i) {
    const double one_over_denom_this_term =
        1.0 / (eps + gsl::at(delta_slopes, i));
    denom += one_over_denom_this_term;
    num += gsl::at(slopes, i) * one_over_denom_this_term;
  }
  result.convergence_rate = -num / denom;

  // Compute number of pile up modes
  // First, if the convergence rate is nearly zero, return zero pile up modes.
  // A convergence rate near zero typically means that the function is not at
  // all resolved (e.g. a step function), so the power spectrum is approximately
  // flat. If the convergence rate is approximately flat, it's not realistic to
  // attempt to distinguish whatever small, residual convergence might be
  // present vs. pile up modes. Returning zero for an approximately flat power
  // monitor also avoids dividing by approximately zero (or, in the case of
  // an exactly flat power monitor, by exactly zero).
  if (std::abs(result.convergence_rate) < 1.e-10) {
    result.number_of_pile_up_modes = 0.0;
  } else {
    double number_of_pile_up_modes = 0.0;
    for (size_t j = 2; j < n_tilde - 1; ++j) {
      const size_t j_max = std::min(n_tilde - 1, j + 4);
      mode_numbers_for_fit.resize(j_max - j + 1);
      mode_powers_for_fit.resize(j_max - j + 1);
      for (size_t i = j; i <= j_max; ++i) {
        mode_numbers_for_fit[i - j] = static_cast<double>(i);
        mode_powers_for_fit[i - j] = log_power[i];
      }
      double local_convergence_rate =
          std::numeric_limits<double>::signaling_NaN();
      if (mode_numbers_for_fit.size() > 2) {
        local_convergence_rate =
            -intrp::linear_regression(mode_numbers_for_fit, mode_powers_for_fit)
                 .slope;
      } else if (mode_numbers_for_fit.size() == 2 and
                 mode_numbers_for_fit[1] != mode_numbers_for_fit[0]) {
        local_convergence_rate =
            (mode_powers_for_fit[1] - mode_powers_for_fit[0]) /
            (mode_numbers_for_fit[1] - mode_numbers_for_fit[0]);
      } else {
        // cannot measure slope, so skip this term in sum
        continue;
      }
      const double conv_ratio =
          square(local_convergence_rate / result.convergence_rate);
      // Avoid underflow: if conv_ratio < 16, just add zero.
      // exp(-32.0*16.0) ~ 1.0e-195, which is still large enough to
      // avoid underflow.
      number_of_pile_up_modes +=
          conv_ratio < 16.0 ? exp(-32.0 * conv_ratio) : 0.0;
    }
    result.number_of_pile_up_modes = number_of_pile_up_modes;
  }
  return result;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_DIM(_, data)                                          \
  template std::array<DataVector, DIM(data)> power_monitors(              \
      const DTYPE(data) & u, const Mesh<DIM(data)>& mesh);                \
  template void power_monitors(                                           \
      const gsl::not_null<std::array<DataVector, DIM(data)>*> result,     \
      const DTYPE(data) & u, const Mesh<DIM(data)>& mesh);                \
  template std::array<double, DIM(data)> relative_truncation_error(       \
      const DTYPE(data) & tensor_component, const Mesh<DIM(data)>& mesh); \
  template std::array<double, DIM(data)> absolute_truncation_error(       \
      const DTYPE(data) & tensor_component, const Mesh<DIM(data)>& mesh);

GENERATE_INSTANTIATIONS(INSTANTIATE_DIM, (DataVector, ComplexDataVector),
                        (1, 2, 3))

#undef INSTANTIATE
#undef DIM

void accumulate_b3_tensor_component_sums(
    const gsl::not_null<DataVector*> sum_sq_radial,
    const gsl::not_null<DataVector*> counts_radial,
    const gsl::not_null<DataVector*> sum_sq_angular,
    const gsl::not_null<DataVector*> counts_angular,
    const double* const spec_buf, const size_t n_r, const size_t n_r_max,
    const int spin_weight, const std::vector<std::vector<size_t>>& offsets_by_l,
    double* const gathered, double* const modal_buf) {
  const size_t l_max = offsets_by_l.size() - 1;
  const auto abs_spin_weight = static_cast<size_t>(std::abs(spin_weight));

  for (size_t l = 0; l <= l_max; ++l) {
    if (l < abs_spin_weight) {
      // Spin-weighted spherical harmonics vanish for l < |s|; skip.
      continue;
    }
    const size_t spectral_size_l = (n_r_max - l + 2) / 2;
    const auto& offsets = offsets_by_l[l];
    const size_t n_modes_l = offsets.size();

    // Gather radial profiles for all same-l SH modes into a contiguous buffer.
    // Layout: gathered[k * n_r + i_r] for the k-th offset at this l.
    for (size_t k = 0; k < n_modes_l; ++k) {
      std::copy(spec_buf + offsets[k] * n_r,        // NOLINT
                spec_buf + (offsets[k] + 1) * n_r,  // NOLINT
                gathered + k * n_r);                // NOLINT
    }

    // Apply the Jacobi NTM for angular degree l:
    //   modal_buf[spectral_size_l x n_modes_l]
    //     = NTM_l[spectral_size_l x n_r] * gathered[n_r x n_modes_l].
    const auto& ntm =
        Spectral::nodal_to_modal_matrix<Spectral::Basis::ZernikeB3,
                                        Spectral::Quadrature::GaussRadauUpper>(
            n_r, l, n_r_max);
    dgemm_<true>('N', 'N', spectral_size_l, n_modes_l, n_r, 1.0, ntm.data(),
                 ntm.spacing(), gathered, n_r, 0.0, modal_buf, spectral_size_l);

    // Accumulate squared Jacobi coefficients into radial and angular bins.
    for (size_t k_spec = 0; k_spec < spectral_size_l; ++k_spec) {
      const size_t n_total = l + 2 * k_spec;
      const size_t radial_mode = (n_total + 1) / 2;
      for (size_t col = 0; col < n_modes_l; ++col) {
        const double coeff_sq =
            square(modal_buf[k_spec + spectral_size_l * col]);  // NOLINT
        (*sum_sq_radial)[radial_mode] += coeff_sq;
        (*counts_radial)[radial_mode] += 1.0;
        (*sum_sq_angular)[l] += coeff_sq;
        (*counts_angular)[l] += 1.0;
      }
    }
  }
}

void normalize_b3_power(const gsl::not_null<DataVector*> result,
                        const DataVector& sum_sq, const DataVector& counts) {
  result->destructive_resize(sum_sq.size());
  for (size_t i = 0; i < sum_sq.size(); ++i) {
    (*result)[i] = counts[i] == 0.0 ? 0.0 : sqrt(sum_sq[i] / counts[i]);
  }
}

}  // namespace PowerMonitors
