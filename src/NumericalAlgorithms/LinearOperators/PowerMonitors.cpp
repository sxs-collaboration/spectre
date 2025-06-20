// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/PowerMonitors.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "NumericalAlgorithms/Interpolation/LinearRegression.hpp"
#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
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
  }
}

template <typename VectorType, size_t Dim>
std::array<DataVector, Dim> power_monitors(const VectorType& u,
                                           const Mesh<Dim>& mesh) {
  std::array<DataVector, Dim> result{};
  power_monitors(make_not_null(&result), u, mesh);
  return result;
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

}  // namespace PowerMonitors
