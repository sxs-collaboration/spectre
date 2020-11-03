// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"

#include <array>
#include <boost/math/special_functions/binomial.hpp>
#include <cmath>
#include <complex>
#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StaticCache.hpp"
#include "Utilities/TMPL.hpp"

/// \cond

namespace Spectral::Swsh {

SpinWeightedSphericalHarmonic::SpinWeightedSphericalHarmonic(
    const int spin, const size_t l, const int m) noexcept
    : spin_{spin}, l_{l}, m_{m} {
  overall_prefactor_ = 1.0;
  const double double_l = l;
  const double double_m = m;
  const double double_spin = spin;
  if (std::abs(m) > std::abs(spin)) {
    for (size_t i = 0; i < static_cast<size_t>(std::abs(m) - std::abs(spin));
         ++i) {
      const double double_i = i;
      overall_prefactor_ *= (double_l + std::abs(double_m) - double_i) /
                            (double_l - (std::abs(double_spin) + double_i));
    }
  } else if (std::abs(spin) > std::abs(m)) {
    for (size_t i = 0; i < static_cast<size_t>(std::abs(spin) - std::abs(m));
         ++i) {
      const double double_i = i;
      overall_prefactor_ *= (double_l - (std::abs(double_m) + double_i)) /
                            (double_l + std::abs(double_spin) - double_i);
    }
  }
  // if neither is greater (they are equal), then the prefactor is 1.0
  overall_prefactor_ *= (2.0 * l + 1.0) / (4.0 * M_PI);
  overall_prefactor_ = sqrt(overall_prefactor_);
  overall_prefactor_ *= (m % 2) == 0 ? 1.0 : -1.0;

  // gcc warns about the casts in ways that are impossible to satisfy
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
  if (static_cast<int>(l) < std::abs(spin)) {
    if (spin < 0) {
      r_prefactors_ =
          std::vector<double>(l + static_cast<size_t>(std::abs(spin)) + 1, 0.0);
    }
  } else {
    // the casts in the reserve are in correct order, but clang-format
    // erroneously requests a change

    // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
    r_prefactors_.reserve(static_cast<size_t>(static_cast<int>(l) - spin + 1));
    for (int r = 0; r <= (static_cast<int>(l) - spin); ++r) {
      if (r + spin - m >= 0 and static_cast<int>(l) - r + m >= 0) {
        r_prefactors_.push_back(
            boost::math::binomial_coefficient<double>(
                static_cast<size_t>(static_cast<int>(l) - spin),
                static_cast<size_t>(r)) *
            boost::math::binomial_coefficient<double>(
                static_cast<size_t>(static_cast<int>(l) + spin),
                static_cast<size_t>(spin - m + r)) *
            (((static_cast<int>(l) - r - spin) % 2) == 0 ? 1.0 : -1.0));
      } else {
        r_prefactors_.push_back(0.0);
      }
    }
  }
#pragma GCC diagnostic pop
}

void SpinWeightedSphericalHarmonic::evaluate(
    const gsl::not_null<ComplexDataVector*> result, const DataVector& theta,
    const DataVector& phi, const DataVector& sin_theta_over_2,
    const DataVector& cos_theta_over_2) const noexcept {
  result->destructive_resize(theta.size());
  *result = 0.0;
  DataVector theta_factor{theta.size()};
  for (int r = 0; r <= (static_cast<int>(l_) - spin_); ++r) {
    if (2 * static_cast<int>(l_) > 2 * r + spin_ - m_) {
      theta_factor = pow(cos_theta_over_2, 2 * r + spin_ - m_) *
                     pow(sin_theta_over_2,
                         2 * static_cast<int>(l_) - (2 * r + spin_ - m_));
    } else if (2 * static_cast<int>(l_) < 2 * r + spin_ - m_) {
      theta_factor = pow(cos_theta_over_2 / sin_theta_over_2,
                         2 * r + spin_ - m_ - 2 * static_cast<int>(l_)) *
                     pow(cos_theta_over_2, 2 * l_);
    } else {
      theta_factor = pow(cos_theta_over_2, 2 * l_);
    }
    *result += gsl::at(r_prefactors_, r) * theta_factor;
  }
  // optimization note: this has not been compared with a complex `exp`
  // function, and it is not obvious which should be faster in practice.
  *result *=
      overall_prefactor_ *
      (std::complex<double>(1.0, 0.0) * cos(static_cast<double>(m_) * phi) +
       std::complex<double>(0.0, 1.0) * sin(static_cast<double>(m_) * phi));
}

ComplexDataVector SpinWeightedSphericalHarmonic::evaluate(
    const DataVector& theta, const DataVector& phi,
    const DataVector& sin_theta_over_2,
    const DataVector& cos_theta_over_2) const noexcept {
  ComplexDataVector result{theta.size(), 0.0};
  evaluate(make_not_null(&result), theta, phi, sin_theta_over_2,
           cos_theta_over_2);
  return result;
}

std::complex<double> SpinWeightedSphericalHarmonic::evaluate(
    const double theta, const double phi) const noexcept {
  std::complex<double> accumulator = 0.0;
  const double cos_theta_over_two = cos(0.5 * theta);
  const double sin_theta_over_two = sin(0.5 * theta);
  double theta_factor = std::numeric_limits<double>::signaling_NaN();
  for (int r = 0; r <= (static_cast<int>(l_) - spin_); ++r) {
    if (2 * static_cast<int>(l_) > 2 * r + spin_ - m_) {
      theta_factor = pow(cos_theta_over_two, 2 * r + spin_ - m_) *
                     pow(sin_theta_over_two,
                         2 * static_cast<int>(l_) - (2 * r + spin_ - m_));
    } else if (2 * static_cast<int>(l_) < 2 * r + spin_ - m_) {
      theta_factor = pow(cos_theta_over_two / sin_theta_over_two,
                         2 * r + spin_ - m_ - 2 * static_cast<int>(l_)) *
                     pow(cos_theta_over_two, 2 * l_);
    } else {
      theta_factor = pow(cos_theta_over_two, 2 * l_);
    }
    accumulator += gsl::at(r_prefactors_, r) *
                   std::complex<double>(cos(static_cast<double>(m_) * phi),
                                        sin(static_cast<double>(m_) * phi)) *
                   theta_factor;
  }
  accumulator *= overall_prefactor_;
  return accumulator;
}

void SpinWeightedSphericalHarmonic::pup(PUP::er& p) noexcept {
  p | spin_;
  p | l_;
  p | m_;
  p | overall_prefactor_;
  p | r_prefactors_;
}

// A function for indexing a desired element in one of the caches stored
// in `ClenshawRecurrenceConstants`.
// Useful for accessing the `beta_constant`, `alpha_constant`, or
// `alpha_prefactor` recurrence constants.
size_t clenshaw_cache_index(const size_t l_max, const int spin, const int l,
                            const int m) noexcept {
  return goldberg_mode_index(l_max - 2, static_cast<size_t>(l - 2), m) -
         static_cast<size_t>(square(spin));
}

// see the detailed doxygen for `SwshInterpolator` for full mathematical details
// of the recurrence constant computations
template <int Spin>
struct ClenshawRecurrenceConstants {
  ClenshawRecurrenceConstants() = default;

  explicit ClenshawRecurrenceConstants(size_t l_max) noexcept
      : alpha_prefactor{square(l_max + 1) -
                        square(static_cast<size_t>(std::abs(Spin)))},
        alpha_constant{square(l_max + 1) -
                       square(static_cast<size_t>(std::abs(Spin)))},
        beta_constant{square(l_max + 1) -
                      square(static_cast<size_t>(std::abs(Spin)))},
        harmonic_at_l_min_prefactors{2 * l_max + 1},
        harmonic_at_l_min_plus_one_recurrence_prefactors{2 * l_max + 1},
        harmonic_m_recurrence_prefactors{2 * l_max + 1} {
    ASSERT(static_cast<int>(l_max) > Spin,
           "l_max must be greater than the spin-weight when computing "
           "ClenshawRecurrenceConstants");
    double l_plus_k = std::numeric_limits<double>::signaling_NaN();
    double l_min_plus_k = std::numeric_limits<double>::signaling_NaN();
    double a = std::numeric_limits<double>::signaling_NaN();
    double b = std::numeric_limits<double>::signaling_NaN();
    int l_min = 0;
    double prefactor_accumulator = std::numeric_limits<double>::signaling_NaN();
    lambda.reserve(2 * l_max + 1);
    for (int m = -static_cast<int>(l_max); m <= static_cast<int>(l_max); ++m) {
      a = static_cast<double>(std::abs(Spin + m));
      b = static_cast<double>(std::abs(Spin - m));
      l_min = std::max(std::abs(m), std::abs(Spin));

      // gcc warns about an optimization that doesn't work if we overflow. None
      // of this will overflow provided l_max is not unreasonably high (less
      // than ~10^5 will not overflow).
      for (int l = l_min + 2; l <= static_cast<int>(l_max); ++l) {
        // start caching at 2 greater than the l_min for a given m. Those are
        // the last terms needed by (descending) Clenshaw sum.
        l_plus_k = static_cast<double>(l) - 0.5 * (a + b);
        alpha_prefactor[clenshaw_cache_index(l_max, Spin, l, m)] =
            0.5 * sqrt((2.0 * l + 1.0) * (2.0 * l - 1.0) /
                       (l_plus_k * (l_plus_k + a + b) * (l_plus_k + a) *
                        (l_plus_k + b)));
        alpha_constant[clenshaw_cache_index(l_max, Spin, l, m)] =
            alpha_prefactor[clenshaw_cache_index(l_max, Spin, l, m)] *
            ((square(a) - square(b)) / (2.0 * l - 2.0));
        alpha_prefactor[clenshaw_cache_index(l_max, Spin, l, m)] *= (2.0 * l);
        beta_constant[clenshaw_cache_index(l_max, Spin, l, m)] =
            -sqrt((2.0 * l + 1.0) * (l_plus_k + a - 1.0) *
                  (l_plus_k + b - 1.0) * (l_plus_k - 1.0) *
                  (l_plus_k + a + b - 1.0) /
                  ((2.0 * l - 3.0) * l_plus_k * (l_plus_k + a + b) *
                   (l_plus_k + a) * (l_plus_k + b))) *
            (2.0 * l) / (2.0 * l - 2.0);
      }
      lambda.push_back(Spin >= -m ? 0 : Spin + m);

      // pre-compute the prefactors for the lowest order harmonics for each m
      prefactor_accumulator = 1.0;
      l_min_plus_k = -0.5 * (std::abs(Spin + m) + b) + l_min;
      for (int i = 1; i <= b; ++i) {
        if (l_min_plus_k + a + i > 0.0) {
          prefactor_accumulator *= static_cast<double>(l_min_plus_k + a + i);
        }
        if (l_min_plus_k + i > 0.0) {
          prefactor_accumulator /= static_cast<double>(l_min_plus_k + i);
        }
      }
      prefactor_accumulator =
          sqrt(prefactor_accumulator * (2.0 * l_min + 1.0) / (4.0 * M_PI));
      prefactor_accumulator *=
          ((m + gsl::at(lambda, m + static_cast<int>(l_max))) % 2) == 0 ? 1.0
                                                                        : -1.0;
      // this is the right order of the casts, other orders give the wrong
      // answer

      // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
      harmonic_at_l_min_prefactors[static_cast<size_t>(
          m + static_cast<int>(l_max))] = prefactor_accumulator;

      // pre-compute the prefactors for bootstrapping the second-to-lowest order
      // harmonics for each m

      // this is the right order of the casts, other orders give the wrong
      // answer

      // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
      harmonic_at_l_min_plus_one_recurrence_prefactors[static_cast<size_t>(
          m + static_cast<int>(l_max))] =
          sqrt((2.0 * (l_min) + 3.0) * (l_min_plus_k + 1.0) *
               (l_min_plus_k + a + b + 1.0) /
               ((2.0 * (l_min) + 1.0) * (l_min_plus_k + a + 1.0) *
                (l_min_plus_k + b + 1.0)));
    }
    // separate loop because we'll need the lambdas entirely populated for this
    // set of prefactors
    int lambda_difference = 0;
    for (int m = -static_cast<int>(l_max); m <= static_cast<int>(l_max); ++m) {
      if (std::abs(m) > std::abs(Spin)) {
        l_min = std::max(std::abs(m), std::abs(Spin));
        a = std::abs(Spin + m);
        b = std::abs(Spin - m);
        l_min_plus_k = -0.5 * (std::abs(Spin + m) + b) + l_min;

        prefactor_accumulator =
            sqrt((2.0 * std::abs(m) + 1.0) * (l_min_plus_k + a + b - 1.0) *
                 (l_min_plus_k + a + b) /
                 ((2.0 * std::abs(m) - 1.0) * (l_min_plus_k + a) *
                  (l_min_plus_k + b)));
        // there is an extra `1` in these expressions to account for the -1 out
        // front of the recurrence relations.
        lambda_difference = gsl::at(lambda, m + static_cast<int>(l_max)) + 1;
        if (m > 0) {
          lambda_difference -= gsl::at(lambda, m - 1 + static_cast<int>(l_max));
        } else {
          lambda_difference -= gsl::at(lambda, m + 1 + static_cast<int>(l_max));
        }
        prefactor_accumulator *= lambda_difference % 2 == 0 ? 1.0 : -1.0;
        // this is the right order of the casts, other orders give the wrong
        // answer

        // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
        harmonic_m_recurrence_prefactors[static_cast<size_t>(
            m + static_cast<int>(l_max))] = prefactor_accumulator;
      }
    }
  }

  /// Serialization for Charm++.
  void pup(PUP::er& p) noexcept {  // NOLINT
    p | alpha_prefactor;
    p | alpha_constant;
    p | beta_constant;
    p | lambda;
    p | harmonic_at_l_min_prefactors;
    p | harmonic_at_l_min_plus_one_recurrence_prefactors;
    p | harmonic_m_recurrence_prefactors;
  }

  // Tables are stored in a triangular Goldberg style
  DataVector alpha_prefactor;
  DataVector alpha_constant;
  DataVector beta_constant;
  std::vector<int> lambda;
  DataVector harmonic_at_l_min_prefactors;
  DataVector harmonic_at_l_min_plus_one_recurrence_prefactors;
  DataVector harmonic_m_recurrence_prefactors;
};

// A lazy static cache interface for retrieving `ClenshawRecurrenceConstants`.
template <int Spin>
const ClenshawRecurrenceConstants<Spin>& cached_clenshaw_factors(
    const size_t l_max) noexcept {
  const static auto lazy_clenshaw_cache =
      make_static_cache<CacheRange<0, collocation_maximum_l_max>>(
          [](const size_t local_l_max) noexcept {
            return ClenshawRecurrenceConstants<Spin>{local_l_max};
          });
  return lazy_clenshaw_cache(l_max);
}

SwshInterpolator::SwshInterpolator(const DataVector& theta,
                                   const DataVector& phi,
                                   const size_t l_max) noexcept
    : l_max_{l_max},
      raw_libsharp_coefficient_buffer_{
          size_of_libsharp_coefficient_vector(l_max)},
      raw_goldberg_coefficient_buffer_{square(l_max + 1)} {
  cos_m_phi_ = std::vector<DataVector>(l_max + 1);
  sin_m_phi_ = std::vector<DataVector>(l_max + 1);
  cos_theta_ = cos(theta);
  sin_theta_ = sin(theta);
  cos_theta_over_two_ = cos(0.5 * theta);
  sin_theta_over_two_ = sin(0.5 * theta);
  // evaluate cos(m phi) and sin(m phi) via recurrence
  cos_m_phi_[0] = DataVector{phi.size(), 1.0};
  sin_m_phi_[0] = DataVector{phi.size(), 0.0};
  const DataVector m_phi_beta = sin(phi);
  const DataVector m_phi_alpha = 2.0 * square(sin(0.5 * phi));
  for (size_t m = 1; m <= l_max; ++m) {
    cos_m_phi_[m] = cos_m_phi_[m - 1] - (m_phi_alpha * cos_m_phi_[m - 1] +
                                         m_phi_beta * sin_m_phi_[m - 1]);
    sin_m_phi_[m] = sin_m_phi_[m - 1] - (m_phi_alpha * sin_m_phi_[m - 1] -
                                         m_phi_beta * cos_m_phi_[m - 1]);
  }
}

template <int Spin>
void SwshInterpolator::interpolate(
    const gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> interpolated,
    const SpinWeighted<ComplexModalVector, Spin>& goldberg_modes)
    const noexcept {
  ASSERT(l_max_ != 0,
         "Attempting to perform interpolation with a default-constructed "
         "SwshInterpolator. The SwshInterpolator must be constructed with the "
         "angular coordinates to perform interpolation.");
  interpolated->destructive_resize(cos_theta_.size());
  interpolated->data() = 0.0;

  // used only if s=0;
  SpinWeighted<ComplexDataVector, Spin> cached_base_harmonic;

  // used during both recurrence legs
  SpinWeighted<ComplexDataVector, Spin> current_cached_harmonic;
  SpinWeighted<ComplexDataVector, Spin> current_cached_harmonic_l_plus_one;

  // perform the Clenshaw sums over positive m >= 0.
  for (int m = 0; m <= static_cast<int>(l_max_); ++m) {
    if (std::abs(Spin) >= std::abs(m)) {
      direct_evaluation_swsh_at_l_min(make_not_null(&current_cached_harmonic),
                                      m);
      evaluate_swsh_at_l_min_plus_one(
          make_not_null(&current_cached_harmonic_l_plus_one),
          current_cached_harmonic, m);
    } else {
      evaluate_swsh_m_recurrence_at_l_min(
          make_not_null(&current_cached_harmonic), m);
      evaluate_swsh_at_l_min_plus_one(
          make_not_null(&current_cached_harmonic_l_plus_one),
          current_cached_harmonic, m);
    }
    if (Spin == 0 and m == 0) {
      cached_base_harmonic = current_cached_harmonic;
    }
    clenshaw_sum(interpolated, current_cached_harmonic,
                 current_cached_harmonic_l_plus_one, goldberg_modes, m);
  }
  // perform the Clenshaw sums over m < 0.
  for (int m = -1; m >= -static_cast<int>(l_max_); --m) {
    // initialize the recurrence for negative m
    if (m == -1 and Spin == 0) {
      current_cached_harmonic = cached_base_harmonic;
    }
    if (std::abs(Spin) >= std::abs(m)) {
      direct_evaluation_swsh_at_l_min(make_not_null(&current_cached_harmonic),
                                      m);
      evaluate_swsh_at_l_min_plus_one(
          make_not_null(&current_cached_harmonic_l_plus_one),
          current_cached_harmonic, m);
    } else {
      evaluate_swsh_m_recurrence_at_l_min(
          make_not_null(&current_cached_harmonic), m);
      evaluate_swsh_at_l_min_plus_one(
          make_not_null(&current_cached_harmonic_l_plus_one),
          current_cached_harmonic, m);
    }
    clenshaw_sum(interpolated, current_cached_harmonic,
                 current_cached_harmonic_l_plus_one, goldberg_modes, m);
  }
}

template <int Spin>
void SwshInterpolator::interpolate(
    const gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> interpolated,
    const SpinWeighted<ComplexDataVector, Spin>& libsharp_collocation)
    const noexcept {
  ASSERT(l_max_ != 0,
         "Attempting to perform interpolation with a default-constructed "
         "SwshInterpolator. The SwshInterpolator must be constructed with the "
         "angular coordinates to perform interpolation.");
  SpinWeighted<ComplexModalVector, Spin> libsharp_modes;
  // this function is 'const', but modifies the internal buffer. The reason to
  // allow it to be 'const' anyways is that no interface makes any assumption
  // about the starting state of the internal buffer; it is kept exclusively to
  // save allocations.
  libsharp_modes.set_data_ref(raw_libsharp_coefficient_buffer_.data(),
                              raw_libsharp_coefficient_buffer_.size());
  swsh_transform(l_max_, 1, make_not_null(&libsharp_modes),
                 libsharp_collocation);
  SpinWeighted<ComplexModalVector, Spin> goldberg_modes;
  libsharp_to_goldberg_modes(make_not_null(&goldberg_modes), libsharp_modes,
                             l_max_);
  interpolate(interpolated, goldberg_modes);
}

template <int Spin>
void SwshInterpolator::direct_evaluation_swsh_at_l_min(
    const gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> harmonic,
    const int m) const noexcept {
  ASSERT(l_max_ != 0,
         "Attempting to perform spin-weighted evaluation with a "
         "default-constructed SwshInterpolator. The SwshInterpolator must be "
         "constructed with the angular coordinates to perform function "
         "evaluation.");
  const auto& clenshaw_factors = cached_clenshaw_factors<Spin>(l_max_);
  // for this evaluation, we don't worry about recurrence because it will only
  // be called for m between -s and +s, and s should always be small. In
  // principle, it is probably true that a more complicated recurrence exists
  // for this case, but would require a bit of derivation work
  harmonic->data() =
      // clang-tidy: this is the right order of the casts, other orders give the
      // wrong answer

      // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
      clenshaw_factors.harmonic_at_l_min_prefactors[static_cast<size_t>(
          m + static_cast<int>(l_max_))] *
      (std::complex<double>(1.0, 0.0) * gsl::at(cos_m_phi_, std::abs(m)) +
       std::complex<double>(0.0, 1.0) * (m >= 0 ? 1.0 : -1.0) *
           gsl::at(sin_m_phi_, std::abs(m))) *
      pow(sin_theta_over_two_, static_cast<size_t>(std::abs(Spin + m))) *
      pow(cos_theta_over_two_, static_cast<size_t>(std::abs(Spin - m)));
}

template <int Spin>
void SwshInterpolator::evaluate_swsh_at_l_min_plus_one(
    const gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> harmonic,
    const SpinWeighted<ComplexDataVector, Spin>& harmonic_at_l_min,
    const int m) const noexcept {
  ASSERT(l_max_ != 0,
         "Attempting to perform spin-weighted evaluation with a "
         "default-constructed SwshInterpolator. The SwshInterpolator must be "
         "constructed with the angular coordinates to perform function "
         "evaluation.");
  const auto& clenshaw_factors = cached_clenshaw_factors<Spin>(l_max_);
  const double a = std::abs(Spin + m);
  const double b = std::abs(Spin - m);
  harmonic->data() =
      clenshaw_factors
          // clang-tidy: this is the right order of the casts, other orders give
          // the wrong answer

          // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
          .harmonic_at_l_min_plus_one_recurrence_prefactors[static_cast<size_t>(
              m + static_cast<int>(l_max_))] *
      harmonic_at_l_min.data() *
      (a + 1.0 + 0.5 * (a + b + 2.0) * (cos_theta_ - 1.0));
}

template <int Spin>
void SwshInterpolator::evaluate_swsh_m_recurrence_at_l_min(
    const gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> harmonic,
    const int m) const noexcept {
  ASSERT(l_max_ != 0,
         "Attempting to perform spin-weighted evaluation with a "
         "default-constructed SwshInterpolator. The SwshInterpolator must be "
         "constructed with the angular coordinates to perform function "
         "evaluation.");
  const auto& clenshaw_factors = cached_clenshaw_factors<Spin>(l_max_);
  harmonic->data() =
      // this is the right order of the casts, other orders give the wrong
      // answer

      // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
      clenshaw_factors.harmonic_m_recurrence_prefactors[static_cast<size_t>(
          m + static_cast<int>(l_max_))] *
      (sin_theta_ / 2.0) * harmonic->data();
  if (m > 0) {
    harmonic->data() *= (std::complex<double>(1.0, 0.0) * cos_m_phi_[1] +
                         std::complex<double>(0.0, 1.0) * sin_m_phi_[1]);
  } else {
    harmonic->data() *= (std::complex<double>(1.0, 0.0) * cos_m_phi_[1] -
                         std::complex<double>(0.0, 1.0) * sin_m_phi_[1]);
  }
}

template <int Spin>
void SwshInterpolator::clenshaw_sum(
    const gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> interpolation,
    const SpinWeighted<ComplexDataVector, Spin>& l_min_harmonic,
    const SpinWeighted<ComplexDataVector, Spin>& l_min_plus_one_harmonic,
    const SpinWeighted<ComplexModalVector, Spin>& goldberg_modes,
    const int m) const noexcept {
  ASSERT(l_max_ != 0,
         "Attempting to perform spin-weighted evaluation with a "
         "default-constructed SwshInterpolator. The SwshInterpolator must be "
         "constructed with the angular coordinates to perform function "
         "evaluation.");
  // Since we need various combinations of the three-term recurrence constants
  // up to two orders higher, we write recurrence results to a cyclic
  // three-element cache
  std::array<ComplexDataVector, 3> recurrence_cache;
  recurrence_cache[2] = ComplexDataVector{interpolation->size(), 0.0};
  recurrence_cache[1] = ComplexDataVector{interpolation->size(), 0.0};
  recurrence_cache[0] = ComplexDataVector{interpolation->size(), 0.0};
  const auto& clenshaw_factors = cached_clenshaw_factors<Spin>(l_max_);

  for (auto l = static_cast<int>(l_max_);
       l > std::max(std::abs(Spin), std::abs(m)); l--) {
    // We want to define some cache_offset so that we can index the three
    // elements of recurrence_cache with indices cache_offset%3,
    // (cache_offset+1)%3, and (cache_offset+2)%3, and so that cache_offset
    // decreases by one on each iteration. The "obvious" way to do this is to
    // choose cache_offset = l - l_max, so that cache_offset starts at zero and
    // then decreases each iteration. However, this gives negative values of
    // cache_offset, and C++ modular arithmetic doesn't behave the way we'd want
    // for that process at negative values. But note that adding any multiple of
    // 3 to the "obvious" value of cache_offset will give identical indexing,
    // and choosing this multiple of 3 large enough (i.e. larger than l_max)
    // guarantees that the new cache_offset is positive for all l. So we choose
    // to add 3*l_max to the "obvious" value of cache_offset; in other words we
    // define cache_offset = l + 2 l_max.
    // In future, if this trick needs to be re-implemented in another use-case,
    // it should instead be factored out into a separate rotating cache utility.
    const int cache_offset = (l + 2 * static_cast<int>(l_max_));
    // gcc warns about the casts in ways that are impossible to satisfy
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
    gsl::at(recurrence_cache, (cache_offset) % 3) =
        goldberg_modes.data()[square(static_cast<size_t>(l)) +
                              static_cast<size_t>(l + m)];
    if (l < static_cast<int>(l_max_)) {
      gsl::at(recurrence_cache, (cache_offset) % 3) +=
          (clenshaw_factors
               .alpha_constant[clenshaw_cache_index(l_max_, Spin, l + 1, m)] +
           cos_theta_ * clenshaw_factors.alpha_prefactor[clenshaw_cache_index(
                            l_max_, Spin, l + 1, m)]) *
          gsl::at(recurrence_cache, (cache_offset + 1) % 3);
    }
    if (l < static_cast<int>(l_max_) - 1) {
      gsl::at(recurrence_cache, (cache_offset) % 3) +=
          clenshaw_factors
              .beta_constant[clenshaw_cache_index(l_max_, Spin, l + 2, m)] *
          gsl::at(recurrence_cache, (cache_offset + 2) % 3);
    }
  }
  const int l_min = std::max(std::abs(Spin), std::abs(m));
  const int cache_offset = (l_min + 2 * static_cast<int>(l_max_));

  if (l_max_ >=
      static_cast<size_t>(std::max(std::abs(Spin), std::abs(m))) + 2) {
    *interpolation +=
        l_min_harmonic *
            goldberg_modes.data()[square(static_cast<size_t>(l_min)) +
                                  static_cast<size_t>(l_min + m)] +
        l_min_plus_one_harmonic *
            gsl::at(recurrence_cache, (cache_offset + 1) % 3) +
        l_min_harmonic * gsl::at(recurrence_cache, (cache_offset + 2) % 3) *
            clenshaw_factors.beta_constant[clenshaw_cache_index(
                l_max_, Spin, std::max(std::abs(Spin), std::abs(m)) + 2, m)];
  } else {
    *interpolation +=
        l_min_harmonic *
            goldberg_modes.data()[square(static_cast<size_t>(l_min)) +
                                  static_cast<size_t>(l_min + m)] +
        l_min_plus_one_harmonic *
            gsl::at(recurrence_cache, (cache_offset + 1) % 3);
  }
#pragma GCC diagnostic pop
}

void SwshInterpolator::pup(PUP::er& p) noexcept {
  p | l_max_;
  p | cos_theta_;
  p | sin_theta_;
  p | cos_theta_over_two_;
  p | sin_theta_over_two_;
  p | sin_m_phi_;
  p | cos_m_phi_;
  p | raw_libsharp_coefficient_buffer_;
  p | raw_goldberg_coefficient_buffer_;
}

#define GET_SPIN(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INTERPOLATION_INSTANTIATION(r, data)                                  \
  template struct ClenshawRecurrenceConstants<GET_SPIN(data)>;                \
  template const ClenshawRecurrenceConstants<GET_SPIN(data)>&                 \
  cached_clenshaw_factors<GET_SPIN(data)>(const size_t l_max) noexcept;       \
  template void SwshInterpolator::interpolate<GET_SPIN(data)>(                \
      const gsl::not_null<SpinWeighted<ComplexDataVector, GET_SPIN(data)>*>   \
          interpolated,                                                       \
      const SpinWeighted<ComplexModalVector, GET_SPIN(data)>& goldberg_modes) \
      const noexcept;                                                         \
  template void SwshInterpolator::interpolate<GET_SPIN(data)>(                \
      const gsl::not_null<SpinWeighted<ComplexDataVector, GET_SPIN(data)>*>   \
          interpolated,                                                       \
      const SpinWeighted<ComplexDataVector, GET_SPIN(data)>&                  \
          libsharp_collocation) const noexcept;                               \
  template void                                                               \
  SwshInterpolator::direct_evaluation_swsh_at_l_min<GET_SPIN(data)>(          \
      const gsl::not_null<SpinWeighted<ComplexDataVector, GET_SPIN(data)>*>   \
          harmonic,                                                           \
      const int m) const noexcept;                                            \
  template void                                                               \
  SwshInterpolator::evaluate_swsh_at_l_min_plus_one<GET_SPIN(data)>(          \
      const gsl::not_null<SpinWeighted<ComplexDataVector, GET_SPIN(data)>*>   \
          harmonic,                                                           \
      const SpinWeighted<ComplexDataVector, GET_SPIN(data)>&                  \
          harmonic_at_l_min,                                                  \
      const int m) const noexcept;                                            \
  template void                                                               \
  SwshInterpolator::evaluate_swsh_m_recurrence_at_l_min<GET_SPIN(data)>(      \
      const gsl::not_null<SpinWeighted<ComplexDataVector, GET_SPIN(data)>*>   \
          harmonic,                                                           \
      const int m) const noexcept;                                            \
  template void SwshInterpolator::clenshaw_sum<GET_SPIN(data)>(               \
      const gsl::not_null<SpinWeighted<ComplexDataVector, GET_SPIN(data)>*>   \
          interpolation,                                                      \
      const SpinWeighted<ComplexDataVector, GET_SPIN(data)>& l_min_harmonic,  \
      const SpinWeighted<ComplexDataVector, GET_SPIN(data)>&                  \
          l_min_plus_one_harmonic,                                            \
      const SpinWeighted<ComplexModalVector, GET_SPIN(data)>& goldberg_modes, \
      const int m) const noexcept;

GENERATE_INSTANTIATIONS(INTERPOLATION_INSTANTIATION, (-2, -1, 0, 1, 2))

#undef INTERPOLATION_INSTANTIATION
#undef GET_SPIN

}  // namespace Spectral::Swsh
/// \endcond
