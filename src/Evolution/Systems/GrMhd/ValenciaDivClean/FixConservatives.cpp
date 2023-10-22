// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/ExtractPoint.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/Simd/Simd.hpp"
#include "Utilities/TMPL.hpp"

namespace {

// This class codes Eq. (B.34), rewritten as a standard-form
// polynomial in (W - 1) or (W - ((tau/D) - (B^2/D) + 1)) for better
// numerical behavior.
//
// If lower_bound is 0, the implemented function is
// ((B^2/D) / 2 - (tau/D)) (1 + 2 (B^2/D) mu^2 + (B^2/D)^2 mu^2)
// + (W-1) (2 ((B^2/D) - (tau/D)) (1 + (B^2/D) mu^2) + (B^2/D) mu^2 + 1)
// + (W-1)^2 ((B^2/D) - (tau/D) + 3/2 (B^2/D) mu^2 + 2)
// + (W-1)^3
// where mu^2 = (B.S)^2 / (B^2 S^2).  A nice property of this form is
// that, because we've already modified tau to satisfy B.39, its value
// at W-1 = 0 is guaranteed to be negative, even in the presence of
// roundoff error.
//
// If lower_bound is LB = ((tau/D) - (B^2/D)), the implemented function is
// - 1/2 (B^2/D) (1 + mu^2 (tau/D) ((tau/D) + 2))
// + (W-[LB+1]) (1 + (B^2/D) mu^2 + LB ((B^2/D) mu^2 + LB + 2))
// + (W-[LB+1])^2 (2 LB + 3/2 (B^2/D) mu^2 + 2)
// + (W-[LB+1])^3
// This form is only used if LB > 0, so, as in the previous case, it
// is guaranteed to be negative at W-[LB+1] = 0.
template <typename T>
class FunctionOfLorentzFactor {
 public:
  FunctionOfLorentzFactor(const T b_squared_over_d, const T tau_over_d,
                          [[maybe_unused]] const T normalized_s_dot_b,
                          const T lower_bound) {
    const auto mask = lower_bound == 0.0;
    const auto zero_bound_values =
        zero_bound(b_squared_over_d, tau_over_d, normalized_s_dot_b);
    const auto nonzero_bound_values = nonzero_bound(
        b_squared_over_d, tau_over_d, normalized_s_dot_b, lower_bound);
    for (size_t i = 0; i < 4; ++i) {
      gsl::at(coefficients_, i) =
          simd::select(mask, gsl::at(zero_bound_values, i),
                       gsl::at(nonzero_bound_values, i));
    }
  }

  T operator()(const T excess_lorentz_factor) const {
    return evaluate_polynomial(coefficients_, excess_lorentz_factor);
  }

 private:
  static std::array<T, 4> zero_bound(const T& b_squared_over_d,
                                     const T& tau_over_d,
                                     const T& normalized_s_dot_b) {
    return std::array{
        (0.5 * b_squared_over_d - tau_over_d) *
            (square(normalized_s_dot_b) * b_squared_over_d *
                 (b_squared_over_d + 2.0) +
             1.0),
        2.0 * (square(normalized_s_dot_b) * b_squared_over_d + 1.0) *
                (b_squared_over_d - tau_over_d) +
            b_squared_over_d * square(normalized_s_dot_b) + 1.0,
        b_squared_over_d - tau_over_d +
            1.5 * square(normalized_s_dot_b) * b_squared_over_d + 2.0,
        T(1.0)};
  }

  static std::array<T, 4> nonzero_bound(const T& b_squared_over_d,
                                        const T& tau_over_d,
                                        const T& normalized_s_dot_b,
                                        const T& lower_bound) {
    return std::array{
        -0.5 * b_squared_over_d *
            (1.0 +
             square(normalized_s_dot_b) * tau_over_d * (tau_over_d + 2.0)),
        1.0 + square(normalized_s_dot_b) * b_squared_over_d +
            lower_bound * (2.0 + square(normalized_s_dot_b) * b_squared_over_d +
                           lower_bound),
        2.0 + 1.5 * square(normalized_s_dot_b) * b_squared_over_d +
            2.0 * lower_bound,
        T(1.0)};
  }

  std::array<T, 4> coefficients_;
};
template <typename T>
FunctionOfLorentzFactor(T b_squared_over_d, T tau_over_d, T normalized_s_dot_b,
                        T lower_bound) -> FunctionOfLorentzFactor<T>;
}  // namespace

namespace grmhd::ValenciaDivClean {
FixConservatives::FixConservatives(
    const double minimum_rest_mass_density_times_lorentz_factor,
    const double rest_mass_density_times_lorentz_factor_cutoff,
    double minimum_electron_fraction, double electron_fraction_cutoff,
    const double safety_factor_for_magnetic_field,
    const double safety_factor_for_momentum_density,
    const double safety_factor_for_momentum_density_cutoff_d,
    const double safety_factor_for_momentum_density_slope, const bool enable,
    const Options::Context& context)
    : minimum_rest_mass_density_times_lorentz_factor_(
          minimum_rest_mass_density_times_lorentz_factor),
      rest_mass_density_times_lorentz_factor_cutoff_(
          rest_mass_density_times_lorentz_factor_cutoff),
      minimum_electron_fraction_(minimum_electron_fraction),
      electron_fraction_cutoff_(electron_fraction_cutoff),
      one_minus_safety_factor_for_magnetic_field_(
          1.0 - safety_factor_for_magnetic_field),
      one_minus_safety_factor_for_momentum_density_(
          1.0 - safety_factor_for_momentum_density),
      safety_factor_for_momentum_density_cutoff_d_(
          safety_factor_for_momentum_density_cutoff_d),
      safety_factor_for_momentum_density_slope_(
          safety_factor_for_momentum_density_slope),
      enable_(enable) {
  if (minimum_rest_mass_density_times_lorentz_factor_ >
      rest_mass_density_times_lorentz_factor_cutoff_) {
    PARSE_ERROR(context,
                "The minimum value of D (a.k.a. rest mass density times "
                "Lorentz factor) ("
                    << minimum_rest_mass_density_times_lorentz_factor_
                    << ") must be less than or equal to the cutoff value of D ("
                    << rest_mass_density_times_lorentz_factor_cutoff_ << ')');
  }
  if (minimum_electron_fraction_ > electron_fraction_cutoff_) {
    PARSE_ERROR(context,
                "The minimum value of electron fraction Y_e ("
                    << minimum_electron_fraction_
                    << ") must be less than or equal to the cutoff value ("
                    << electron_fraction_cutoff_ << ')');
  }
  if (safety_factor_for_momentum_density_slope_ < 0.0) {
    PARSE_ERROR(context,
                "The option SafetyFactorForSSlope cannot be negative.");
  }
  if (safety_factor_for_momentum_density +
          safety_factor_for_momentum_density_slope_ *
              log10(safety_factor_for_momentum_density_cutoff_d /
                    minimum_rest_mass_density_times_lorentz_factor_) >=
      1.0) {
    PARSE_ERROR(context,
                "SafetyFactorForSSlope too large, will lead to unphysical "
                "cutoff at low density");
  }
}

// NOLINTNEXTLINE(google-runtime-references)
void FixConservatives::pup(PUP::er& p) {
  p | minimum_rest_mass_density_times_lorentz_factor_;
  p | rest_mass_density_times_lorentz_factor_cutoff_;
  p | minimum_electron_fraction_;
  p | electron_fraction_cutoff_;
  p | one_minus_safety_factor_for_magnetic_field_;
  p | one_minus_safety_factor_for_momentum_density_;
  p | safety_factor_for_momentum_density_cutoff_d_;
  p | safety_factor_for_momentum_density_slope_;
  p | enable_;
}

// WARNING!
// Notation of Foucart is not that of SpECTRE
// SpECTRE           Foucart
// {\tilde D}        \rho_*
// {\tilde \tau}     \tau
// {\tilde S}_k      S_k
// {\tilde B}^k      B^k \sqrt{g}
// \rho              \rho_0
// \gamma_{mn}       g_{mn}
bool FixConservatives::operator()(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric) const {
  bool needed_fixing = false;
  if (not enable_) {
    return needed_fixing;
  }
  const size_t size = get<0>(tilde_b).size();
  Variables<tmpl::list<::Tags::TempScalar<1>, ::Tags::TempScalar<2>,
                       ::Tags::TempScalar<3>>>
      temp_buffer(size);

  Scalar<DataVector>& tilde_b_squared = get<::Tags::TempScalar<1>>(temp_buffer);
  dot_product(make_not_null(&tilde_b_squared), tilde_b, tilde_b,
              spatial_metric);

  Scalar<DataVector>& tilde_s_squared = get<::Tags::TempScalar<2>>(temp_buffer);
  dot_product(make_not_null(&tilde_s_squared), *tilde_s, *tilde_s,
              inv_spatial_metric);

  Scalar<DataVector>& tilde_s_dot_tilde_b =
      get<::Tags::TempScalar<3>>(temp_buffer);
  dot_product(make_not_null(&tilde_s_dot_tilde_b), *tilde_s, tilde_b);

  const double one_over_one_minus_safety_factor_for_magnetic_field =
      1.0 / one_minus_safety_factor_for_magnetic_field_;
  const double one_over_safety_factor_for_momentum_density_cutoff_d =
      1.0 / safety_factor_for_momentum_density_cutoff_d_;

  const auto fix_impl = [&](const size_t grid_index, auto use_simd,
                            const auto completion_mask) {
    using SimdType =
        tmpl::conditional_t<std::decay_t<decltype(use_simd)>::value,
                            simd::batch<double>, double>;
    const auto load = [&grid_index, &use_simd](const auto& data_vector) {
      (void)use_simd;
      if constexpr (std::decay_t<decltype(use_simd)>::value) {
        return simd::load_unaligned(&data_vector[grid_index]);
      } else {
        return data_vector[grid_index];
      }
    };

    auto d_tilde = load(get(*tilde_d));
    const auto sqrt_det_g = load(get(sqrt_det_spatial_metric));
    const SimdType one_over_sqrt_det_g = 1.0 / sqrt_det_g;
    SimdType one_over_d_tilde = 1.0 / d_tilde;
    SimdType rest_mass_density_times_lorentz_factor =
        d_tilde * one_over_sqrt_det_g;

    // Increase electron fraction if necessary
    auto ye_tilde = load(get(*tilde_ye));
    if (const auto ye_mask = ye_tilde < electron_fraction_cutoff_ * d_tilde;
        simd::any(ye_mask)) {
      needed_fixing = true;
      ye_tilde =
          simd::select(ye_mask, minimum_electron_fraction_ * d_tilde, ye_tilde);
    }

    // Increase mass density if necessary
    if (const auto tilde_d_mask =
            rest_mass_density_times_lorentz_factor <
            rest_mass_density_times_lorentz_factor_cutoff_;
        simd::any(tilde_d_mask)) {
      needed_fixing = true;
      d_tilde = simd::select(
          tilde_d_mask,
          minimum_rest_mass_density_times_lorentz_factor_ * sqrt_det_g,
          d_tilde);
      ye_tilde = d_tilde * one_over_d_tilde * ye_tilde;
      one_over_d_tilde = 1.0 / d_tilde;
      rest_mass_density_times_lorentz_factor =
          simd::select(tilde_d_mask,
                       static_cast<SimdType>(
                           minimum_rest_mass_density_times_lorentz_factor_),
                       rest_mass_density_times_lorentz_factor);
    }

    // Increase internal energy if necessary
    auto tau_tilde = load(get(*tilde_tau));
    const auto b_tilde_squared = load(get(tilde_b_squared));
    // Equation B.39 of Foucart
    if (const auto tilde_tau_mask =
            b_tilde_squared > one_minus_safety_factor_for_magnetic_field_ * 2. *
                                  tau_tilde * sqrt_det_g;
        simd::any(tilde_tau_mask)) {
      needed_fixing = true;
      tau_tilde =
          simd::select(tilde_tau_mask,
                       0.5 * b_tilde_squared *
                           one_over_one_minus_safety_factor_for_magnetic_field *
                           one_over_sqrt_det_g,
                       tau_tilde);
    }

    // Decrease momentum density if necessary
    auto s_tilde_squared = load(get(tilde_s_squared));
    // Equation B.24 of Foucart
    const SimdType tau_over_d = tau_tilde * one_over_d_tilde;
    // Equation B.23 of Foucart
    const SimdType b_squared_over_d =
        b_tilde_squared * one_over_sqrt_det_g * one_over_d_tilde;
    const auto s_tilde_dot_b_tilde = load(get(tilde_s_dot_tilde_b));

    // Equation B.27 of Foucart
    const SimdType normalized_s_dot_b = simd::select(
        (b_tilde_squared > 1.e-16 * d_tilde and
         s_tilde_squared > 1.e-16 * square(d_tilde)),
        s_tilde_dot_b_tilde / sqrt(b_tilde_squared * s_tilde_squared),
        SimdType{0.});

    // Equation B.40 of Foucart
    const auto lower_bound_of_lorentz_factor_minus_one =
        simd::max(tau_over_d - b_squared_over_d, SimdType{0.});
    // Equation B.31 of Foucart
    const auto upper_bound_for_s_tilde_squared =
        [&b_squared_over_d, &d_tilde, &lower_bound_of_lorentz_factor_minus_one,
         &normalized_s_dot_b](const auto local_excess_lorentz_factor) {
          const auto local_lorentz_factor_minus_one =
              lower_bound_of_lorentz_factor_minus_one +
              local_excess_lorentz_factor;
          return square(1.0 + local_lorentz_factor_minus_one +
                        b_squared_over_d) *
                 local_lorentz_factor_minus_one *
                 (2.0 + local_lorentz_factor_minus_one) /
                 (square(1.0 + local_lorentz_factor_minus_one) +
                  square(normalized_s_dot_b) * b_squared_over_d *
                      (b_squared_over_d +
                       2.0 * (1.0 + local_lorentz_factor_minus_one))) *
                 square(d_tilde);
        };
    const SimdType simple_upper_bound_for_s_tilde_squared =
        upper_bound_for_s_tilde_squared(SimdType{0.});

    // If s_tilde_squared is small enough, no fix is needed. Otherwise, we need
    // to do some real work.
    const auto one_minus_safety_factor_for_momentum_density_at_density =
        simd::select(
            rest_mass_density_times_lorentz_factor >
                safety_factor_for_momentum_density_cutoff_d_,
            SimdType{one_minus_safety_factor_for_momentum_density_},
            one_minus_safety_factor_for_momentum_density_ +
                safety_factor_for_momentum_density_slope_ *
                    log10(
                        rest_mass_density_times_lorentz_factor *
                        one_over_safety_factor_for_momentum_density_cutoff_d));
    if (const auto tilde_s_mask =
            s_tilde_squared >
                one_minus_safety_factor_for_momentum_density_at_density *
                    simple_upper_bound_for_s_tilde_squared and
            not completion_mask;
        simd::any(tilde_s_mask)) {
      // Find root of Equation B.34 of Foucart
      // NOTE:
      // - This assumes minimum specific enthalpy is 1.
      // - SpEC implements a more complicated formula (B.32) which is equivalent
      // - Bounds on root are given by Equation  B.40 of Foucart
      // - In regions where the solution is just above atmosphere we sometimes
      //   obtain an upper bound on the Lorentz factor somewhere around ~1e5,
      //   while the actual Lorentz factor is only 1+1e-6. This leads to
      //   situations where the solver must perform many (over 50) iterations to
      //   converge. A simple way of avoiding this is to check that
      //   [W_{lower_bound}, 10 * W_{lower_bound}] brackets the root and then
      //   use 10 * W_{lower_bound} as the upper bound. This reduces the number
      //   of iterations for the TOMS748 algorithm to converge to less than 10.
      //   Note that the factor 10 is chosen arbitrarily and could probably be
      //   reduced if required. The reasoning behind 10 is that it is unlikely
      //   the Lorentz factor will increase by a factor of 10 from one time step
      //   to the next in a physically meaning situation, and so 10 provides a
      //   reasonable bound.
      const auto f_of_lorentz_factor = FunctionOfLorentzFactor{
          b_squared_over_d, tau_over_d, normalized_s_dot_b,
          lower_bound_of_lorentz_factor_minus_one};
      SimdType upper_bound =
          simd::select(lower_bound_of_lorentz_factor_minus_one == 0.0,
                       tau_over_d, b_squared_over_d);

      SimdType excess_lorentz_factor = 0.0;
      if (const auto upper_not_zero_mask = upper_bound != 0.0;
          simd::any(upper_not_zero_mask)) {
        const SimdType f_at_lower = f_of_lorentz_factor(SimdType{0.0});
        const SimdType candidate_upper_bound =
            9.0 * (lower_bound_of_lorentz_factor_minus_one + 1.0);
        // The if-based implementation is here as a reference since it's
        // likely easier to understand.
        //
        // auto f_at_upper = std::numeric_limits<SimdType>::signaling_NaN();
        // if (upper_bound < candidate_upper_bound) {
        //   f_at_upper = f_of_lorentz_factor(upper_bound);
        // } else {
        //   f_at_upper = f_of_lorentz_factor(candidate_upper_bound);
        //   if (f_at_upper > 0.0) {
        //     upper_bound = candidate_upper_bound;
        //   } else {
        //     f_at_upper = f_of_lorentz_factor(upper_bound);
        //   }
        // }
        auto f_at_upper = f_of_lorentz_factor(upper_bound);
        if (const auto not_upper_less_candidate_bound =
                not(upper_bound < candidate_upper_bound);
            simd::any(not_upper_less_candidate_bound)) {
          const SimdType f_at_candidate_upper_bound =
              f_of_lorentz_factor(candidate_upper_bound);
          upper_bound = simd::select((f_at_candidate_upper_bound > 0.0) and
                                         not_upper_less_candidate_bound,
                                     candidate_upper_bound, upper_bound);
          f_at_upper = simd::select((f_at_candidate_upper_bound > 0.0) and
                                        not_upper_less_candidate_bound,
                                    f_at_candidate_upper_bound, f_at_upper);
        }

        try {
          excess_lorentz_factor = RootFinder::toms748<true>(
              f_of_lorentz_factor, SimdType{0.0}, upper_bound, f_at_lower,
              f_at_upper, 1.e-14, 1.e-14, 100, not tilde_s_mask);
        } catch (std::exception& exception) {
          // clang-format makes the streamed text hard to read in code...
          // clang-format off
        ERROR(
            "Failed to fix conserved variables because the root finder failed "
            "to find the lorentz factor.\n"
            "  upper_bound = "
            << std::scientific << std::setprecision(18)
            << upper_bound
            << "\n  lower_bound_of_lorentz_factor_minus_one = "
            << lower_bound_of_lorentz_factor_minus_one
            << "\n  s_tilde_squared = " << s_tilde_squared
            << "\n  d_tilde = " << d_tilde
            << "\n  sqrt_det_g = " << sqrt_det_g
            << "\n  tau_tilde = " << tau_tilde
            << "\n  b_tilde_squared = " << b_tilde_squared
            << "\n  b_squared_over_d = " << b_squared_over_d
            << "\n  tau_over_d = " << tau_over_d
            << "\n  normalized_s_dot_b = " << normalized_s_dot_b
            << "\n  tilde_s =\n" << extract_point(*tilde_s, grid_index)
            << "\n  tilde_b =\n" << extract_point(tilde_b, grid_index)
            << "\n  spatial_metric =\n"
            << extract_point(spatial_metric, grid_index)
            << "\n  inv_spatial_metric =\n"
            << extract_point(inv_spatial_metric, grid_index) << "\n"
            << "The message of the exception thrown by the root finder "
               "is:\n"
            << exception.what());
          // clang-format on
        }
      }

      const auto rescaling_factor = simd::select(
          tilde_s_mask,
          simd::min(
              sqrt(one_minus_safety_factor_for_momentum_density_ *
                   upper_bound_for_s_tilde_squared(excess_lorentz_factor) /
                   (s_tilde_squared + 1.e-16 * square(d_tilde))),
              SimdType{1.}),
          SimdType{1.});
      if (UNLIKELY(simd::any(rescaling_factor < 1.))) {
        needed_fixing = true;
        for (size_t i = 0; i < 3; i++) {
          if constexpr (std::decay_t<decltype(use_simd)>::value) {
            SimdType s_tilde = load(tilde_s->get(i));
            s_tilde *= rescaling_factor;
            simd::store_unaligned(&tilde_s->get(i)[grid_index], s_tilde);
          } else {
            tilde_s->get(i)[grid_index] *= rescaling_factor;
          }
        }
      }
    }

    if constexpr (std::decay_t<decltype(use_simd)>::value) {
      simd::store_unaligned(&get(*tilde_d)[grid_index], d_tilde);
      simd::store_unaligned(&get(*tilde_ye)[grid_index], ye_tilde);
      simd::store_unaligned(&get(*tilde_tau)[grid_index], tau_tilde);
    } else {
      get(*tilde_d)[grid_index] = d_tilde;
      get(*tilde_ye)[grid_index] = ye_tilde;
      get(*tilde_tau)[grid_index] = tau_tilde;
    }
  };

#ifdef SPECTRE_USE_XSIMD
  constexpr size_t simd_width = simd::size<simd::batch<double>>();
  if (size < simd_width) {
    for (size_t s = 0; s < size; s++) {
      fix_impl(s, std::false_type{}, false);
    }
  } else {
    const size_t vectorized_size = size - size % simd_width;
    for (size_t s = 0; s < vectorized_size; s += simd_width) {
      fix_impl(s, std::true_type{},
               simd::mask_type_t<simd::batch<double>>{false});
    }
    if (const size_t remainder = size - vectorized_size; remainder > 0) {
      const auto complete_mask = simd::make_sequence<simd::batch<double>>() <
                                 (simd_width - static_cast<double>(remainder));
      fix_impl(size - simd_width, std::true_type{}, complete_mask);
    }
  }
#else
  for (size_t s = 0; s < size; s++) {
    fix_impl(s, std::false_type{}, false);
  }
#endif
  return needed_fixing;
}

bool operator==(const FixConservatives& lhs, const FixConservatives& rhs) {
  return lhs.minimum_rest_mass_density_times_lorentz_factor_ ==
             rhs.minimum_rest_mass_density_times_lorentz_factor_ and
         lhs.rest_mass_density_times_lorentz_factor_cutoff_ ==
             rhs.rest_mass_density_times_lorentz_factor_cutoff_ and
         lhs.one_minus_safety_factor_for_magnetic_field_ ==
             rhs.one_minus_safety_factor_for_magnetic_field_ and
         lhs.one_minus_safety_factor_for_momentum_density_ ==
             rhs.one_minus_safety_factor_for_momentum_density_ and
         lhs.safety_factor_for_momentum_density_cutoff_d_ ==
             rhs.safety_factor_for_momentum_density_cutoff_d_ and
         lhs.safety_factor_for_momentum_density_slope_ ==
             rhs.safety_factor_for_momentum_density_slope_ and
         lhs.enable_ == rhs.enable_;
}

bool operator!=(const FixConservatives& lhs, const FixConservatives& rhs) {
  return not(lhs == rhs);
}
}  // namespace grmhd::ValenciaDivClean
