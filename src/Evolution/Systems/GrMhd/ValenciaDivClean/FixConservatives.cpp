// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <pup.h>  // IWYU pragma: keep

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <array>

// IWYU pragma: no_forward_declare Tensor

namespace {

// This class codes Eq. (B.34), rewritten as a standard-form
// polynomial in (W - 1) for better numerical behavior.  The
// implemented function is
// ((B^2/D) / 2 - (tau/D)) (1 + 2 (B^2/D) mu^2 + (B^2/D)^2 mu^2)
// + (W-1) (2 ((B^2/D) - (tau/D)) (1 + (B^2/D) mu^2) + (B^2/D) mu^2 + 1)
// + (W-1)^2 ((B^2/D) - (tau/D) + 3/2 (B^2/D) mu^2 + 2)
// + (W-1)^3
// where mu^2 = (B.S)^2 / (B^2 S^2).  A nice property of this form is
// that, because we've already modified tau to satisfy B.39, its value
// at W-1 = 0 is guaranteed to be negative, even in the presence of
// roundoff error.
class FunctionOfLorentzFactor {
 public:
  FunctionOfLorentzFactor(const double b_squared_over_d,
                          const double tau_over_d,
                          const double normalized_s_dot_b)
      : coefficients_{
            {(0.5 * b_squared_over_d - tau_over_d) *
                 (square(normalized_s_dot_b) * b_squared_over_d *
                      (b_squared_over_d + 2.0) +
                  1.0),
             2.0 * (square(normalized_s_dot_b) * b_squared_over_d + 1.0) *
                     (b_squared_over_d - tau_over_d) +
                 b_squared_over_d * square(normalized_s_dot_b) + 1.0,
             b_squared_over_d - tau_over_d +
                 1.5 * square(normalized_s_dot_b) * b_squared_over_d + 2.0,
             1.0}} {}

  double operator()(const double lorentz_factor_minus_one) const {
    return evaluate_polynomial(coefficients_, lorentz_factor_minus_one);
  }

 private:
  std::array<double, 4> coefficients_;
};
}  // namespace

namespace grmhd::ValenciaDivClean {
FixConservatives::FixConservatives(
    const double minimum_rest_mass_density_times_lorentz_factor,
    const double rest_mass_density_times_lorentz_factor_cutoff,
    const double safety_factor_for_magnetic_field,
    const double safety_factor_for_momentum_density,
    const Options::Context& context)
    : minimum_rest_mass_density_times_lorentz_factor_(
          minimum_rest_mass_density_times_lorentz_factor),
      rest_mass_density_times_lorentz_factor_cutoff_(
          rest_mass_density_times_lorentz_factor_cutoff),
      one_minus_safety_factor_for_magnetic_field_(
          1.0 - safety_factor_for_magnetic_field),
      one_minus_safety_factor_for_momentum_density_(
          1.0 - safety_factor_for_momentum_density) {
  if (minimum_rest_mass_density_times_lorentz_factor_ >
      rest_mass_density_times_lorentz_factor_cutoff_) {
    PARSE_ERROR(context,
                "The minimum value of D (a.k.a. rest mass density times "
                "Lorentz factor) ("
                    << minimum_rest_mass_density_times_lorentz_factor_
                    << ") must be less than or equal to the cutoff value of D ("
                    << rest_mass_density_times_lorentz_factor_cutoff_ << ')');
  }
}

// clang-tidy: google-runtime-references
void FixConservatives::pup(PUP::er& p) {  // NOLINT
  p | minimum_rest_mass_density_times_lorentz_factor_;
  p | rest_mass_density_times_lorentz_factor_cutoff_;
  p | one_minus_safety_factor_for_magnetic_field_;
  p | one_minus_safety_factor_for_momentum_density_;
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
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric) const {
  bool needed_fixing = false;
  const size_t size = get<0>(tilde_b).size();
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>, ::Tags::TempScalar<3>>>
      temp_buffer(size);

  DataVector& rest_mass_density_times_lorentz_factor =
      get(get<::Tags::TempScalar<0>>(temp_buffer));
  rest_mass_density_times_lorentz_factor =
      get(*tilde_d) / get(sqrt_det_spatial_metric);

  Scalar<DataVector>& tilde_b_squared = get<::Tags::TempScalar<1>>(temp_buffer);
  dot_product(make_not_null(&tilde_b_squared), tilde_b, tilde_b,
              spatial_metric);

  Scalar<DataVector>& tilde_s_squared = get<::Tags::TempScalar<2>>(temp_buffer);
  dot_product(make_not_null(&tilde_s_squared), *tilde_s, *tilde_s,
              inv_spatial_metric);

  Scalar<DataVector>& tilde_s_dot_tilde_b =
      get<::Tags::TempScalar<3>>(temp_buffer);
  dot_product(make_not_null(&tilde_s_dot_tilde_b), *tilde_s, tilde_b);

  for (size_t s = 0; s < size; s++) {
    // Increase density if necessary
    double& d_tilde = get(*tilde_d)[s];
    const double sqrt_det_g = get(sqrt_det_spatial_metric)[s];
    if (rest_mass_density_times_lorentz_factor[s] <
        rest_mass_density_times_lorentz_factor_cutoff_) {
      needed_fixing = true;
      d_tilde = minimum_rest_mass_density_times_lorentz_factor_ * sqrt_det_g;
    }

    // Increase internal energy if necessary
    double& tau_tilde = get(*tilde_tau)[s];
    const double b_tilde_squared = get(tilde_b_squared)[s];
    // Equation B.39 of Foucart
    if (b_tilde_squared > one_minus_safety_factor_for_magnetic_field_ * 2. *
                              tau_tilde * sqrt_det_g) {
      needed_fixing = true;
      tau_tilde = 0.5 * b_tilde_squared /
                  one_minus_safety_factor_for_magnetic_field_ / sqrt_det_g;
    }

    // Decrease momentum density if necessary
    const double s_tilde_squared = get(tilde_s_squared)[s];
    // Equation B.24 of Foucart
    const double tau_over_d = tau_tilde / d_tilde;
    // Equation B.23 of Foucart
    const double b_squared_over_d = b_tilde_squared / sqrt_det_g / d_tilde;
    // Equation B.27 of Foucart
    const double normalized_s_dot_b =
        (b_tilde_squared > 1.e-16 * d_tilde and
         s_tilde_squared > 1.e-16 * square(d_tilde))
            ? get(tilde_s_dot_tilde_b)[s] /
                  sqrt(b_tilde_squared * s_tilde_squared)
            : 0.;

    // Equation B.40 of Foucart
    const double lower_bound_of_lorentz_factor_minus_one =
        std::max(tau_over_d - b_squared_over_d, 0.);
    // Equation B.31 of Foucart
    const auto upper_bound_for_s_tilde_squared =
        [&b_squared_over_d, &d_tilde,
         &normalized_s_dot_b](const double local_lorentz_factor_minus_one) {
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
    const double simple_upper_bound_for_s_tilde_squared =
        upper_bound_for_s_tilde_squared(
            lower_bound_of_lorentz_factor_minus_one);

    // If s_tilde_squared is small enough, no fix is needed. Otherwise, we need
    // to do some real work.
    if (s_tilde_squared > one_minus_safety_factor_for_momentum_density_ *
                              simple_upper_bound_for_s_tilde_squared) {
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
          b_squared_over_d, tau_over_d, normalized_s_dot_b};
      double upper_bound_of_lorentz_factor_minus_one = tau_over_d;

      double lorentz_factor_minus_one =
          std::numeric_limits<double>::signaling_NaN();
      if (equal_within_roundoff(lower_bound_of_lorentz_factor_minus_one,
                                upper_bound_of_lorentz_factor_minus_one)) {
        lorentz_factor_minus_one = lower_bound_of_lorentz_factor_minus_one;
      } else {
        try {
          const double f_at_lower =
              f_of_lorentz_factor(lower_bound_of_lorentz_factor_minus_one);
          const double candidate_upper_bound =
              10.0 * (lower_bound_of_lorentz_factor_minus_one + 1.0) - 1.0;
          double f_at_upper = std::numeric_limits<double>::signaling_NaN();
          if (upper_bound_of_lorentz_factor_minus_one < candidate_upper_bound) {
            f_at_upper =
                f_of_lorentz_factor(upper_bound_of_lorentz_factor_minus_one);
          } else {
            f_at_upper = f_of_lorentz_factor(candidate_upper_bound);
            if (std::signbit(f_at_upper) != std::signbit(f_at_lower)) {
              upper_bound_of_lorentz_factor_minus_one = candidate_upper_bound;
            } else {
              f_at_upper =
                  f_of_lorentz_factor(upper_bound_of_lorentz_factor_minus_one);
            }
          }

          lorentz_factor_minus_one = RootFinder::toms748(
              f_of_lorentz_factor, lower_bound_of_lorentz_factor_minus_one,
              upper_bound_of_lorentz_factor_minus_one, f_at_lower, f_at_upper,
              1.e-14, 1.e-14, 100);
        } catch (std::exception& exception) {
          // clang-format makes the streamed text hard to read in code...
          // clang-format off
        ERROR(
            "Failed to fix conserved variables because the root finder failed "
            "to find the lorentz factor.\n"
            "  upper_bound = "
            << std::scientific << std::setprecision(18)
            << upper_bound_of_lorentz_factor_minus_one
            << "\n  lower_bound = " << lower_bound_of_lorentz_factor_minus_one
            << "\n  s_tilde_squared = " << s_tilde_squared
            << "\n  d_tilde = " << d_tilde
            << "\n  sqrt_det_g = " << sqrt_det_g
            << "\n  tau_tilde = " << tau_tilde
            << "\n  b_tilde_squared = " << b_tilde_squared
            << "\n  b_squared_over_d = " << b_squared_over_d
            << "\n  tau_over_d = " << tau_over_d
            << "\n  normalized_s_dot_b = " << normalized_s_dot_b << "\n"
            << "The message of the exception thrown by the root finder "
               "is:\n"
            << exception.what());
          // clang-format on
        }
      }

      const double rescaling_factor =
          sqrt(one_minus_safety_factor_for_momentum_density_ *
               upper_bound_for_s_tilde_squared(lorentz_factor_minus_one) /
               (s_tilde_squared + 1.e-16 * square(d_tilde)));
      if (rescaling_factor < 1.) {
        needed_fixing = true;
        for (size_t i = 0; i < 3; i++) {
          tilde_s->get(i)[s] *= rescaling_factor;
        }
      }
    }
  }
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
             rhs.one_minus_safety_factor_for_momentum_density_;
}

bool operator!=(const FixConservatives& lhs, const FixConservatives& rhs) {
  return not(lhs == rhs);
}
}  // namespace grmhd::ValenciaDivClean
