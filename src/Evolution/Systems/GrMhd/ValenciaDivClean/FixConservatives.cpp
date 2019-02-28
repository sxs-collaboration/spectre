// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <pup.h>  // IWYU pragma: keep

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <array>

// IWYU pragma: no_forward_declare Tensor

namespace {

class FunctionOfLorentzFactor {
 public:
  FunctionOfLorentzFactor(const double b_squared_over_d,
                          const double tau_over_d,
                          const double normalized_s_dot_b) noexcept
      : b_squared_over_d_(b_squared_over_d),
        tau_over_d_(tau_over_d),
        normalized_s_dot_b_(normalized_s_dot_b) {}

  // This function codes Eq. (B.34)
  double operator()(const double lorentz_factor) const noexcept {
    return (lorentz_factor + b_squared_over_d_ - tau_over_d_ - 1.0) *
               (square(lorentz_factor) +
                b_squared_over_d_ * square(normalized_s_dot_b_) *
                    (b_squared_over_d_ + 2.0 * lorentz_factor)) -
           0.5 * b_squared_over_d_ -
           0.5 * b_squared_over_d_ * square(normalized_s_dot_b_) *
               (square(lorentz_factor) - 1.0 +
                2.0 * lorentz_factor * b_squared_over_d_ +
                square(b_squared_over_d_));
  }

 private:
  const double b_squared_over_d_;
  const double tau_over_d_;
  const double normalized_s_dot_b_;
};
}  // namespace

namespace VariableFixing {
FixConservatives::FixConservatives(
    const double minimum_rest_mass_density_times_lorentz_factor,
    const double rest_mass_density_times_lorentz_factor_cutoff,
    const double safety_factor_for_magnetic_field,
    const double safety_factor_for_momentum_density,
    const OptionContext& context)
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
void FixConservatives::pup(PUP::er& p) noexcept {  // NOLINT
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
void FixConservatives::operator()(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric) const noexcept {
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
      d_tilde = minimum_rest_mass_density_times_lorentz_factor_ * sqrt_det_g;
    }

    // Increase internal energy if necessary
    double& tau_tilde = get(*tilde_tau)[s];
    const double b_tilde_squared = get(tilde_b_squared)[s];
    // Equation B.39 of Foucart
    if (b_tilde_squared > one_minus_safety_factor_for_magnetic_field_ * 2. *
                              tau_tilde * sqrt_det_g) {
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
    const double lower_bound_of_lorentz_factor =
        std::max(1. + tau_over_d - b_squared_over_d, 1.);
    // Equation B.31 of Foucart evaluated at lower bound of lorentz factor
    const double simple_upper_bound_for_s_tilde_squared =
        square(lower_bound_of_lorentz_factor + b_squared_over_d) *
        (square(lower_bound_of_lorentz_factor) - 1.) /
        (square(lower_bound_of_lorentz_factor) +
         square(normalized_s_dot_b) * b_squared_over_d *
             (b_squared_over_d + 2. * lower_bound_of_lorentz_factor)) *
        square(d_tilde);

    // If s_tilde_squared is small enough, no fix is needed. Otherwise, we need
    // to do some real work.
    if (s_tilde_squared > one_minus_safety_factor_for_momentum_density_ *
                              simple_upper_bound_for_s_tilde_squared) {
      // Find root of Equation B.34 of Foucart
      // NOTE: This assumes minimum specific enthalpy is 1.
      // SpEC implements a more complicated formula (B.32) which is equivalent
      // Bounds on root are given by Equation  B.40 of Foucart
      const auto f_of_lorentz_factor = FunctionOfLorentzFactor{
          b_squared_over_d, tau_over_d, normalized_s_dot_b};
      const double upper_bound_of_lorentz_factor = 1.0 + tau_over_d;
      const double lorentz_factor =
          (equal_within_roundoff(lower_bound_of_lorentz_factor,
                                 upper_bound_of_lorentz_factor)
               ? lower_bound_of_lorentz_factor
               :
               // NOLINTNEXTLINE(clang-analyzer-core)
               RootFinder::toms748(
                   f_of_lorentz_factor, lower_bound_of_lorentz_factor,
                   upper_bound_of_lorentz_factor, 1.e-14, 1.e-14, 50));

      const double upper_bound_for_s_tilde_squared =
          square(lorentz_factor + b_squared_over_d) *
          (square(lorentz_factor) - 1.) /
          (square(lorentz_factor) +
           square(normalized_s_dot_b) * b_squared_over_d *
               (b_squared_over_d + 2. * lorentz_factor)) *
          square(d_tilde);
      const double rescaling_factor =
          sqrt(one_minus_safety_factor_for_momentum_density_ *
               upper_bound_for_s_tilde_squared /
               (s_tilde_squared + 1.e-16 * square(d_tilde)));
      if (rescaling_factor < 1.) {
        for (size_t i = 0; i < 3; i++) {
          tilde_s->get(i)[s] *= rescaling_factor;
        }
      }
    }
  }
}

bool operator==(const FixConservatives& lhs,
                const FixConservatives& rhs) noexcept {
  return lhs.minimum_rest_mass_density_times_lorentz_factor_ ==
             rhs.minimum_rest_mass_density_times_lorentz_factor_ and
         lhs.rest_mass_density_times_lorentz_factor_cutoff_ ==
             rhs.rest_mass_density_times_lorentz_factor_cutoff_ and
         lhs.one_minus_safety_factor_for_magnetic_field_ ==
             rhs.one_minus_safety_factor_for_magnetic_field_ and
         lhs.one_minus_safety_factor_for_momentum_density_ ==
             rhs.one_minus_safety_factor_for_momentum_density_;
}

bool operator!=(const FixConservatives& lhs,
                const FixConservatives& rhs) noexcept {
  return not(lhs == rhs);
}
}  // namespace VariableFixing
