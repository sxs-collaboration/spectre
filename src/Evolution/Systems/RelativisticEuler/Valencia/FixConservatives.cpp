// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/RelativisticEuler/Valencia/FixConservatives.hpp"

#include <cmath>
#include <cstddef>
#include <ostream>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

namespace RelativisticEuler::Valencia {

template <size_t Dim>
FixConservatives<Dim>::FixConservatives(
    const double minimum_rest_mass_density_times_lorentz_factor,
    const double rest_mass_density_times_lorentz_factor_cutoff,
    const double safety_factor_for_momentum_density,
    const Options::Context& context)
    : minimum_rest_mass_density_times_lorentz_factor_(
          minimum_rest_mass_density_times_lorentz_factor),
      rest_mass_density_times_lorentz_factor_cutoff_(
          rest_mass_density_times_lorentz_factor_cutoff),
      one_minus_safety_factor_for_momentum_density_(
          1.0 - safety_factor_for_momentum_density) {
  if (minimum_rest_mass_density_times_lorentz_factor_ >
      rest_mass_density_times_lorentz_factor_cutoff_) {
    PARSE_ERROR(
        context,
        "The cutoff value of D (D = rest mass density * Lorentz factor) must "
        "not be below the minimum value of D.\nValues given: D_min = "
            << minimum_rest_mass_density_times_lorentz_factor_
            << ", D_cutoff = " << rest_mass_density_times_lorentz_factor_cutoff_
            << ".");
  }
}

template <size_t Dim>
void FixConservatives<Dim>::pup(PUP::er& p) {
  p | minimum_rest_mass_density_times_lorentz_factor_;
  p | rest_mass_density_times_lorentz_factor_cutoff_;
  p | one_minus_safety_factor_for_momentum_density_;
}

template <size_t Dim>
void FixConservatives<Dim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> tilde_s,
    const tnsr::II<DataVector, Dim, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric) const {
  const size_t number_of_points = get(sqrt_det_spatial_metric).size();
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>>>
      temp_buffer(number_of_points);

  // rest mass density times Lorentz factor, denoted in the dox as D
  DataVector& rho_w = get(get<::Tags::TempScalar<0>>(temp_buffer));
  rho_w = get(*tilde_d) / get(sqrt_det_spatial_metric);

  Scalar<DataVector>& tilde_s_squared = get<::Tags::TempScalar<1>>(temp_buffer);
  dot_product(make_not_null(&tilde_s_squared), *tilde_s, *tilde_s,
              inv_spatial_metric);

  for (size_t s = 0; s < number_of_points; ++s) {
    const double sqrt_det_spatial_metric_s = get(sqrt_det_spatial_metric)[s];

    // Fix tilde D
    double& tilde_d_s = get(*tilde_d)[s];
    if (rho_w[s] < rest_mass_density_times_lorentz_factor_cutoff_) {
      tilde_d_s = sqrt_det_spatial_metric_s *
                  minimum_rest_mass_density_times_lorentz_factor_;
    }

    // Fix tilde tau
    double& tilde_tau_s = get(*tilde_tau)[s];
    if (tilde_tau_s < 0.0) {
      tilde_tau_s = 0.0;
    }

    // Fix tilde S
    const double tilde_s_squared_max =
        tilde_tau_s * (tilde_tau_s + 2.0 * tilde_d_s);
    double& tilde_s_squared_s = get(tilde_s_squared)[s];
    if (tilde_s_squared_s >
        one_minus_safety_factor_for_momentum_density_ * tilde_s_squared_max) {
      const double rescaling_factor = sqrt(
          one_minus_safety_factor_for_momentum_density_ * tilde_s_squared_max /
          (tilde_s_squared_s + 1.e-16 * square(tilde_d_s)));
      if (rescaling_factor < 1.0) {
        for (size_t i = 0; i < Dim; ++i) {
          tilde_s->get(i)[s] *= rescaling_factor;
        }
      }
    }
  }
}

template <size_t Dim>
bool operator==(const FixConservatives<Dim>& lhs,
                const FixConservatives<Dim>& rhs) {
  return lhs.minimum_rest_mass_density_times_lorentz_factor_ ==
             rhs.minimum_rest_mass_density_times_lorentz_factor_ and
         lhs.rest_mass_density_times_lorentz_factor_cutoff_ ==
             rhs.rest_mass_density_times_lorentz_factor_cutoff_ and
         lhs.one_minus_safety_factor_for_momentum_density_ ==
             rhs.one_minus_safety_factor_for_momentum_density_;
}

template <size_t Dim>
bool operator!=(const FixConservatives<Dim>& lhs,
                const FixConservatives<Dim>& rhs) {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE_CLASS(_, data)                              \
  template class FixConservatives<DIM(data)>;                   \
  template bool operator==(const FixConservatives<DIM(data)>&,  \
                           const FixConservatives<DIM(data)>&); \
  template bool operator!=(const FixConservatives<DIM(data)>&,  \
                           const FixConservatives<DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE_CLASS, (1, 2, 3))

#undef INSTANTIATE_CLASS
#undef DIM
}  // namespace RelativisticEuler::Valencia
