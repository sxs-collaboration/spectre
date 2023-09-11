// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/VariableFixing/ParameterizedDeleptonization.hpp"

#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/ParseError.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace VariableFixing {

ParameterizedDeleptonization::ParameterizedDeleptonization(
    const bool enable, const double high_density_scale,
    const double low_density_scale,
    const double electron_fraction_at_high_density,
    const double electron_fraction_at_low_density,
    const double electron_fraction_correction_scale,
    const Options::Context& context)
    : enable_(enable),
      high_density_scale_(high_density_scale),
      low_density_scale_(low_density_scale),
      electron_fraction_at_high_density_(electron_fraction_at_high_density),
      electron_fraction_at_low_density_(electron_fraction_at_low_density),
      electron_fraction_half_sum_(0.5 * (electron_fraction_at_high_density +
                                         electron_fraction_at_low_density)),
      electron_fraction_half_difference_(0.5 *
                                         (electron_fraction_at_high_density -
                                          electron_fraction_at_low_density)),
      electron_fraction_correction_scale_(electron_fraction_correction_scale) {
  if (high_density_scale_ < low_density_scale_) {
    PARSE_ERROR(context, "The high density scale("
                             << high_density_scale_
                             << ") should be greater than the low"
                                " density scale ("
                             << low_density_scale_ << ')');
  }
  // high density material should have a lower Ye
  if (electron_fraction_at_high_density_ > electron_fraction_at_low_density_) {
    PARSE_ERROR(context, "The Ye at high density("
                             << electron_fraction_at_high_density_
                             << ") should be less than the Ye at low"
                                " density ("
                             << electron_fraction_at_low_density_ << ')');
  }
}

// NOLINTNEXTLINE(google-runtime-references)
void ParameterizedDeleptonization::pup(PUP::er& p) {
  p | enable_;
  p | high_density_scale_;
  p | low_density_scale_;
  p | electron_fraction_at_high_density_;
  p | electron_fraction_at_low_density_;
  p | electron_fraction_half_sum_;
  p | electron_fraction_half_difference_;
  p | electron_fraction_correction_scale_;
}

template <size_t ThermodynamicDim>
void ParameterizedDeleptonization::operator()(
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const gsl::not_null<Scalar<DataVector>*> electron_fraction,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<Scalar<DataVector>*> specific_enthalpy,
    const Scalar<DataVector>& rest_mass_density,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) const {
  if (LIKELY(enable_)) {
    for (size_t i = 0; i < rest_mass_density.get().size(); i++) {
      correct_electron_fraction(specific_internal_energy, electron_fraction,
                                pressure, specific_enthalpy, rest_mass_density,
                                equation_of_state, i);
    }
  }
}

template <size_t ThermodynamicDim>
void ParameterizedDeleptonization::correct_electron_fraction(
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const gsl::not_null<Scalar<DataVector>*> electron_fraction,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<Scalar<DataVector>*> specific_enthalpy,
    const Scalar<DataVector>& rest_mass_density,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    const size_t grid_index) const {
  // calculate x
  const auto x_of_rho = std::max(
      -1.0,
      std::min(1.0, (2.0 * std::log10(rest_mass_density.get()[grid_index]) -
                     std::log10(high_density_scale_) -
                     std::log10(low_density_scale_)) /
                        (std::log10(high_density_scale_) -
                         std::log10(low_density_scale_))));

  // calculate ye_new
  const auto electron_fraction_from_analytic =
      electron_fraction_half_sum_ +
      x_of_rho * electron_fraction_half_difference_ +
      electron_fraction_correction_scale_ *
          (1.0 - abs(x_of_rho) +
           4.0 * abs(x_of_rho) * (abs(x_of_rho) - 0.5) * (abs(x_of_rho) - 1.0));

  // Ye can only decrease with this prescription
  if (electron_fraction_from_analytic < electron_fraction->get()[grid_index]) {
    electron_fraction->get()[grid_index] = electron_fraction_from_analytic;
  }

  // call EoS (eventually Ye dependent tabulated eos here)
  if constexpr (ThermodynamicDim == 1) {
    pressure->get()[grid_index] = get(equation_of_state.pressure_from_density(
        Scalar<double>{rest_mass_density.get()[grid_index]}));
    specific_internal_energy->get()[grid_index] =
        get(equation_of_state.specific_internal_energy_from_density(
            Scalar<double>{rest_mass_density.get()[grid_index]}));
  } else if constexpr (ThermodynamicDim == 2) {
    pressure->get()[grid_index] =
        get(equation_of_state.pressure_from_density_and_energy(
            Scalar<double>{rest_mass_density.get()[grid_index]},
            Scalar<double>{specific_internal_energy->get()[grid_index]}));
  }
  specific_enthalpy->get()[grid_index] =
      get(hydro::relativistic_specific_enthalpy(
          Scalar<double>{rest_mass_density.get()[grid_index]},
          Scalar<double>{specific_internal_energy->get()[grid_index]},
          Scalar<double>{pressure->get()[grid_index]}));
}

bool operator==(const ParameterizedDeleptonization& lhs,
                const ParameterizedDeleptonization& rhs) {
  return lhs.enable_ == rhs.enable_ and
         lhs.high_density_scale_ == rhs.high_density_scale_ and
         lhs.low_density_scale_ == rhs.low_density_scale_ and
         lhs.electron_fraction_at_high_density_ ==
             rhs.electron_fraction_at_high_density_ and
         lhs.electron_fraction_at_low_density_ ==
             rhs.electron_fraction_at_low_density_ and
         lhs.electron_fraction_half_sum_ == rhs.electron_fraction_half_sum_ and
         lhs.electron_fraction_half_difference_ ==
             rhs.electron_fraction_half_difference_ and
         lhs.electron_fraction_correction_scale_ ==
             rhs.electron_fraction_correction_scale_;
}

bool operator!=(const ParameterizedDeleptonization& lhs,
                const ParameterizedDeleptonization& rhs) {
  return not(lhs == rhs);
}

#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                             \
  template class ParameterizedDeleptonization;  \
  template bool operator==(                                \
      const ParameterizedDeleptonization& lhs,  \
      const ParameterizedDeleptonization& rhs); \
  template bool operator!=(                                \
      const ParameterizedDeleptonization& lhs,  \
      const ParameterizedDeleptonization& rhs);

#undef INSTANTIATION

#define INSTANTIATION(r, data)                                           \
  template void ParameterizedDeleptonization::operator()(     \
      const gsl::not_null<Scalar<DataVector>*> specific_internal_energy, \
      const gsl::not_null<Scalar<DataVector>*> electron_fraction,        \
      const gsl::not_null<Scalar<DataVector>*> pressure,                 \
      const gsl::not_null<Scalar<DataVector>*> specific_enthalpy,        \
      const Scalar<DataVector>& rest_mass_density,                       \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>&   \
          equation_of_state) const;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef THERMO_DIM
#undef INSTANTIATION

}  // namespace VariableFixing
