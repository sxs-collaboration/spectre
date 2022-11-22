// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/PiecewisePolytropicFluid.hpp"

#include <cmath>
#include <limits>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_forward_declare Tensor

// Assign polytropic exponents & constants based on primitive values
namespace {

void choose_polytropic_properties(
    const gsl::not_null<double*> polytropic_constant,
    const gsl::not_null<double*> polytropic_exponent,
    const double primitive_vector, const double primitive_comparison,
    const double hi_constant, const double lo_constant,
    const double hi_exponent, const double lo_exponent) {
  // Assumes >= for high density material
  if (primitive_vector < primitive_comparison) {
    *polytropic_constant = lo_constant;
    *polytropic_exponent = lo_exponent;
  } else {
    *polytropic_constant = hi_constant;
    *polytropic_exponent = hi_exponent;
  }
}
}  // namespace

namespace EquationsOfState {
template <bool IsRelativistic>
PiecewisePolytropicFluid<IsRelativistic>::PiecewisePolytropicFluid(
    double transition_density, double polytropic_constant_lo,
    double polytropic_exponent_lo, double polytropic_exponent_hi)
    : transition_density_(transition_density),
      transition_pressure_(polytropic_constant_lo *
                           pow(transition_density_, polytropic_exponent_lo)),
      transition_spec_eint_(
          polytropic_constant_lo / (polytropic_exponent_lo - 1.0) *
          pow(transition_density_, polytropic_exponent_lo - 1.0)),
      polytropic_constant_lo_(polytropic_constant_lo),
      polytropic_exponent_lo_(polytropic_exponent_lo),
      polytropic_constant_hi_(
          polytropic_constant_lo *
          pow(transition_density,
              polytropic_exponent_lo - polytropic_exponent_hi)),
      polytropic_exponent_hi_(polytropic_exponent_hi) {}

EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <bool IsRelativistic>,
                                     PiecewisePolytropicFluid<IsRelativistic>,
                                     double, 1)
EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <bool IsRelativistic>,
                                     PiecewisePolytropicFluid<IsRelativistic>,
                                     DataVector, 1)

template <bool IsRelativistic>
bool PiecewisePolytropicFluid<IsRelativistic>::operator==(
    const PiecewisePolytropicFluid<IsRelativistic>& rhs) const {
  return (transition_density_ == rhs.transition_density_) and
         (polytropic_constant_lo_ == rhs.polytropic_constant_lo_) and
         (polytropic_exponent_lo_ == rhs.polytropic_exponent_lo_) and
         (polytropic_constant_hi_ == rhs.polytropic_constant_hi_) and
         (polytropic_exponent_hi_ == rhs.polytropic_exponent_hi_);
}

template <bool IsRelativistic>
bool PiecewisePolytropicFluid<IsRelativistic>::operator!=(
    const PiecewisePolytropicFluid<IsRelativistic>& rhs) const {
  return not(*this == rhs);
}

template <bool IsRelativistic>
bool PiecewisePolytropicFluid<IsRelativistic>::is_equal(
    const EquationOfState<IsRelativistic, 1>& rhs) const {
  const auto& derived_ptr =
      dynamic_cast<const PiecewisePolytropicFluid<IsRelativistic>* const>(&rhs);
  return derived_ptr != nullptr and *derived_ptr == *this;
}

template <bool IsRelativistic>
std::unique_ptr<EquationOfState<IsRelativistic, 1>>
PiecewisePolytropicFluid<IsRelativistic>::get_clone() const {
  auto clone =
      std::make_unique<PiecewisePolytropicFluid<IsRelativistic>>(*this);
  return std::unique_ptr<EquationOfState<IsRelativistic, 1>>(std::move(clone));
}

template <bool IsRelativistic>
PiecewisePolytropicFluid<IsRelativistic>::PiecewisePolytropicFluid(
    CkMigrateMessage* msg)
    : EquationOfState<IsRelativistic, 1>(msg) {}

template <bool IsRelativistic>
void PiecewisePolytropicFluid<IsRelativistic>::pup(PUP::er& p) {
  EquationOfState<IsRelativistic, 1>::pup(p);
  p | transition_density_;
  p | transition_pressure_;
  p | transition_spec_eint_;
  p | polytropic_constant_lo_;
  p | polytropic_exponent_lo_;
  p | polytropic_constant_hi_;
  p | polytropic_exponent_hi_;
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType>
PiecewisePolytropicFluid<IsRelativistic>::pressure_from_density_impl(
    const Scalar<DataType>& rest_mass_density) const {
  double polytropic_constant = std::numeric_limits<double>::signaling_NaN();
  double polytropic_exponent = std::numeric_limits<double>::signaling_NaN();

  auto result = make_with_value<Scalar<DataType>>(rest_mass_density, 0.0);

  for (size_t i = 0; i < get_size(get(rest_mass_density)); ++i) {
    const double density = get_element(get(rest_mass_density), i);
    // select high or low polytropic constant & exponent
    choose_polytropic_properties(
        &polytropic_constant, &polytropic_exponent, density,
        transition_density_, polytropic_constant_hi_, polytropic_constant_lo_,
        polytropic_exponent_hi_, polytropic_exponent_lo_);
    // calculate pressure from density
    get_element(get(result), i) =
        polytropic_constant * pow(density, polytropic_exponent);
  }
  return result;
}

// Relativistic specific enthalpy
template <>
template <class DataType>
Scalar<DataType>
PiecewisePolytropicFluid<true>::rest_mass_density_from_enthalpy_impl(
    const Scalar<DataType>& specific_enthalpy) const {
  const double transition_spec_enthalpy =
      1.0 + transition_spec_eint_ + transition_pressure_ / transition_density_;

  double polytropic_constant = std::numeric_limits<double>::signaling_NaN();
  double polytropic_exponent = std::numeric_limits<double>::signaling_NaN();

  auto result = make_with_value<Scalar<DataType>>(specific_enthalpy, 0.0);

  for (size_t i = 0; i < get_size(get(specific_enthalpy)); ++i) {
    const double spec_enthalpy = get_element(get(specific_enthalpy), i);
    // select high or low polytropic constant & exponent
    choose_polytropic_properties(
        &polytropic_constant, &polytropic_exponent, spec_enthalpy,
        transition_spec_enthalpy, polytropic_constant_hi_,
        polytropic_constant_lo_, polytropic_exponent_hi_,
        polytropic_exponent_lo_);
    // calculate density from relativistic specific enthalpy
    get_element(get(result), i) =
        pow(((polytropic_exponent - 1.0) /
             (polytropic_constant * polytropic_exponent)) *
                (spec_enthalpy - 1.0 -
                 (polytropic_exponent - polytropic_exponent_lo_) /
                     ((polytropic_exponent_hi_ - 1.0) *
                      (polytropic_exponent_lo_ - 1.0)) *
                     polytropic_constant_lo_ *
                     pow(transition_density_, polytropic_exponent_lo_ - 1.0)),
            1.0 / (polytropic_exponent - 1.0));
  }
  return result;
}

// Newtonian specific enthalpy
template <>
template <class DataType>
Scalar<DataType>
PiecewisePolytropicFluid<false>::rest_mass_density_from_enthalpy_impl(
    const Scalar<DataType>& specific_enthalpy) const {
  const double transition_spec_enthalpy =
      transition_spec_eint_ + transition_pressure_ / transition_density_;

  double polytropic_constant = std::numeric_limits<double>::signaling_NaN();
  double polytropic_exponent = std::numeric_limits<double>::signaling_NaN();

  auto result = make_with_value<Scalar<DataType>>(specific_enthalpy, 0.0);

  for (size_t i = 0; i < get_size(get(specific_enthalpy)); ++i) {
    const double spec_enthalpy = get_element(get(specific_enthalpy), i);
    // select high or low polytropic constant & exponent
    choose_polytropic_properties(
        &polytropic_constant, &polytropic_exponent, spec_enthalpy,
        transition_spec_enthalpy, polytropic_constant_hi_,
        polytropic_constant_lo_, polytropic_exponent_hi_,
        polytropic_exponent_lo_);
    // calculate density from Newtonian specific enthalpy
    get_element(get(result), i) =
        pow(((polytropic_exponent - 1.0) /
             (polytropic_constant * polytropic_exponent)) *
                (spec_enthalpy -
                 (polytropic_exponent - polytropic_exponent_lo_) /
                     ((polytropic_exponent_hi_ - 1.0) *
                      (polytropic_exponent_lo_ - 1.0)) *
                     polytropic_constant_lo_ *
                     pow(transition_density_, polytropic_exponent_lo_ - 1.0)),
            1.0 / (polytropic_exponent - 1.0));
  }
  return result;
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType> PiecewisePolytropicFluid<IsRelativistic>::
    specific_internal_energy_from_density_impl(
        const Scalar<DataType>& rest_mass_density) const {
  double polytropic_constant = std::numeric_limits<double>::signaling_NaN();
  double polytropic_exponent = std::numeric_limits<double>::signaling_NaN();

  auto result = make_with_value<Scalar<DataType>>(rest_mass_density, 0.0);

  for (size_t i = 0; i < get_size(get(rest_mass_density)); ++i) {
    const double density = get_element(get(rest_mass_density), i);
    // select high or low polytropic constant & exponent
    choose_polytropic_properties(
        &polytropic_constant, &polytropic_exponent, density,
        transition_density_, polytropic_constant_hi_, polytropic_constant_lo_,
        polytropic_exponent_hi_, polytropic_exponent_lo_);
    // calculate specific internal energy from density
    get_element(get(result), i) =
        polytropic_constant / (polytropic_exponent - 1.0) *
            pow(density, polytropic_exponent - 1.0) +
        (polytropic_exponent - polytropic_exponent_lo_) /
            ((polytropic_exponent_hi_ - 1.0) *
             (polytropic_exponent_lo_ - 1.0)) *
            polytropic_constant_lo_ *
            pow(transition_density_, (polytropic_exponent_lo_ - 1.0));
  }
  return result;
}

// Chi = dP/drho
template <bool IsRelativistic>
template <class DataType>
Scalar<DataType>
PiecewisePolytropicFluid<IsRelativistic>::chi_from_density_impl(
    const Scalar<DataType>& rest_mass_density) const {
  double polytropic_constant = std::numeric_limits<double>::signaling_NaN();
  double polytropic_exponent = std::numeric_limits<double>::signaling_NaN();

  auto result = make_with_value<Scalar<DataType>>(rest_mass_density, 0.0);

  for (size_t i = 0; i < get_size(get(rest_mass_density)); ++i) {
    const double density = get_element(get(rest_mass_density), i);
    // select high or low polytropic constant & exponent
    choose_polytropic_properties(
        &polytropic_constant, &polytropic_exponent, density,
        transition_density_, polytropic_constant_hi_, polytropic_constant_lo_,
        polytropic_exponent_hi_, polytropic_exponent_lo_);
    // calculate chi from density
    get_element(get(result), i) = polytropic_constant * polytropic_exponent *
                                  pow(density, polytropic_exponent - 1.0);
  }
  return result;
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType> PiecewisePolytropicFluid<IsRelativistic>::
    kappa_times_p_over_rho_squared_from_density_impl(
        const Scalar<DataType>& rest_mass_density) const {
  return make_with_value<Scalar<DataType>>(get(rest_mass_density), 0.0);
}

template <bool IsRelativistic>
double PiecewisePolytropicFluid<IsRelativistic>::rest_mass_density_upper_bound()
    const {
  // this bound comes from the dominant energy condition which implies
  // that the pressure is bounded by the total energy density,
  // i.e. p < e = rho * (1 + eps)
  if (IsRelativistic and polytropic_exponent_hi_ > 2.0) {
    const double eint_boundary_constant =
        (polytropic_exponent_hi_ - polytropic_exponent_lo_) *
        polytropic_constant_lo_ /
        ((polytropic_exponent_hi_ - 1.0) * (polytropic_exponent_lo_ - 1.0)) *
        pow(transition_density_, polytropic_exponent_lo_ - 1.0);
    return pow((polytropic_exponent_hi_ - 1.0) /
                   (polytropic_constant_hi_ * (polytropic_exponent_hi_ - 2.0)) *
                   (1.0 + eint_boundary_constant),
               1.0 / (polytropic_exponent_hi_ - 1.0));
  }
  return std::numeric_limits<double>::max();
}
}  // namespace EquationsOfState

template class EquationsOfState::PiecewisePolytropicFluid<true>;
template class EquationsOfState::PiecewisePolytropicFluid<false>;
