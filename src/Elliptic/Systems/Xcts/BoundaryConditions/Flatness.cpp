// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Xcts/BoundaryConditions/Flatness.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts::BoundaryConditions {

template <>
void Flatness<Xcts::Equations::Hamiltonian>::apply(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_conformal_factor_gradient*/,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor*/) {
  get(*conformal_factor_minus_one) = 0.;
}

template <>
void Flatness<Xcts::Equations::HamiltonianAndLapse>::apply(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_minus_one,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_conformal_factor_gradient*/,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_lapse_times_conformal_factor_gradient*/,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor*/,
    const tnsr::i<DataVector, 3>& /*deriv_lapse_times_conformal_factor*/) {
  get(*conformal_factor_minus_one) = 0.;
  get(*lapse_times_conformal_factor_minus_one) = 0.;
}

template <>
void Flatness<Xcts::Equations::HamiltonianLapseAndShift>::apply(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_minus_one,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_conformal_factor_gradient*/,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_lapse_times_conformal_factor_gradient*/,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
    /*n_dot_longitudinal_shift_excess*/,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor*/,
    const tnsr::i<DataVector, 3>& /*deriv_lapse_times_conformal_factor*/,
    const tnsr::iJ<DataVector, 3>& /*deriv_shift_excess*/) {
  get(*conformal_factor_minus_one) = 0.;
  get(*lapse_times_conformal_factor_minus_one) = 0.;
  std::fill(shift_excess->begin(), shift_excess->end(), 0.);
}

template <>
void Flatness<Xcts::Equations::Hamiltonian>::apply_linearized(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_conformal_factor_gradient_correction*/,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor_correction*/) {
  get(*conformal_factor_correction) = 0.;
}

template <>
void Flatness<Xcts::Equations::HamiltonianAndLapse>::apply_linearized(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_conformal_factor_gradient_correction*/,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_lapse_times_conformal_factor_gradient_correction*/,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor_correction*/,
    const tnsr::i<DataVector,
                  3>& /*deriv_lapse_times_conformal_factor_correction*/) {
  get(*conformal_factor_correction) = 0.;
  get(*lapse_times_conformal_factor_correction) = 0.;
}

template <>
void Flatness<Xcts::Equations::HamiltonianLapseAndShift>::apply_linearized(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess_correction,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_conformal_factor_gradient_correction*/,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_lapse_times_conformal_factor_gradient_correction*/,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
    /*n_dot_longitudinal_shift_excess_correction*/,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor_correction*/,
    const tnsr::i<DataVector,
                  3>& /*deriv_lapse_times_conformal_factor_correction*/,
    const tnsr::iJ<DataVector, 3>& /*deriv_shift_excess_correction*/) {
  get(*conformal_factor_correction) = 0.;
  get(*lapse_times_conformal_factor_correction) = 0.;
  std::fill(shift_excess_correction->begin(), shift_excess_correction->end(),
            0.);
}

template <Xcts::Equations EnabledEquations>
bool operator==(const Flatness<EnabledEquations>& /*lhs*/,
                const Flatness<EnabledEquations>& /*rhs*/) {
  return true;
}

template <Xcts::Equations EnabledEquations>
bool operator!=(const Flatness<EnabledEquations>& lhs,
                const Flatness<EnabledEquations>& rhs) {
  return not(lhs == rhs);
}

template <Xcts::Equations EnabledEquations>
PUP::able::PUP_ID Flatness<EnabledEquations>::my_PUP_ID = 0;  // NOLINT

#define EQNS(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                             \
  template class Flatness<EQNS(data)>;                   \
  template bool operator==(const Flatness<EQNS(data)>&,  \
                           const Flatness<EQNS(data)>&); \
  template bool operator!=(const Flatness<EQNS(data)>&,  \
                           const Flatness<EQNS(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Xcts::Equations::Hamiltonian,
                         Xcts::Equations::HamiltonianAndLapse,
                         Xcts::Equations::HamiltonianLapseAndShift))
#undef INSTANTIATE
#undef EQNS

}  // namespace Xcts::BoundaryConditions
