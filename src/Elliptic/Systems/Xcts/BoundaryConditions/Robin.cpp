// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Xcts/BoundaryConditions/Robin.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts::BoundaryConditions {

namespace {
void robin_boundary_condition_scalar(
    const gsl::not_null<Scalar<DataVector>*> n_dot_gradient,
    const Scalar<DataVector>& scalar, const Scalar<DataVector>& r) {
  get(*n_dot_gradient) = -get(scalar) / get(r);
}

void robin_boundary_condition_shift(
    const gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_longitudinal_shift,
    tnsr::I<DataVector, 3> shift, const tnsr::iJ<DataVector, 3>& deriv_shift,
    const Scalar<DataVector>& r, const tnsr::i<DataVector, 3>& face_normal) {
  for (size_t i = 0; i < 3; ++i) {
    shift.get(i) /= get(r);
    for (size_t j = 0; j < 3; ++j) {
      shift.get(i) += face_normal.get(j) * deriv_shift.get(j, i);
    }
  }
  const auto n_dot_shift = dot_product(face_normal, shift);
  for (size_t i = 0; i < 3; ++i) {
    n_dot_longitudinal_shift->get(i) -=
        shift.get(i) + face_normal.get(i) * get(n_dot_shift) / 3.;
  }
}
}  // namespace

template <Xcts::Equations EnabledEquations>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
Robin<EnabledEquations>::get_clone() const {
  return std::make_unique<Robin>(*this);
}

template <Xcts::Equations EnabledEquations>
std::vector<elliptic::BoundaryConditionType>
Robin<EnabledEquations>::boundary_condition_types() const {
  if constexpr (EnabledEquations == Xcts::Equations::Hamiltonian) {
    return {1, elliptic::BoundaryConditionType::Neumann};
  } else if constexpr (EnabledEquations ==
                       Xcts::Equations::HamiltonianAndLapse) {
    return {2, elliptic::BoundaryConditionType::Neumann};
  } else {
    return {5, elliptic::BoundaryConditionType::Neumann};
  }
}

template <>
void Robin<Xcts::Equations::Hamiltonian>::apply(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor*/,
    const tnsr::I<DataVector, 3>& x,
    const tnsr::i<DataVector, 3>& /*face_normal*/) const {
  robin_boundary_condition_scalar(n_dot_conformal_factor_gradient,
                                  *conformal_factor_minus_one, magnitude(x));
}

template <>
void Robin<Xcts::Equations::HamiltonianAndLapse>::apply(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_minus_one,
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor*/,
    const tnsr::i<DataVector, 3>& /*deriv_lapse_times_conformal_factor*/,
    const tnsr::I<DataVector, 3>& x,
    const tnsr::i<DataVector, 3>& /*face_normal*/) const {
  const auto r = magnitude(x);
  robin_boundary_condition_scalar(n_dot_conformal_factor_gradient,
                                  *conformal_factor_minus_one, r);
  robin_boundary_condition_scalar(n_dot_lapse_times_conformal_factor_gradient,
                                  *lapse_times_conformal_factor_minus_one, r);
}

template <>
void Robin<Xcts::Equations::HamiltonianLapseAndShift>::apply(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_minus_one,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        n_dot_longitudinal_shift_excess,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor*/,
    const tnsr::i<DataVector, 3>& /*deriv_lapse_times_conformal_factor*/,
    const tnsr::iJ<DataVector, 3>& deriv_shift_excess,
    const tnsr::I<DataVector, 3>& x,
    const tnsr::i<DataVector, 3>& face_normal) const {
  const auto r = magnitude(x);
  robin_boundary_condition_scalar(n_dot_conformal_factor_gradient,
                                  *conformal_factor_minus_one, r);
  robin_boundary_condition_scalar(n_dot_lapse_times_conformal_factor_gradient,
                                  *lapse_times_conformal_factor_minus_one, r);
  robin_boundary_condition_shift(n_dot_longitudinal_shift_excess, *shift_excess,
                                 deriv_shift_excess, r, face_normal);
}

template <>
void Robin<Xcts::Equations::Hamiltonian>::apply_linearized(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_conformal_factor_gradient_correction,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor_correction*/,
    const tnsr::I<DataVector, 3>& x,
    const tnsr::i<DataVector, 3>& /*face_normal*/) const {
  robin_boundary_condition_scalar(n_dot_conformal_factor_gradient_correction,
                                  *conformal_factor_correction, magnitude(x));
}

template <>
void Robin<Xcts::Equations::HamiltonianAndLapse>::apply_linearized(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_conformal_factor_gradient_correction,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient_correction,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor_correction*/,
    const tnsr::i<DataVector,
                  3>& /*deriv_lapse_times_conformal_factor_correction*/,
    const tnsr::I<DataVector, 3>& x,
    const tnsr::i<DataVector, 3>& /*face_normal*/) const {
  const auto r = magnitude(x);
  robin_boundary_condition_scalar(n_dot_conformal_factor_gradient_correction,
                                  *conformal_factor_correction, r);
  robin_boundary_condition_scalar(
      n_dot_lapse_times_conformal_factor_gradient_correction,
      *lapse_times_conformal_factor_correction, r);
}

template <>
void Robin<Xcts::Equations::HamiltonianLapseAndShift>::apply_linearized(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess_correction,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_conformal_factor_gradient_correction,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        n_dot_longitudinal_shift_excess_correction,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor_correction*/,
    const tnsr::i<DataVector,
                  3>& /*deriv_lapse_times_conformal_factor_correction*/,
    const tnsr::iJ<DataVector, 3>& deriv_shift_excess_correction,
    const tnsr::I<DataVector, 3>& x,
    const tnsr::i<DataVector, 3>& face_normal) const {
  const auto r = magnitude(x);
  robin_boundary_condition_scalar(n_dot_conformal_factor_gradient_correction,
                                  *conformal_factor_correction, r);
  robin_boundary_condition_scalar(
      n_dot_lapse_times_conformal_factor_gradient_correction,
      *lapse_times_conformal_factor_correction, r);
  robin_boundary_condition_shift(n_dot_longitudinal_shift_excess_correction,
                                 *shift_excess_correction,
                                 deriv_shift_excess_correction, r, face_normal);
}

template <Xcts::Equations EnabledEquations>
bool operator==(const Robin<EnabledEquations>& /*lhs*/,
                const Robin<EnabledEquations>& /*rhs*/) {
  return true;
}

template <Xcts::Equations EnabledEquations>
bool operator!=(const Robin<EnabledEquations>& lhs,
                const Robin<EnabledEquations>& rhs) {
  return not(lhs == rhs);
}

template <Xcts::Equations EnabledEquations>
PUP::able::PUP_ID Robin<EnabledEquations>::my_PUP_ID = 0;  // NOLINT

#define EQNS(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                          \
  template class Robin<EQNS(data)>;                   \
  template bool operator==(const Robin<EQNS(data)>&,  \
                           const Robin<EQNS(data)>&); \
  template bool operator!=(const Robin<EQNS(data)>&, const Robin<EQNS(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Xcts::Equations::Hamiltonian,
                         Xcts::Equations::HamiltonianAndLapse,
                         Xcts::Equations::HamiltonianLapseAndShift))
#undef INSTANTIATE
#undef EQNS

}  // namespace Xcts::BoundaryConditions
