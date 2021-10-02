// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"

#include <array>
#include <pup.h>  // IWYU pragma: keep

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Elasticity::ConstitutiveRelations {

template <size_t Dim>
IsotropicHomogeneous<Dim>::IsotropicHomogeneous(double bulk_modulus,
                                                double shear_modulus)
    : bulk_modulus_(bulk_modulus), shear_modulus_(shear_modulus) {}

template <>
void IsotropicHomogeneous<3>::stress(
    const gsl::not_null<tnsr::II<DataVector, 3>*> stress,
    const tnsr::ii<DataVector, 3>& strain,
    const tnsr::I<DataVector, 3>& /*x*/) const {
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      stress->get(i, j) = -2. * shear_modulus_ * strain.get(i, j);
    }
  }
  auto trace_term = make_with_value<DataVector>(strain, 0.);
  for (size_t i = 0; i < 3; ++i) {
    trace_term += strain.get(i, i);
  }
  trace_term *= lame_parameter();
  for (size_t i = 0; i < 3; ++i) {
    stress->get(i, i) -= trace_term;
  }
}

template <>
void IsotropicHomogeneous<2>::stress(
    const gsl::not_null<tnsr::II<DataVector, 2>*> stress,
    const tnsr::ii<DataVector, 2>& strain,
    const tnsr::I<DataVector, 2>& /*x*/) const {
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      stress->get(i, j) = -2. * shear_modulus_ * strain.get(i, j);
    }
  }
  auto trace_term = make_with_value<DataVector>(strain, 0.);
  for (size_t i = 0; i < 2; ++i) {
    trace_term += strain.get(i, i);
  }
  trace_term *= 2. * (3. * bulk_modulus_ - 2. * shear_modulus_) *
                shear_modulus_ / (3. * bulk_modulus_ + 4. * shear_modulus_);
  for (size_t i = 0; i < 2; ++i) {
    stress->get(i, i) -= trace_term;
  }
}

template <size_t Dim>
double IsotropicHomogeneous<Dim>::bulk_modulus() const {
  return bulk_modulus_;
}

template <size_t Dim>
double IsotropicHomogeneous<Dim>::shear_modulus() const {
  return shear_modulus_;
}

template <size_t Dim>
double IsotropicHomogeneous<Dim>::lame_parameter() const {
  return bulk_modulus_ - 2. * shear_modulus_ / 3.;
}

template <size_t Dim>
double IsotropicHomogeneous<Dim>::youngs_modulus() const {
  return 9. * bulk_modulus_ * shear_modulus_ /
         (3. * bulk_modulus_ + shear_modulus_);
}

template <size_t Dim>
double IsotropicHomogeneous<Dim>::poisson_ratio() const {
  return (3. * bulk_modulus_ - 2. * shear_modulus_) /
         (6. * bulk_modulus_ + 2. * shear_modulus_);
}

template <size_t Dim>
void IsotropicHomogeneous<Dim>::pup(PUP::er& p) {
  p | bulk_modulus_;
  p | shear_modulus_;
}

template <size_t Dim>
bool operator==(const IsotropicHomogeneous<Dim>& lhs,
                const IsotropicHomogeneous<Dim>& rhs) {
  return lhs.bulk_modulus() == rhs.bulk_modulus() and
         lhs.shear_modulus() == rhs.shear_modulus();
}
template <size_t Dim>
bool operator!=(const IsotropicHomogeneous<Dim>& lhs,
                const IsotropicHomogeneous<Dim>& rhs) {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                            \
  template class IsotropicHomogeneous<DIM(data)>;                       \
  template bool operator==(const IsotropicHomogeneous<DIM(data)>& lhs,  \
                           const IsotropicHomogeneous<DIM(data)>& rhs); \
  template bool operator!=(const IsotropicHomogeneous<DIM(data)>& lhs,  \
                           const IsotropicHomogeneous<DIM(data)>& rhs);

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3))

#undef DIM
#undef INSTANTIATE


}  // namespace Elasticity::ConstitutiveRelations
