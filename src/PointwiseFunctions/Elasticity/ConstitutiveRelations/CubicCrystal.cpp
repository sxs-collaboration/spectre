// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/CubicCrystal.hpp"

#include <array>
#include <pup.h>  // IWYU pragma: keep

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace Elasticity::ConstitutiveRelations {

CubicCrystal::CubicCrystal(const double c_11, const double c_12,
                           const double c_44) noexcept
    : c_11_(c_11), c_12_(c_12), c_44_(c_44) {
  ASSERT(
      c_11_ >= c_12_,
      "c_11 must be bigger than c_12, but are c_11="
          << c_11 << " and c_12=" << c_12
          << ". This is because the youngs_modulus "
             "must be positive and the poisson ratio smaller or equal to 0.5.");
}

void CubicCrystal::stress(const gsl::not_null<tnsr::II<DataVector, 3>*> stress,
                          const tnsr::ii<DataVector, 3>& strain,
                          const tnsr::I<DataVector, 3>& /*x*/) const noexcept {
  get<2, 2>(*stress) = -get<0, 0>(strain);
  for (size_t i = 1; i < 3; ++i) {
    stress->get(2, 2) -= strain.get(i, i);
  }
  stress->get(2, 2) *= c_12_;
  for (size_t i = 0; i < 3; ++i) {
    stress->get(i, i) = stress->get(2, 2) - (c_11_ - c_12_) * strain.get(i, i);
    for (size_t j = 0; j < i; ++j) {
      stress->get(i, j) = -2. * c_44_ * strain.get(i, j);
    }
  }
}

double CubicCrystal::c_11() const noexcept { return c_11_; }

double CubicCrystal::c_12() const noexcept { return c_12_; }

double CubicCrystal::c_44() const noexcept { return c_44_; }

// through \lambda = c_{12} = \frac{E\nu}{(1+\nu)(1-2\nu)}
double CubicCrystal::youngs_modulus() const noexcept {
  return (c_11_ + 2. * c_12_) * (c_11_ - c_12_) / (c_11_ + c_12_);
}

double CubicCrystal::poisson_ratio() const noexcept {
  return 1. / (1. + (c_11_ / c_12_));
}

/// \cond
PUP::able::PUP_ID CubicCrystal::my_PUP_ID = 0;
/// \endcond

void CubicCrystal::pup(PUP::er& p) noexcept {
  p | c_11_;
  p | c_12_;
  p | c_44_;
}

bool operator==(const CubicCrystal& lhs, const CubicCrystal& rhs) noexcept {
  return lhs.c_11() == rhs.c_11() and lhs.c_12() == rhs.c_12() and
         lhs.c_44() == rhs.c_44();
}

bool operator!=(const CubicCrystal& lhs, const CubicCrystal& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace Elasticity::ConstitutiveRelations
