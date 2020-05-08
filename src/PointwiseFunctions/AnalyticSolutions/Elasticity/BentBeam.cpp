// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Elasticity/BentBeam.hpp"

#include <algorithm>
#include <array>
#include <pup.h>  // IWYU pragma: keep

#include "DataStructures/DataVector.hpp"     // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace Elasticity {
namespace Solutions {

BentBeam::BentBeam(double length, double height, double bending_moment,
                   constitutive_relation_type constitutive_relation) noexcept
    : length_(length),
      height_(height),
      bending_moment_(bending_moment),
      constitutive_relation_(std::move(constitutive_relation)) {}

tuples::TaggedTuple<Tags::Displacement<2>> BentBeam::variables(
    const tnsr::I<DataVector, 2>& x,
    tmpl::list<Tags::Displacement<2>> /*meta*/) const noexcept {
  const double youngs_modulus = constitutive_relation_.youngs_modulus();
  const double poisson_ratio = constitutive_relation_.poisson_ratio();
  const double prefactor =
      12. * bending_moment_ / (youngs_modulus * cube(height_));
  return {tnsr::I<DataVector, 2>{
      {{-prefactor * get<0>(x) * get<1>(x),
        prefactor / 2. *
            (square(get<0>(x)) + poisson_ratio * square(get<1>(x)) -
             square(length_) / 4.)}}}};
}

tuples::TaggedTuple<Tags::Strain<2>> BentBeam::variables(
    const tnsr::I<DataVector, 2>& x, tmpl::list<Tags::Strain<2>> /*meta*/) const
    noexcept {
  const double youngs_modulus = constitutive_relation_.youngs_modulus();
  const double poisson_ratio = constitutive_relation_.poisson_ratio();
  const double prefactor =
      12. * bending_moment_ / (youngs_modulus * cube(height_));
  auto result = make_with_value<tnsr::ii<DataVector, 2>>(x, 0.);
  get<0, 0>(result) = -prefactor * get<1>(x);
  get<1, 1>(result) = prefactor * poisson_ratio * get<1>(x);
  return {std::move(result)};
}

tuples::TaggedTuple<Tags::Stress<2>> BentBeam::variables(
    const tnsr::I<DataVector, 2>& x, tmpl::list<Tags::Stress<2>> /*meta*/) const
    noexcept {
  auto result = make_with_value<tnsr::II<DataVector, 2>>(x, 0.);
  get<0, 0>(result) = 12. * bending_moment_ / cube(height_) * get<1>(x);
  return {std::move(result)};
}

tuples::TaggedTuple<::Tags::FixedSource<Tags::Displacement<2>>>
BentBeam::variables(
    const tnsr::I<DataVector, 2>& x,
    tmpl::list<::Tags::FixedSource<Tags::Displacement<2>>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::I<DataVector, 2>>(x, 0.)};
}

void BentBeam::pup(PUP::er& p) noexcept {
  p | length_;
  p | height_;
  p | bending_moment_;
  p | constitutive_relation_;
}

bool operator==(const BentBeam& lhs, const BentBeam& rhs) noexcept {
  return lhs.length_ == rhs.length_ and lhs.height_ == rhs.height_ and
         lhs.bending_moment_ == rhs.bending_moment_ and
         lhs.constitutive_relation_ == rhs.constitutive_relation_;
}

bool operator!=(const BentBeam& lhs, const BentBeam& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace Solutions
}  // namespace Elasticity
