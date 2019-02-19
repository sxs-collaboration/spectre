// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Elasticity/BentBeam.hpp"

#include <cmath>
#include <pup.h>  // IWYU pragma: keep

#include "DataStructures/DataVector.hpp"     // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"
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
      3. * bending_moment_ / (2. * youngs_modulus * pow<3>(height_));
  return {tnsr::I<DataVector, 2>{
      {{-prefactor * get<0>(x) * get<1>(x),
        prefactor / 2. *
            (pow<2>(get<0>(x)) + poisson_ratio * pow<2>(get<1>(x)) -
             pow<2>(length_))}}}};
}

tuples::TaggedTuple<Tags::Stress<2>> BentBeam::variables(
    const tnsr::I<DataVector, 2>& x, tmpl::list<Tags::Stress<2>> /*meta*/) const
    noexcept {
  auto result = make_with_value<tnsr::II<DataVector, 2>>(x, 0.);
  get<0, 0>(result) = 3. * bending_moment_ / (2. * pow<3>(height_)) * get<1>(x);
  return {std::move(result)};
}

tuples::TaggedTuple<::Tags::Source<Tags::Displacement<2>>> BentBeam::variables(
    const tnsr::I<DataVector, 2>& x,
    tmpl::list<::Tags::Source<Tags::Displacement<2>>> /*meta*/) const noexcept {
  return {make_with_value<tnsr::I<DataVector, 2>>(x, 0.)};
}

tuples::TaggedTuple<::Tags::Source<Tags::Stress<2>>> BentBeam::variables(
    const tnsr::I<DataVector, 2>& x,
    tmpl::list<::Tags::Source<Tags::Stress<2>>> /*meta*/) const noexcept {
  return {make_with_value<tnsr::II<DataVector, 2>>(x, 0.)};
}

void BentBeam::pup(PUP::er& p) noexcept {
  p | length_;
  p | height_;
  p | bending_moment_;
  p | constitutive_relation_;
}

bool operator==(const BentBeam& lhs, const BentBeam& rhs) noexcept {
  return lhs.length() == rhs.length() and lhs.height() == rhs.height() and
         lhs.bending_moment() == rhs.bending_moment() and
         lhs.constitutive_relation() == rhs.constitutive_relation();
}

bool operator!=(const BentBeam& lhs, const BentBeam& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace Solutions
}  // namespace Elasticity
