// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/ForceFree/MagnetosphericWald.hpp"

#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/Options.hpp"
#include "Options/ParseError.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace ForceFree::AnalyticData {

MagnetosphericWald::MagnetosphericWald(const double spin,
                                       const Options::Context& context)
    : spin_(spin),
      background_spacetime_{1.0, {{0.0, 0.0, spin_}}, {{0.0, 0.0, 0.0}}} {
  if (abs(spin_) > 1.0) {
    PARSE_ERROR(context, "The magnitude of the dimensionless spin ("
                             << spin_ << ") cannot be bigger than 1.0");
  }
}

MagnetosphericWald::MagnetosphericWald(CkMigrateMessage* msg)
    : InitialData(msg) {}

std::unique_ptr<evolution::initial_data::InitialData>
MagnetosphericWald::get_clone() const {
  return std::make_unique<MagnetosphericWald>(*this);
}

void MagnetosphericWald::pup(PUP::er& p) {
  InitialData::pup(p);
  p | spin_;
  p | background_spacetime_;
}

// NOLINTNEXTLINE
PUP::able::PUP_ID MagnetosphericWald::my_PUP_ID = 0;

tuples::TaggedTuple<Tags::TildeE> MagnetosphericWald::variables(
    const tnsr::I<DataVector, 3>& x, tmpl::list<Tags::TildeE> /*meta*/) {
  return {make_with_value<tnsr::I<DataVector, 3>>(x, 0.0)};
}

tuples::TaggedTuple<Tags::TildeB> MagnetosphericWald::variables(
    const tnsr::I<DataVector, 3>& x, tmpl::list<Tags::TildeB> /*meta*/) const {
  auto tilde_b =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(x, 0.0);

  const double a_squared = square(spin_);

  const auto& x_bar = get<0>(x);
  const auto& y_bar = get<1>(x);
  const auto& z_bar = get<2>(x);

  const DataVector r_squared = get(dot_product(x, x));
  const DataVector r = sqrt(r_squared);
  const DataVector r_to_the_fourth = square(r_squared);
  const DataVector z_squared = square(z_bar);

  const DataVector temp1 = r_to_the_fourth + a_squared * z_squared;
  const DataVector temp2 =
      square(temp1) + 2.0 * r_to_the_fourth * r * (r_squared - a_squared);
  const DataVector temp3 = temp1 * (r_squared - z_squared) -
                           4.0 * r_to_the_fourth * (r_squared + z_squared);

  get<0>(tilde_b) =
      (spin_ * x_bar - r * y_bar) * temp2 + spin_ * r * x_bar * temp3;
  get<1>(tilde_b) =
      (r * x_bar + spin_ * y_bar) * temp2 + spin_ * r * y_bar * temp3;

  get<0>(tilde_b) *= spin_ * z_bar / (r_to_the_fourth * square(temp1));
  get<1>(tilde_b) *= spin_ * z_bar / (r_to_the_fourth * square(temp1));

  get<2>(tilde_b) = 1.0 + (a_squared * z_squared / r_to_the_fourth);
  get<2>(tilde_b) +=
      a_squared *
      (1.0 - z_squared * (a_squared + z_squared) *
                 (5.0 * r_to_the_fourth + a_squared * z_squared) /
                 square(temp1)) /
      (r_squared * r);

  return tilde_b;
}

tuples::TaggedTuple<Tags::TildePsi> MagnetosphericWald::variables(
    const tnsr::I<DataVector, 3>& x, tmpl::list<Tags::TildePsi> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(x, 0.0)};
}

tuples::TaggedTuple<Tags::TildePhi> MagnetosphericWald::variables(
    const tnsr::I<DataVector, 3>& x, tmpl::list<Tags::TildePhi> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(x, 0.0)};
}

tuples::TaggedTuple<Tags::TildeQ> MagnetosphericWald::variables(
    const tnsr::I<DataVector, 3>& x, tmpl::list<Tags::TildeQ> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(x, 0.0)};
}

bool operator==(const MagnetosphericWald& lhs, const MagnetosphericWald& rhs) {
  return lhs.spin_ == rhs.spin_ and
         lhs.background_spacetime_ == rhs.background_spacetime_;
}

bool operator!=(const MagnetosphericWald& lhs, const MagnetosphericWald& rhs) {
  return not(lhs == rhs);
}

}  // namespace ForceFree::AnalyticData
