// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/ForceFree/RotatingDipole.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace ForceFree::AnalyticData {

RotatingDipole::RotatingDipole(const double vector_potential_amplitude,
                               const double varpi0, const double delta,
                               const double angular_velocity,
                               const double tilt_angle,
                               const Options::Context& context)
    : vector_potential_amplitude_(vector_potential_amplitude),
      varpi0_(varpi0),
      delta_(delta),
      angular_velocity_(angular_velocity),
      tilt_angle_(tilt_angle) {
  if (varpi0 < 0.0) {
    PARSE_ERROR(context, "The length constant varpi0 ("
                             << varpi0_ << ") cannot be negative");
  }
  if (delta < 0.0) {
    PARSE_ERROR(context,
                "The small number delta (" << delta_ << ") cannot be negative");
  }
  if (abs(angular_velocity) >= 1.0) {
    PARSE_ERROR(context, "The rotation angular velocity ("
                             << angular_velocity_
                             << ") must be between -1.0 and 1.0");
  }
  if ((tilt_angle < 0.0) or (tilt_angle > M_PI)) {
    PARSE_ERROR(context, "The rotator tilt angle ("
                             << tilt_angle_ << ") must be between 0 and Pi");
  }
}

RotatingDipole::RotatingDipole(CkMigrateMessage* msg) : InitialData(msg) {}

std::unique_ptr<evolution::initial_data::InitialData>
RotatingDipole::get_clone() const {
  return std::make_unique<RotatingDipole>(*this);
}

void RotatingDipole::pup(PUP::er& p) {
  InitialData::pup(p);
  p | vector_potential_amplitude_;
  p | varpi0_;
  p | delta_;
  p | angular_velocity_;
  p | tilt_angle_;
  p | background_spacetime_;
}

PUP::able::PUP_ID RotatingDipole::my_PUP_ID = 0;

tuples::TaggedTuple<Tags::TildeE> RotatingDipole::variables(
    const tnsr::I<DataVector, 3>& coords, tmpl::list<Tags::TildeE> /*meta*/) {
  return {make_with_value<tnsr::I<DataVector, 3>>(coords, 0.0)};
}

tuples::TaggedTuple<Tags::TildeB> RotatingDipole::variables(
    const tnsr::I<DataVector, 3>& coords,
    tmpl::list<Tags::TildeB> /*meta*/) const {
  auto result = make_with_value<tnsr::I<DataVector, 3>>(coords, 0.0);

  const double sin_alpha = sin(tilt_angle_);
  const double cos_alpha = cos(tilt_angle_);

  // Coordinates and magnetic fields in the tilted axis
  const auto& x = get<0>(coords);
  const auto& y = get<1>(coords);
  const auto& z = get<2>(coords);
  const DataVector x_prime = cos_alpha * x - sin_alpha * z;
  const DataVector z_prime = sin_alpha * x + cos_alpha * z;

  auto tilde_b_prime = make_with_value<tnsr::I<DataVector, 3>>(coords, 0.0);

  // Regularized dipole field
  const DataVector r_squared = get(dot_product(coords, coords));
  const DataVector one_over_radius_factor =
      1.0 / pow<5>(sqrt(r_squared + square(delta_)));
  get<0>(tilde_b_prime) = 3.0 * x_prime * z_prime * one_over_radius_factor;
  get<1>(tilde_b_prime) = 3.0 * y * z_prime * one_over_radius_factor;
  get<2>(tilde_b_prime) =
      (3.0 * square(z_prime) - r_squared + 2.0 * square(delta_)) *
      one_over_radius_factor;

  // Rotation
  get<0>(result) =
      cos_alpha * get<0>(tilde_b_prime) + sin_alpha * get<2>(tilde_b_prime);
  get<1>(result) = get<1>(tilde_b_prime);
  get<2>(result) =
      -sin_alpha * get<0>(tilde_b_prime) + cos_alpha * get<2>(tilde_b_prime);

  return result;
}

tuples::TaggedTuple<Tags::TildePsi> RotatingDipole::variables(
    const tnsr::I<DataVector, 3>& coords, tmpl::list<Tags::TildePsi> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(coords, 0.0)};
}

tuples::TaggedTuple<Tags::TildePhi> RotatingDipole::variables(
    const tnsr::I<DataVector, 3>& coords, tmpl::list<Tags::TildePhi> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(coords, 0.0)};
}

tuples::TaggedTuple<Tags::TildeQ> RotatingDipole::variables(
    const tnsr::I<DataVector, 3>& coords, tmpl::list<Tags::TildeQ> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(coords, 0.0)};
}

std::optional<Scalar<DataVector>> RotatingDipole::interior_mask(
    const tnsr::I<DataVector, 3>& x) {
  std::optional<Scalar<DataVector>> result{};

  DataVector r_squared = get(dot_product(x, x));

  // NS radius rescaled to 1.0 on grid.
  const double ns_radius_squared = square(1.0);

  if (min(r_squared) < ns_radius_squared) {
    // Allocate the mask vector
    const size_t num_grid_points = get<0>(x).size();
    result = Scalar<DataVector>{num_grid_points};

    for (size_t i = 0; i < num_grid_points; ++i) {
      if (r_squared[i] < ns_radius_squared) {
        // Interior
        get(result.value())[i] = -1.0;
      } else {
        // Exterior
        get(result.value())[i] = +1.0;
      }
    }
  }

  return result;
}

bool operator==(const RotatingDipole& lhs, const RotatingDipole& rhs) {
  return lhs.vector_potential_amplitude_ == rhs.vector_potential_amplitude_ and
         lhs.varpi0_ == rhs.varpi0_ and lhs.delta_ == rhs.delta_ and
         lhs.angular_velocity_ == rhs.angular_velocity_ and
         lhs.tilt_angle_ == rhs.tilt_angle_ and
         lhs.background_spacetime_ == rhs.background_spacetime_;
}

bool operator!=(const RotatingDipole& lhs, const RotatingDipole& rhs) {
  return not(lhs == rhs);
}

}  // namespace ForceFree::AnalyticData
