// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Elasticity/BoundaryConditions/LaserBeam.hpp"

#include <cmath>
#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/CrossProduct.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Elasticity::BoundaryConditions {

namespace {
// Recursive calculation of Hermite polynomials H_n(x)
DataVector hermite_polynomial(const size_t n, const DataVector& x) {
  if (n == 0) {
    return make_with_value<DataVector>(x, 1.0);
  }
  if (n == 1) {
    return 2.0 * x;
  }
  DataVector h_prev = make_with_value<DataVector>(x, 1.0);  // H_0
  DataVector h_curr = 2.0 * x;                              // H_1
  DataVector h_next = make_with_value<DataVector>(x, 0.0);

  for (size_t i = 1; i < n; ++i) {
    h_next = 2.0 * x * h_curr - 2.0 * static_cast<double>(i) * h_prev;
    h_prev = h_curr;
    h_curr = h_next;
  }
  return h_curr;
}
}  // namespace

LaserBeam::LaserBeam(const double beam_width,
                     const std::vector<ModeInfo>& modes,
                     const Options::Context& context)
    : beam_width_(beam_width), modes_(modes) {
  if (beam_width_ <= 0.) {
    PARSE_ERROR(context, "The BeamWidth must be positive, but is "
                             << beam_width_ << ".");
  }
}

void LaserBeam::apply(
    const gsl::not_null<tnsr::I<DataVector, 3>*> /*displacement*/,
    const gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_minus_stress,
    const tnsr::iJ<DataVector, 3>& /*deriv_displacement*/,
    const tnsr::I<DataVector, 3>& x,
    const tnsr::i<DataVector, 3>& face_normal) const {
  // The (u, v) basis below is chosen from the normal at the first point, so
  // the normal must be the same at all points (up to roundoff). This depends
  // on the domain chosen in the input file, so it is checked in all build
  // types.
  for (size_t d = 0; d < 3; ++d) {
    if (not(max(abs(face_normal.get(d) - face_normal.get(d)[0])) < 1.e-12)) {
      ERROR(
          "LaserBeam assumes a flat surface (constant normal), but "
          "component "
          << d << " of the face normal varies across the face.");
    }
  }

  // Part of x perpendicular to the normal
  const auto n_dot_x = dot_product(x, face_normal);
  auto r = x;
  for (size_t d = 0; d < 3; ++d) {
    r.get(d) -= get(n_dot_x) * face_normal.get(d);
  }

  // In-plane basis (u, v), see the class documentation. u starts as the
  // reference tangent t and is then projected into the surface.
  auto u = make_with_value<tnsr::I<DataVector, 3>>(x, 0.);
  if (std::abs(get<0>(face_normal)[0]) > 0.9) {
    get<1>(u) = 1.;
  } else {
    get<0>(u) = 1.;
  }
  const auto t_dot_n = dot_product(u, face_normal);
  for (size_t d = 0; d < 3; ++d) {
    u.get(d) -= get(t_dot_n) * face_normal.get(d);
  }
  const auto u_mag = magnitude(u);
  for (size_t d = 0; d < 3; ++d) {
    u.get(d) /= get(u_mag);
  }
  // v = n x u
  const auto v = cross_product(face_normal, u);

  // Coordinates of x in the (u, v) basis
  const DataVector coord_u = get(dot_product(r, u));
  const DataVector coord_v = get(dot_product(r, v));

  // Incoherent sum of the mode intensities.
  // The beam width is r_0, the 1/e intensity radius (w = sqrt(2) r_0 is the
  // 1/e^2 intensity radius). The Hermite argument sqrt(2) u / w = u / r_0.
  const double one_over_r0 = 1.0 / beam_width_;

  // Accumulator for the final intensity profile (P_total)
  DataVector summed_intensity = make_with_value<DataVector>(coord_u, 0.0);

  for (const auto& mode : modes_) {
    const DataVector H_n = hermite_polynomial(mode.n, coord_u * one_over_r0);
    const DataVector H_m = hermite_polynomial(mode.m, coord_v * one_over_r0);

    // Calculate the "Shape" of the intensity for this mode: (H_n * H_m)^2
    // We square it immediately because we are adding Intensities, not Fields.
    const DataVector mode_shape = square(H_n * H_m);

    // Normalization integral for this mode (n, m), so that each mode with
    // unit weight exerts unit total force:
    // Integral[ (H_n(u/r_0) H_m(v/r_0))^2 exp(-r^2/r_0^2) dA ]
    //   = pi r_0^2 2^(n+m) n! m!
    const double normalization_const =
        M_PI * square(beam_width_) *
        pow(2.0, static_cast<double>(mode.n + mode.m)) *
        static_cast<double>(factorial(mode.n)) *
        static_cast<double>(factorial(mode.m));

    // Add this mode's contribution to the total pressure
    // P_total += weight * (Shape / Normalization)
    // If weight is negative, this subtracts pressure (virtual work).
    summed_intensity += mode.weight * (mode_shape / normalization_const);
  }

  // Gaussian envelope.
  // The term exp(-r^2/r_0^2) is common to all Hermite-Gaussian intensities
  // and was factored out of the integration constant above.
  const DataVector beam_profile =
      summed_intensity *
      exp(-(square(coord_u) + square(coord_v)) / square(beam_width_));

  get<0>(*n_dot_minus_stress) = -beam_profile * get<0>(face_normal);
  get<1>(*n_dot_minus_stress) = -beam_profile * get<1>(face_normal);
  get<2>(*n_dot_minus_stress) = -beam_profile * get<2>(face_normal);
}

void LaserBeam::apply_linearized(
    const gsl::not_null<tnsr::I<DataVector, 3>*> /*displacement*/,
    const gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_minus_stress,
    const tnsr::iJ<DataVector, 3>& /*deriv_displacement*/) {
  get<0>(*n_dot_minus_stress) = 0.;
  get<1>(*n_dot_minus_stress) = 0.;
  get<2>(*n_dot_minus_stress) = 0.;
}

// Comparison
bool operator==(const LaserBeam& lhs, const LaserBeam& rhs) {
  return lhs.beam_width() == rhs.beam_width() and lhs.modes() == rhs.modes();
}

bool operator!=(const LaserBeam& lhs, const LaserBeam& rhs) {
  return not(lhs == rhs);
}

bool operator==(const LaserBeam::ModeInfo& lhs,
                const LaserBeam::ModeInfo& rhs) {
  return lhs.n == rhs.n and lhs.m == rhs.m and lhs.weight == rhs.weight;
}
bool operator!=(const LaserBeam::ModeInfo& lhs,
                const LaserBeam::ModeInfo& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID LaserBeam::my_PUP_ID = 0;  // NOLINT

}  // namespace Elasticity::BoundaryConditions
