// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Elasticity/BoundaryConditions/LaserBeam.hpp"

#include <cmath>
#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
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

double factorial(const size_t n) {
  return (n == 1 or n == 0) ? 1.0 : n * factorial(n - 1);
}
}  // namespace

LaserBeam::LaserBeam(const double beam_width,
                     const std::vector<ModeInfo>& modes,
                     const Options::Context& context)
    : beam_width_(beam_width), modes_(modes) {
  // Check finiteness first: std::isfinite does not raise FE_INVALID, whereas
  // an ordered comparison with NaN does, and floating-point exceptions are
  // trapped in SpECTRE executables and tests.
  if (not std::isfinite(beam_width_) or beam_width_ <= 0.) {
    PARSE_ERROR(context, "The BeamWidth must be positive and finite, but is "
                             << beam_width_ << ".");
  }
  if (modes_.empty()) {
    PARSE_ERROR(context,
                "Modes must contain at least one mode, e.g. "
                "'- {N: 0, M: 0, Weight: 1.}' for a TEM00 beam.");
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
  const auto n_dot_x = get<0>(face_normal) * get<0>(x) +
                       get<1>(face_normal) * get<1>(x) +
                       get<2>(face_normal) * get<2>(x);

  const DataVector r_vec_0 = get<0>(x) - n_dot_x * get<0>(face_normal);
  const DataVector r_vec_1 = get<1>(x) - n_dot_x * get<1>(face_normal);
  const DataVector r_vec_2 = get<2>(x) - n_dot_x * get<2>(face_normal);

  // In-plane basis (u, v), see the class documentation
  const double nx_ref = get<0>(face_normal)[0];

  double tx = 1.0, ty = 0.0, tz = 0.0;
  if (std::abs(nx_ref) > 0.9) {
    tx = 0.0;
    ty = 1.0;
  }

  const auto t_dot_n = tx * get<0>(face_normal) + ty * get<1>(face_normal) +
                       tz * get<2>(face_normal);

  DataVector u_vec_0 = tx - t_dot_n * get<0>(face_normal);
  DataVector u_vec_1 = ty - t_dot_n * get<1>(face_normal);
  DataVector u_vec_2 = tz - t_dot_n * get<2>(face_normal);

  // Normalize basis_u
  const DataVector u_mag =
      sqrt(square(u_vec_0) + square(u_vec_1) + square(u_vec_2));

  for (size_t i = 0; i < u_mag.size(); ++i) {
    u_vec_0[i] /= u_mag[i];
    u_vec_1[i] /= u_mag[i];
    u_vec_2[i] /= u_mag[i];
  }

  // basis_v = n x basis_u
  const auto v_vec_0 =
      get<1>(face_normal) * u_vec_2 - get<2>(face_normal) * u_vec_1;
  const auto v_vec_1 =
      get<2>(face_normal) * u_vec_0 - get<0>(face_normal) * u_vec_2;
  const auto v_vec_2 =
      get<0>(face_normal) * u_vec_1 - get<1>(face_normal) * u_vec_0;

  // Coordinates of x in the (u, v) basis
  const DataVector coord_u =
      r_vec_0 * u_vec_0 + r_vec_1 * u_vec_1 + r_vec_2 * u_vec_2;
  const DataVector coord_v =
      r_vec_0 * v_vec_0 + r_vec_1 * v_vec_1 + r_vec_2 * v_vec_2;

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
    DataVector mode_shape = square(H_n * H_m);

    // Normalization integral for this mode (n, m), so that each mode with
    // unit weight exerts unit total force:
    // Integral[ (H_n(u/r_0) H_m(v/r_0))^2 exp(-r^2/r_0^2) dA ]
    //   = pi r_0^2 2^(n+m) n! m!
    const double normalization_const = M_PI * square(beam_width_) *
                                       pow(2.0, mode.n + mode.m) *
                                       factorial(mode.n) * factorial(mode.m);

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
