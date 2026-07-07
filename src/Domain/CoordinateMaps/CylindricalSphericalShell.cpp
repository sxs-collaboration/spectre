// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/CylindricalSphericalShell.hpp"

#include <cmath>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain::CoordinateMaps {

CylindricalSphericalShell::CylindricalSphericalShell(const double x_inner_lower,
                                                     const double x_inner_upper,
                                                     const double x_outer_lower,
                                                     const double x_outer_upper,
                                                     const double r_inner,
                                                     const double r_sphere)
    : x_inner_lower_(x_inner_lower),
      x_inner_upper_(x_inner_upper),
      x_outer_lower_(x_outer_lower),
      x_outer_upper_(x_outer_upper),
      r_inner_(r_inner),
      r_sphere_(r_sphere) {
  ASSERT(x_inner_lower < x_inner_upper, "x_inner_lower ("
                                            << x_inner_lower
                                            << ") must be less than "
                                               "x_inner_upper ("
                                            << x_inner_upper << ").");
  ASSERT(x_outer_lower < x_outer_upper, "x_outer_lower ("
                                            << x_outer_lower
                                            << ") must be less than "
                                               "x_outer_upper ("
                                            << x_outer_upper << ").");
  ASSERT(r_inner > 0.0, "r_inner (" << r_inner << ") must be positive.");
  ASSERT(std::abs(x_outer_lower) < r_sphere,
         "|x_outer_lower| (" << std::abs(x_outer_lower)
                             << ") must be less than r_sphere (" << r_sphere
                             << ").");
  ASSERT(std::abs(x_outer_upper) < r_sphere,
         "|x_outer_upper| (" << std::abs(x_outer_upper)
                             << ") must be less than r_sphere (" << r_sphere
                             << ").");
#ifdef SPECTRE_DEBUG
  const double r_outer_lower =
      std::sqrt(r_sphere * r_sphere - x_outer_lower * x_outer_lower);
  const double r_outer_upper =
      std::sqrt(r_sphere * r_sphere - x_outer_upper * x_outer_upper);
  ASSERT(r_inner < r_outer_lower, "r_inner ("
                                      << r_inner
                                      << ") must be less than r_outer at "
                                         "x_outer_lower, which is "
                                      << r_outer_lower << ".");
  ASSERT(r_inner < r_outer_upper, "r_inner ("
                                      << r_inner
                                      << ") must be less than r_outer at "
                                         "x_outer_upper, which is "
                                      << r_outer_upper << ".");
#endif  // SPECTRE_DEBUG
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> CylindricalSphericalShell::operator()(
    const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& xi = source_coords[0];
  const ReturnType& eta = source_coords[1];
  const ReturnType& zeta = source_coords[2];
  const ReturnType alpha = 0.5 * (xi + 1.0);
  const ReturnType beta = 0.5 * (zeta + 1.0);
  const ReturnType x_inner =
      x_inner_lower_ + beta * (x_inner_upper_ - x_inner_lower_);
  const ReturnType x_outer =
      x_outer_lower_ + beta * (x_outer_upper_ - x_outer_lower_);
  const ReturnType r_outer = sqrt(r_sphere_ * r_sphere_ - x_outer * x_outer);
  const ReturnType x = (1.0 - alpha) * x_inner + alpha * x_outer;
  const ReturnType r = r_inner_ * (1.0 - alpha) + alpha * r_outer;
  return {x, r * cos(eta), r * sin(eta)};
}

std::optional<std::array<double, 3>> CylindricalSphericalShell::inverse(
    const std::array<double, 3>& target_coords) const {
  const double x = target_coords[0];
  const double y = target_coords[1];
  const double z = target_coords[2];
  const double r = std::sqrt(y * y + z * z);
  const double eta = std::atan2(z, y);
  const double delta_x_inner = x_inner_upper_ - x_inner_lower_;
  const double delta_x_outer = x_outer_upper_ - x_outer_lower_;

  // Solve for beta in [0,1]
  //
  // For a given beta, the radial blending parameter alpha satisfies
  //   r = (1-alpha)*r_inner_ + alpha*r_o(beta)
  //   x = (1-alpha)*x_i(beta) + alpha*x_o(beta)
  // Eliminating alpha gives the scalar equation F(beta) = 0:
  //   F(beta) = (r - r_inner_)*(x_o(beta) - x_i(beta))
  //             - (x - x_i(beta))*(r_o(beta) - r_inner_)
  const auto f_beta = [&](const double beta) {
    const double x_i = x_inner_lower_ + beta * delta_x_inner;
    const double x_o = x_outer_lower_ + beta * delta_x_outer;
    const double r_o = std::sqrt(r_sphere_ * r_sphere_ - x_o * x_o);
    return (r - r_inner_) * (x_o - x_i) - (x - x_i) * (r_o - r_inner_);
  };

  const double f_lower = f_beta(0.0);
  const double f_upper = f_beta(1.0);

  // When the source point lies exactly on a corner (xi= \pm 1, zeta= \pm 1),
  // the mathematical value of F is zero at beta=0 or beta=1, but floating-point
  // roundoff requires a relative tolerance to snap beta to the endpoint.
  const double f_scale = std::max(std::abs(f_lower), std::abs(f_upper));
  double beta{};
  if (f_lower * f_upper > 0.0) {
    if (equal_within_roundoff(f_lower, 0.0, 1.0e-10, f_scale)) {
      beta = 0.0;
    } else if (equal_within_roundoff(f_upper, 0.0, 1.0e-10, f_scale)) {
      beta = 1.0;
    } else {
      // Root not bracketed: the point lies outside the axial range of the map.
      return std::nullopt;
    }
  } else {
    constexpr double tol = 1.0e-15;
    try {
      beta = RootFinder::toms748(f_beta, 0.0, 1.0, f_lower, f_upper, tol, tol);
      // LCOV_EXCL_START
    } catch (std::exception&) {
      ERROR(
          "CylindricalSphericalShell::inverse: toms748 failed after "
          "bracketing: F(0)="
          << f_lower << " F(1)=" << f_upper);
      // LCOV_EXCL_STOP
    }
  }

  const double x_o = x_outer_lower_ + beta * delta_x_outer;
  const double r_o = std::sqrt(r_sphere_ * r_sphere_ - x_o * x_o);
  const double alpha = (r - r_inner_) / (r_o - r_inner_);

  if (alpha < -1.0e-10 or alpha > 1.0 + 1.0e-10) {
    return std::nullopt;
  }

  const double xi = 2.0 * alpha - 1.0;
  const double zeta = 2.0 * beta - 1.0;
  return std::array<double, 3>{xi, eta, zeta};
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
CylindricalSphericalShell::jacobian(
    const std::array<T, 3>& source_coords) const {
  using DataType = tt::remove_cvref_wrap_t<T>;
  const DataType& xi = source_coords[0];
  const DataType& eta = source_coords[1];
  const DataType& zeta = source_coords[2];
  tnsr::Ij<DataType, 3, Frame::NoFrame> jac{
      make_with_value<DataType>(dereference_wrapper(xi), 0.0)};
  const DataType alpha = 0.5 * (xi + 1.0);
  const DataType beta = 0.5 * (zeta + 1.0);
  const double delta_x_inner = x_inner_upper_ - x_inner_lower_;
  const double delta_x_outer = x_outer_upper_ - x_outer_lower_;
  const DataType x_inner = x_inner_lower_ + beta * delta_x_inner;
  const DataType x_outer = x_outer_lower_ + beta * delta_x_outer;
  const DataType r_outer = sqrt(r_sphere_ * r_sphere_ - x_outer * x_outer);
  const DataType r = r_inner_ * (1.0 - alpha) + alpha * r_outer;
  // Partial derivatives of x(alpha,beta) and r(alpha,beta):
  //   dx/dxi  = (x_outer - x_inner)/2
  //   dx/dzeta = ((1-alpha)*delta_x_inner + alpha*delta_x_outer)/2
  //   dr/dxi  = (r_outer - r_inner_)/2
  //   dr/dzeta = alpha * (-x_outer/r_outer) * delta_x_outer / 2
  const DataType dx_dxi = 0.5 * (x_outer - x_inner);
  const DataType dx_dzeta =
      0.5 * ((1.0 - alpha) * delta_x_inner + alpha * delta_x_outer);
  const DataType dr_dxi = 0.5 * (r_outer - r_inner_);
  const DataType dr_dzeta = 0.5 * alpha * (-x_outer / r_outer) * delta_x_outer;
  const DataType cos_eta = cos(eta);
  const DataType sin_eta = sin(eta);
  // Row 0: d(x)/d(xi, eta, zeta)
  get<0, 0>(jac) = dx_dxi;
  get<0, 1>(jac) = 0.0;
  get<0, 2>(jac) = dx_dzeta;
  // Row 1: d(y)/d(xi, eta, zeta),  y = r cos(eta)
  get<1, 0>(jac) = cos_eta * dr_dxi;
  get<1, 1>(jac) = -r * sin_eta;
  get<1, 2>(jac) = cos_eta * dr_dzeta;
  // Row 2: d(z)/d(xi, eta, zeta),  z = r sin(eta)
  get<2, 0>(jac) = sin_eta * dr_dxi;
  get<2, 1>(jac) = r * cos_eta;
  get<2, 2>(jac) = sin_eta * dr_dzeta;
  return jac;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
CylindricalSphericalShell::inv_jacobian(
    const std::array<T, 3>& source_coords) const {
  using DataType = tt::remove_cvref_wrap_t<T>;
  const DataType& xi = source_coords[0];
  const DataType& eta = source_coords[1];
  const DataType& zeta = source_coords[2];
  tnsr::Ij<DataType, 3, Frame::NoFrame> inv_jac{
      make_with_value<DataType>(dereference_wrapper(xi), 0.0)};
  const DataType alpha = 0.5 * (xi + 1.0);
  const DataType beta = 0.5 * (zeta + 1.0);
  const double delta_x_inner = x_inner_upper_ - x_inner_lower_;
  const double delta_x_outer = x_outer_upper_ - x_outer_lower_;
  const DataType x_inner = x_inner_lower_ + beta * delta_x_inner;
  const DataType x_outer = x_outer_lower_ + beta * delta_x_outer;
  const DataType r_outer = sqrt(r_sphere_ * r_sphere_ - x_outer * x_outer);
  const DataType r = r_inner_ * (1.0 - alpha) + alpha * r_outer;
  const DataType dx_dxi = 0.5 * (x_outer - x_inner);
  const DataType dx_dzeta =
      0.5 * ((1.0 - alpha) * delta_x_inner + alpha * delta_x_outer);
  const DataType dr_dxi = 0.5 * (r_outer - r_inner_);
  const DataType dr_dzeta = 0.5 * alpha * (-x_outer / r_outer) * delta_x_outer;
  const DataType cos_eta = cos(eta);
  const DataType sin_eta = sin(eta);
  // The Jacobian determinant is det(J) = r * D_reduced where
  //   D_reduced = dx_dzeta * dr_dxi - dx_dxi * dr_dzeta.
  // The inverse Jacobian entries follow from the adjugate formula.
  const DataType D_reduced = dx_dzeta * dr_dxi - dx_dxi * dr_dzeta;
  // Row 0: d(xi)/d(x, y, z)
  get<0, 0>(inv_jac) = -dr_dzeta / D_reduced;
  get<0, 1>(inv_jac) = dx_dzeta * cos_eta / D_reduced;
  get<0, 2>(inv_jac) = dx_dzeta * sin_eta / D_reduced;
  // Row 1: d(eta)/d(x, y, z)
  get<1, 0>(inv_jac) = 0.0;
  get<1, 1>(inv_jac) = -sin_eta / r;
  get<1, 2>(inv_jac) = cos_eta / r;
  // Row 2: d(zeta)/d(x, y, z)
  get<2, 0>(inv_jac) = dr_dxi / D_reduced;
  get<2, 1>(inv_jac) = -dx_dxi * cos_eta / D_reduced;
  get<2, 2>(inv_jac) = -dx_dxi * sin_eta / D_reduced;
  return inv_jac;
}

void CylindricalSphericalShell::pup(PUP::er& p) {
  size_t version = 0;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version >= 0) {
    p | x_inner_lower_;
    p | x_inner_upper_;
    p | x_outer_lower_;
    p | x_outer_upper_;
    p | r_inner_;
    p | r_sphere_;
  }
}

bool operator==(const CylindricalSphericalShell& lhs,
                const CylindricalSphericalShell& rhs) {
  return lhs.x_inner_lower_ == rhs.x_inner_lower_ and
         lhs.x_inner_upper_ == rhs.x_inner_upper_ and
         lhs.x_outer_lower_ == rhs.x_outer_lower_ and
         lhs.x_outer_upper_ == rhs.x_outer_upper_ and
         lhs.r_inner_ == rhs.r_inner_ and lhs.r_sphere_ == rhs.r_sphere_;
}

bool operator!=(const CylindricalSphericalShell& lhs,
                const CylindricalSphericalShell& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>               \
  CylindricalSphericalShell::operator()(                                     \
      const std::array<DTYPE(data), 3>& source_coords) const;                \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  CylindricalSphericalShell::jacobian(                                       \
      const std::array<DTYPE(data), 3>& source_coords) const;                \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  CylindricalSphericalShell::inv_jacobian(                                   \
      const std::array<DTYPE(data), 3>& source_coords) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))

#undef DTYPE
#undef INSTANTIATE

}  // namespace domain::CoordinateMaps
