// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/Tags.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace StrahlkorperTags {

template <typename Frame>
aliases::ThetaPhi<Frame> ThetaPhi<Frame>::function(
    const ::Strahlkorper<Frame>& strahlkorper) noexcept {
  auto temp = strahlkorper.ylm_spherepack().theta_phi_points();
  auto theta_phi = make_with_value<aliases::ThetaPhi<Frame>>(temp[0], 0.0);
  get<0>(theta_phi) = temp[0];
  get<1>(theta_phi) = temp[1];
  return theta_phi;
}

template <typename Frame>
aliases::OneForm<Frame> Rhat<Frame>::function(
    const db::item_type<ThetaPhi<Frame>>& theta_phi) noexcept {
  const auto& theta = get<0>(theta_phi);
  const auto& phi = get<1>(theta_phi);

  auto r_hat = make_with_value<aliases::OneForm<Frame>>(phi, 0.0);

  const DataVector sin_theta = sin(theta);
  get<0>(r_hat) = sin_theta * cos(phi);
  get<1>(r_hat) = sin_theta * sin(phi);
  get<2>(r_hat) = cos(theta);
  return r_hat;
}

template <typename Frame>
aliases::Jacobian<Frame> Jacobian<Frame>::function(
    const db::item_type<ThetaPhi<Frame>>& theta_phi) noexcept {
  const auto& theta = get<0>(theta_phi);
  const auto& phi = get<1>(theta_phi);
  const DataVector sin_phi = sin(phi);
  const DataVector cos_phi = cos(phi);
  const DataVector cos_theta = cos(theta);

  auto jac = make_with_value<aliases::Jacobian<Frame>>(phi, 0.0);
  get<0, 0>(jac) = cos_theta * cos_phi;  // 1/R dx/dth
  get<1, 0>(jac) = cos_theta * sin_phi;  // 1/R dy/dth
  get<2, 0>(jac) = -sin(theta);          // 1/R dz/dth
  get<0, 1>(jac) = -sin_phi;             // 1/(R sin(th)) dx/dph
  get<1, 1>(jac) = cos_phi;              // 1/(R sin(th)) dy/dph

  return jac;
}

template <typename Frame>
aliases::InvJacobian<Frame> InvJacobian<Frame>::function(
    const db::item_type<ThetaPhi<Frame>>& theta_phi) noexcept {
  const auto& theta = get<0>(theta_phi);
  const auto& phi = get<1>(theta_phi);
  const DataVector sin_phi = sin(phi);
  const DataVector cos_phi = cos(phi);
  const DataVector cos_theta = cos(theta);

  auto inv_jac = make_with_value<aliases::InvJacobian<Frame>>(phi, 0.0);
  get<0, 0>(inv_jac) = cos_theta * cos_phi;  // R dth/dx
  get<0, 1>(inv_jac) = cos_theta * sin_phi;  // R dth/dy
  get<0, 2>(inv_jac) = -sin(theta);          // R dth/dz
  get<1, 0>(inv_jac) = -sin_phi;             // R sin(th) dph/dx
  get<1, 1>(inv_jac) = cos_phi;              // R sin(th) dph/dy

  return inv_jac;
}

template <typename Frame>
aliases::InvHessian<Frame> InvHessian<Frame>::function(
    const db::item_type<ThetaPhi<Frame>>& theta_phi) noexcept {
  const auto& theta = get<0>(theta_phi);
  const auto& phi = get<1>(theta_phi);
  const DataVector sin_phi = sin(phi);
  const DataVector cos_phi = cos(phi);
  const DataVector sin_theta = sin(theta);
  const DataVector cos_theta = cos(theta);

  auto inv_hess = make_with_value<aliases::InvHessian<Frame>>(phi, 0.0);
  const DataVector sin_sq_theta = square(sin_theta);
  const DataVector cos_sq_theta = square(cos_theta);
  const DataVector sin_theta_cos_theta = sin_theta * cos_theta;
  const DataVector sin_sq_phi = square(sin_phi);
  const DataVector cos_sq_phi = square(cos_phi);
  const DataVector sin_phi_cos_phi = sin_phi * cos_phi;
  const DataVector csc_theta = 1.0 / sin_theta;
  const DataVector f1 = 1.0 + 2.0 * sin_sq_theta;
  const DataVector cot_theta = cos_theta * csc_theta;

  // R^2 d^2 th/(dx^2)
  get<0, 0, 0>(inv_hess) = cot_theta * (1.0 - cos_sq_phi * f1);
  // R^2 d^2 th/(dxdy)
  get<0, 0, 1>(inv_hess) = -cot_theta * sin_phi_cos_phi * f1;
  // R^2 d^2 th/(dxdz)
  get<0, 0, 2>(inv_hess) = (sin_sq_theta - cos_sq_theta) * cos_phi;
  // R^2 d^2 th/(dydx)
  get<0, 1, 0>(inv_hess) = get<0, 0, 1>(inv_hess);
  // R^2 d^2 th/(dy^2)
  get<0, 1, 1>(inv_hess) = cot_theta * (1.0 - sin_sq_phi * f1);
  // R^2 d^2 th/(dydz)
  get<0, 1, 2>(inv_hess) = (sin_sq_theta - cos_sq_theta) * sin_phi;
  // R^2 d^2 th/(dzdx)
  get<0, 2, 0>(inv_hess) = get<0, 0, 2>(inv_hess);
  // R^2 d^2 th/(dzdy)
  get<0, 2, 1>(inv_hess) = get<0, 1, 2>(inv_hess);
  // R^2 d^2 th/(dz^2)
  get<0, 2, 2>(inv_hess) = 2.0 * sin_theta_cos_theta;
  // R^2 d/dx (sin(th) dph/dx)
  get<1, 0, 0>(inv_hess) = sin_phi_cos_phi * (1.0 + sin_sq_theta) * csc_theta;
  // R^2 d/dx (sin(th) dph/dy)
  get<1, 0, 1>(inv_hess) = (sin_sq_phi - sin_sq_theta * cos_sq_phi) * csc_theta;
  // R^2 d/dy (sin(th) dph/dx)
  get<1, 1, 0>(inv_hess) = (sin_sq_theta * sin_sq_phi - cos_sq_phi) * csc_theta;
  // R^2 d/dy (sin(th) dph/dy)
  get<1, 1, 1>(inv_hess) = -get<1, 0, 0>(inv_hess);
  // R^2 d/dz (sin(th) dph/dx)
  get<1, 2, 0>(inv_hess) = cos_theta * sin_phi;
  // R^2 d/dz (sin(th) dph/dy)
  get<1, 2, 1>(inv_hess) = -cos_theta * cos_phi;

  return inv_hess;
}

template <typename Frame>
aliases::Vector<Frame> CartesianCoords<Frame>::function(
    const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
    const db::item_type<Rhat<Frame>>& r_hat) noexcept {
  auto coords = make_with_value<aliases::Vector<Frame>>(radius, 0.0);
  for (size_t d = 0; d < 3; ++d) {
    coords.get(d) = gsl::at(strahlkorper.center(), d) + r_hat.get(d) * radius;
  }
  return coords;
}

template <typename Frame>
aliases::OneForm<Frame> DxRadius<Frame>::function(
    const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
    const db::item_type<InvJacobian<Frame>>& inv_jac) noexcept {
  auto dx_radius = make_with_value<aliases::OneForm<Frame>>(radius, 0.0);
  const DataVector one_over_r = 1.0 / radius;
  const auto dr = strahlkorper.ylm_spherepack().gradient(radius);
  get<0>(dx_radius) =
      (get<0, 0>(inv_jac) * get<0>(dr) + get<1, 0>(inv_jac) * get<1>(dr)) *
      one_over_r;
  get<1>(dx_radius) =
      (get<0, 1>(inv_jac) * get<0>(dr) + get<1, 1>(inv_jac) * get<1>(dr)) *
      one_over_r;
  get<2>(dx_radius) = get<0, 2>(inv_jac) * get<0>(dr) * one_over_r;
  return dx_radius;
}

template <typename Frame>
aliases::SecondDeriv<Frame> D2xRadius<Frame>::function(
    const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
    const db::item_type<InvJacobian<Frame>>& inv_jac,
    const db::item_type<InvHessian<Frame>>& inv_hess) noexcept {
  auto d2x_radius = make_with_value<aliases::SecondDeriv<Frame>>(radius, 0.0);
  const DataVector one_over_r_squared = 1.0 / square(radius);
  const auto derivs =
      strahlkorper.ylm_spherepack().first_and_second_derivative(radius);

  for (size_t i = 0; i < 3; ++i) {
    // Diagonal terms.  Divide by square(r) later.
    for (size_t k = 0; k < 2; ++k) {  // Angular derivs are 2-dimensional
      d2x_radius.get(i, i) += derivs.first.get(k) * inv_hess.get(k, i, i);
      for (size_t l = 0; l < 2; ++l) {  // Angular derivs are 2-dimensional
        d2x_radius.get(i, i) +=
            derivs.second.get(l, k) * inv_jac.get(k, i) * inv_jac.get(l, i);
      }
    }
    d2x_radius.get(i, i) *= one_over_r_squared;
    // off_diagonal terms.  Symmetrize over i and j.
    // Divide by 2*square(r) later.
    for (size_t j = i + 1; j < 3; ++j) {
      for (size_t k = 0; k < 2; ++k) {  // Angular derivs are 2-dimensional
        d2x_radius.get(i, j) += derivs.first.get(k) *
                                (inv_hess.get(k, i, j) + inv_hess.get(k, j, i));
        for (size_t l = 0; l < 2; ++l) {  // Angular derivs are 2-dimensional
          d2x_radius.get(i, j) +=
              derivs.second.get(l, k) * (inv_jac.get(k, i) * inv_jac.get(l, j) +
                                         inv_jac.get(k, j) * inv_jac.get(l, i));
        }
      }
      d2x_radius.get(i, j) *= 0.5 * one_over_r_squared;
    }
  }
  return d2x_radius;
}

template <typename Frame>
DataVector LaplacianRadius<Frame>::function(
    const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
    const db::item_type<ThetaPhi<Frame>>& theta_phi) noexcept {
  const auto derivs =
      strahlkorper.ylm_spherepack().first_and_second_derivative(radius);
  return get<0, 0>(derivs.second) + get<1, 1>(derivs.second) +
         get<0>(derivs.first) / tan(get<0>(theta_phi));
}

template <typename Frame>
aliases::OneForm<Frame> NormalOneForm<Frame>::function(
    const db::item_type<DxRadius<Frame>>& dx_radius,
    const db::item_type<Rhat<Frame>>& r_hat) noexcept {
  auto one_form = make_with_value<aliases::OneForm<Frame>>(r_hat, 0.0);
  for (size_t d = 0; d < 3; ++d) {
    one_form.get(d) = r_hat.get(d) - dx_radius.get(d);
  }
  return one_form;
}

template <typename Frame>
aliases::Jacobian<Frame> Tangents<Frame>::function(
    const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
    const db::item_type<Rhat<Frame>>& r_hat,
    const db::item_type<Jacobian<Frame>>& jac) noexcept {
  const auto dr = strahlkorper.ylm_spherepack().gradient(radius);
  auto tangents = make_with_value<aliases::Jacobian<Frame>>(radius, 0.0);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      tangents.get(j, i) = dr.get(i) * r_hat.get(j) + radius * jac.get(j, i);
    }
  }
  return tangents;
}

}  // namespace StrahlkorperTags

namespace StrahlkorperTags {
template struct ThetaPhi<Frame::Inertial>;
template struct Rhat<Frame::Inertial>;
template struct Jacobian<Frame::Inertial>;
template struct InvJacobian<Frame::Inertial>;
template struct InvHessian<Frame::Inertial>;
template struct CartesianCoords<Frame::Inertial>;
template struct DxRadius<Frame::Inertial>;
template struct D2xRadius<Frame::Inertial>;
template struct LaplacianRadius<Frame::Inertial>;
template struct NormalOneForm<Frame::Inertial>;
template struct Tangents<Frame::Inertial>;
}  // namespace StrahlkorperTags

namespace StrahlkorperGr {
namespace Tags {
template struct AreaElement<Frame::Inertial>;
}  // namespace Tags
}  // namespace StrahlkorperGr
