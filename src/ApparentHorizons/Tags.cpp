// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/Tags.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace StrahlkorperTags {

template <typename Frame>
void ThetaPhiCompute<Frame>::function(
    const gsl::not_null<aliases::ThetaPhi<Frame>*> theta_phi,
    const ::Strahlkorper<Frame>& strahlkorper) noexcept {
  auto temp = strahlkorper.ylm_spherepack().theta_phi_points();
  destructive_resize_components(theta_phi, temp[0].size());
  get<0>(*theta_phi) = temp[0];
  get<1>(*theta_phi) = temp[1];
}

template <typename Frame>
void RhatCompute<Frame>::function(
    const gsl::not_null<aliases::OneForm<Frame>*> r_hat,
    const aliases::ThetaPhi<Frame>& theta_phi) noexcept {
  const auto& theta = get<0>(theta_phi);
  const auto& phi = get<1>(theta_phi);

  destructive_resize_components(r_hat, theta.size());
  const DataVector sin_theta = sin(theta);
  get<0>(*r_hat) = sin_theta * cos(phi);
  get<1>(*r_hat) = sin_theta * sin(phi);
  get<2>(*r_hat) = cos(theta);
}

template <typename Frame>
void JacobianCompute<Frame>::function(
    const gsl::not_null<aliases::Jacobian<Frame>*> jac,
    const aliases::ThetaPhi<Frame>& theta_phi) noexcept {
  const auto& theta = get<0>(theta_phi);
  const auto& phi = get<1>(theta_phi);
  const DataVector sin_phi = sin(phi);
  const DataVector cos_phi = cos(phi);
  const DataVector cos_theta = cos(theta);

  destructive_resize_components(jac, cos_theta.size());
  get<0, 0>(*jac) = cos_theta * cos_phi;  // 1/R dx/dth
  get<1, 0>(*jac) = cos_theta * sin_phi;  // 1/R dy/dth
  get<2, 0>(*jac) = -sin(theta);          // 1/R dz/dth
  get<0, 1>(*jac) = -sin_phi;             // 1/(R sin(th)) dx/dph
  get<1, 1>(*jac) = cos_phi;              // 1/(R sin(th)) dy/dph
  get<2, 1>(*jac) = 0.0;                  // 1/(R sin(th)) dz/dph
}

template <typename Frame>
void InvJacobianCompute<Frame>::function(
    const gsl::not_null<aliases::InvJacobian<Frame>*> inv_jac,
    const aliases::ThetaPhi<Frame>& theta_phi) noexcept {
  const auto& theta = get<0>(theta_phi);
  const auto& phi = get<1>(theta_phi);
  const DataVector sin_phi = sin(phi);
  const DataVector cos_phi = cos(phi);
  const DataVector cos_theta = cos(theta);

  destructive_resize_components(inv_jac, cos_theta.size());
  get<0, 0>(*inv_jac) = cos_theta * cos_phi;  // R dth/dx
  get<0, 1>(*inv_jac) = cos_theta * sin_phi;  // R dth/dy
  get<0, 2>(*inv_jac) = -sin(theta);          // R dth/dz
  get<1, 0>(*inv_jac) = -sin_phi;             // R sin(th) dph/dx
  get<1, 1>(*inv_jac) = cos_phi;              // R sin(th) dph/dy
  get<1, 2>(*inv_jac) = 0.0;                  // R sin(th) dph/dz
}

template <typename Frame>
void InvHessianCompute<Frame>::function(
    const gsl::not_null<aliases::InvHessian<Frame>*> inv_hess,
    const aliases::ThetaPhi<Frame>& theta_phi) noexcept {
  const auto& theta = get<0>(theta_phi);
  const auto& phi = get<1>(theta_phi);
  const DataVector sin_phi = sin(phi);
  const DataVector cos_phi = cos(phi);
  const DataVector sin_theta = sin(theta);
  const DataVector cos_theta = cos(theta);

  const DataVector sin_sq_theta = square(sin_theta);
  const DataVector cos_sq_theta = square(cos_theta);
  const DataVector sin_theta_cos_theta = sin_theta * cos_theta;
  const DataVector sin_sq_phi = square(sin_phi);
  const DataVector cos_sq_phi = square(cos_phi);
  const DataVector sin_phi_cos_phi = sin_phi * cos_phi;
  const DataVector csc_theta = 1.0 / sin_theta;
  const DataVector f1 = 1.0 + 2.0 * sin_sq_theta;
  const DataVector cot_theta = cos_theta * csc_theta;

  destructive_resize_components(inv_hess, cos_theta.size());
  // R^2 d^2 th/(dx^2)
  get<0, 0, 0>(*inv_hess) = cot_theta * (1.0 - cos_sq_phi * f1);
  // R^2 d^2 th/(dxdy)
  get<0, 0, 1>(*inv_hess) = -cot_theta * sin_phi_cos_phi * f1;
  // R^2 d^2 th/(dxdz)
  get<0, 0, 2>(*inv_hess) = (sin_sq_theta - cos_sq_theta) * cos_phi;
  // R^2 d^2 th/(dydx)
  get<0, 1, 0>(*inv_hess) = get<0, 0, 1>(*inv_hess);
  // R^2 d^2 th/(dy^2)
  get<0, 1, 1>(*inv_hess) = cot_theta * (1.0 - sin_sq_phi * f1);
  // R^2 d^2 th/(dydz)
  get<0, 1, 2>(*inv_hess) = (sin_sq_theta - cos_sq_theta) * sin_phi;
  // R^2 d^2 th/(dzdx)
  get<0, 2, 0>(*inv_hess) = get<0, 0, 2>(*inv_hess);
  // R^2 d^2 th/(dzdy)
  get<0, 2, 1>(*inv_hess) = get<0, 1, 2>(*inv_hess);
  // R^2 d^2 th/(dz^2)
  get<0, 2, 2>(*inv_hess) = 2.0 * sin_theta_cos_theta;
  // R^2 d/dx (sin(th) dph/dx)
  get<1, 0, 0>(*inv_hess) = sin_phi_cos_phi * (1.0 + sin_sq_theta) * csc_theta;
  // R^2 d/dx (sin(th) dph/dy)
  get<1, 0, 1>(*inv_hess) =
      (sin_sq_phi - sin_sq_theta * cos_sq_phi) * csc_theta;
  // R^2 d/dx (sin(th) dph/dz)
  get<1, 0, 2>(*inv_hess) = 0.0;
  // R^2 d/dy (sin(th) dph/dx)
  get<1, 1, 0>(*inv_hess) =
      (sin_sq_theta * sin_sq_phi - cos_sq_phi) * csc_theta;
  // R^2 d/dy (sin(th) dph/dy)
  get<1, 1, 1>(*inv_hess) = -get<1, 0, 0>(*inv_hess);
  // R^2 d/dy (sin(th) dph/dz)
  get<1, 1, 2>(*inv_hess) = 0.0;
  // R^2 d/dz (sin(th) dph/dx)
  get<1, 2, 0>(*inv_hess) = cos_theta * sin_phi;
  // R^2 d/dz (sin(th) dph/dy)
  get<1, 2, 1>(*inv_hess) = -cos_theta * cos_phi;
  // R^2 d/dz (sin(th) dph/dz)
  get<1, 2, 2>(*inv_hess) = 0.0;
}

template <typename Frame>
void RadiusCompute<Frame>::function(
    const gsl::not_null<DataVector*> radius,
    const ::Strahlkorper<Frame>& strahlkorper) noexcept {
  radius->destructive_resize(strahlkorper.ylm_spherepack().physical_size());
  *radius =
      strahlkorper.ylm_spherepack().spec_to_phys(strahlkorper.coefficients());
}

template <typename Frame>
void CartesianCoordsCompute<Frame>::function(
    const gsl::not_null<aliases::Vector<Frame>*> coords,
    const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
    const aliases::OneForm<Frame>& r_hat) noexcept {
  destructive_resize_components(coords, radius.size());
  for (size_t d = 0; d < 3; ++d) {
    coords->get(d) = gsl::at(strahlkorper.center(), d) + r_hat.get(d) * radius;
  }
}

template <typename Frame>
void DxRadiusCompute<Frame>::function(
    const gsl::not_null<aliases::OneForm<Frame>*> dx_radius,
    const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
    const aliases::InvJacobian<Frame>& inv_jac) noexcept {
  destructive_resize_components(dx_radius, radius.size());
  const DataVector one_over_r = 1.0 / radius;
  const auto dr = strahlkorper.ylm_spherepack().gradient(radius);
  get<0>(*dx_radius) =
      (get<0, 0>(inv_jac) * get<0>(dr) + get<1, 0>(inv_jac) * get<1>(dr)) *
      one_over_r;
  get<1>(*dx_radius) =
      (get<0, 1>(inv_jac) * get<0>(dr) + get<1, 1>(inv_jac) * get<1>(dr)) *
      one_over_r;
  get<2>(*dx_radius) = get<0, 2>(inv_jac) * get<0>(dr) * one_over_r;
}

template <typename Frame>
void D2xRadiusCompute<Frame>::function(
    const gsl::not_null<aliases::SecondDeriv<Frame>*> d2x_radius,
    const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
    const aliases::InvJacobian<Frame>& inv_jac,
    const aliases::InvHessian<Frame>& inv_hess) noexcept {
  destructive_resize_components(d2x_radius, radius.size());
  for (auto& component : *d2x_radius) {
    component = 0.0;
  }
  const DataVector one_over_r_squared = 1.0 / square(radius);
  const auto derivs =
      strahlkorper.ylm_spherepack().first_and_second_derivative(radius);

  for (size_t i = 0; i < 3; ++i) {
    // Diagonal terms.  Divide by square(r) later.
    for (size_t k = 0; k < 2; ++k) {  // Angular derivs are 2-dimensional
      d2x_radius->get(i, i) += derivs.first.get(k) * inv_hess.get(k, i, i);
      for (size_t l = 0; l < 2; ++l) {  // Angular derivs are 2-dimensional
        d2x_radius->get(i, i) +=
            derivs.second.get(l, k) * inv_jac.get(k, i) * inv_jac.get(l, i);
      }
    }
    d2x_radius->get(i, i) *= one_over_r_squared;
    // off_diagonal terms.  Symmetrize over i and j.
    // Divide by 2*square(r) later.
    for (size_t j = i + 1; j < 3; ++j) {
      for (size_t k = 0; k < 2; ++k) {  // Angular derivs are 2-dimensional
        d2x_radius->get(i, j) += derivs.first.get(k) * (inv_hess.get(k, i, j) +
                                                        inv_hess.get(k, j, i));
        for (size_t l = 0; l < 2; ++l) {  // Angular derivs are 2-dimensional
          d2x_radius->get(i, j) +=
              derivs.second.get(l, k) * (inv_jac.get(k, i) * inv_jac.get(l, j) +
                                         inv_jac.get(k, j) * inv_jac.get(l, i));
        }
      }
      d2x_radius->get(i, j) *= 0.5 * one_over_r_squared;
    }
  }
}

template <typename Frame>
void LaplacianRadiusCompute<Frame>::function(
    const gsl::not_null<DataVector*> lap_radius,
    const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
    const aliases::ThetaPhi<Frame>& theta_phi) noexcept {
  lap_radius->destructive_resize(radius.size());
  const auto derivs =
      strahlkorper.ylm_spherepack().first_and_second_derivative(radius);
  *lap_radius = get<0, 0>(derivs.second) + get<1, 1>(derivs.second) +
                get<0>(derivs.first) / tan(get<0>(theta_phi));
}

template <typename Frame>
void NormalOneFormCompute<Frame>::function(
    const gsl::not_null<aliases::OneForm<Frame>*> one_form,
    const aliases::OneForm<Frame>& dx_radius,
    const aliases::OneForm<Frame>& r_hat) noexcept {
  destructive_resize_components(one_form, r_hat.begin()->size());
  for (size_t d = 0; d < 3; ++d) {
    one_form->get(d) = r_hat.get(d) - dx_radius.get(d);
  }
}

template <typename Frame>
void TangentsCompute<Frame>::function(
    const gsl::not_null<aliases::Jacobian<Frame>*> tangents,
    const ::Strahlkorper<Frame>& strahlkorper, const DataVector& radius,
    const aliases::OneForm<Frame>& r_hat,
    const aliases::Jacobian<Frame>& jac) noexcept {
  destructive_resize_components(tangents, radius.size());
  const auto dr = strahlkorper.ylm_spherepack().gradient(radius);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      tangents->get(j, i) = dr.get(i) * r_hat.get(j) + radius * jac.get(j, i);
    }
  }
}

}  // namespace StrahlkorperTags

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                             \
  template struct StrahlkorperTags::ThetaPhiCompute<FRAME(data)>;        \
  template struct StrahlkorperTags::RhatCompute<FRAME(data)>;            \
  template struct StrahlkorperTags::JacobianCompute<FRAME(data)>;        \
  template struct StrahlkorperTags::InvJacobianCompute<FRAME(data)>;     \
  template struct StrahlkorperTags::InvHessianCompute<FRAME(data)>;      \
  template struct StrahlkorperTags::RadiusCompute<FRAME(data)>;          \
  template struct StrahlkorperTags::CartesianCoordsCompute<FRAME(data)>; \
  template struct StrahlkorperTags::DxRadiusCompute<FRAME(data)>;        \
  template struct StrahlkorperTags::D2xRadiusCompute<FRAME(data)>;       \
  template struct StrahlkorperTags::LaplacianRadiusCompute<FRAME(data)>; \
  template struct StrahlkorperTags::NormalOneFormCompute<FRAME(data)>;   \
  template struct StrahlkorperTags::TangentsCompute<FRAME(data)>;
GENERATE_INSTANTIATIONS(INSTANTIATE, (Frame::Grid, Frame::Inertial))
#undef INSTANTIATE
#undef FRAME
