// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/BulgedCube.hpp"

#include <boost/none.hpp>
#include <boost/optional.hpp>
#include <cmath>
#include <exception>
#include <functional>  // for std::reference_wrapper
#include <limits>
#include <pup.h>

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
class RootFunction {
 public:
  RootFunction(const double radius, const double sphericity,
               const double physical_r_squared, const double x_sq,
               const double y_sq, const double z_sq) noexcept
      : radius_(radius),
        sphericity_(sphericity),
        physical_r_squared_(physical_r_squared),
        x_sq_(x_sq),
        y_sq_(y_sq),
        z_sq_(z_sq) {}

  double operator()(const double rho) const noexcept {
    if (LIKELY(physical_r_squared_ != 0.0)) {
      const double x_sq_over_r_sq = x_sq_ / physical_r_squared_;
      const double y_sq_over_r_sq = y_sq_ / physical_r_squared_;
      const double z_sq_over_r_sq = z_sq_ / physical_r_squared_;
      return sqrt(physical_r_squared_) -
             radius_ * rho *
                 (1.0 / sqrt(3.0) +
                  sphericity_ *
                      (1.0 / sqrt(1.0 +
                                  square(rho) *
                                      (x_sq_over_r_sq + y_sq_over_r_sq)) +
                       1.0 / sqrt(1.0 +
                                  square(rho) *
                                      (x_sq_over_r_sq + z_sq_over_r_sq)) +
                       1.0 / sqrt(1.0 +
                                  square(rho) *
                                      (y_sq_over_r_sq + z_sq_over_r_sq)) -
                       1.0 / sqrt(2.0 + square(rho) * x_sq_over_r_sq) -
                       1.0 / sqrt(2.0 + square(rho) * y_sq_over_r_sq) -
                       1.0 / sqrt(2.0 + square(rho) * z_sq_over_r_sq)));
    } else {
      return 0.0;
    }
  }
  const double& get_r_sq() noexcept { return physical_r_squared_; }

 private:
  const double radius_;
  const double sphericity_;
  const double physical_r_squared_;
  const double x_sq_;
  const double y_sq_;
  const double z_sq_;
};

boost::optional<double> scaling_factor(RootFunction&& rootfunction) noexcept {
  const double& physical_r_squared = rootfunction.get_r_sq();
  try {
    const double tol = 10.0 * std::numeric_limits<double>::epsilon();
    double rho =
        // NOLINTNEXTLINE(clang-analyzer-core)
        RootFinder::toms748(rootfunction, 0.0, sqrt(3.0) + tol, tol, tol);
    if (LIKELY(physical_r_squared != 0.0)) {
      rho /= sqrt(physical_r_squared);
    }
    // There is no 'else' covering the case physical_r_squared==0.
    // This is because for physical_r_squared==0, we know that we
    // are at the origin x=y=z=0, and we know analytically that this
    // map maps the origin to itself.  In the inverse map function,
    // rho appears only in the combination rho*(x,y,z), so we don't
    // care what we return for rho because x,y,z are zero.  So we
    // don't touch rho here for the case physical_r_squared==0.
    return rho;
  } catch (std::exception& exception) {
    return boost::none;
  }
}
}  // namespace

namespace domain {
namespace CoordinateMaps {
BulgedCube::BulgedCube(const double radius, const double sphericity,
                       const bool use_equiangular_map) noexcept
    : radius_(radius),
      sphericity_(sphericity),
      use_equiangular_map_(use_equiangular_map),
      is_identity_(radius_ == sqrt(3.0) and sphericity_ == 0.0 and
                   not use_equiangular_map_) {
  ASSERT(radius > 0.0, "The radius of the cube must be greater than zero");
  ASSERT(sphericity >= 0.0 and sphericity < 1.0,
         "The sphericity must be strictly less than one.");
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> BulgedCube::operator()(
    const std::array<T, 3>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const auto physical_coordinates = [this](
      const ReturnType& cap_xi, const ReturnType& cap_eta,
      const ReturnType& cap_zeta) noexcept {
    const auto one_over_rho_xi = 1.0 / sqrt(2.0 + square(cap_xi));
    const auto one_over_rho_eta = 1.0 / sqrt(2.0 + square(cap_eta));
    const auto one_over_rho_zeta = 1.0 / sqrt(2.0 + square(cap_zeta));
    const auto one_over_rho_xi_eta =
        1.0 / sqrt(1.0 + square(cap_xi) + square(cap_eta));
    const auto one_over_rho_xi_zeta =
        1.0 / sqrt(1.0 + square(cap_xi) + square(cap_zeta));
    const auto one_over_rho_eta_zeta =
        1.0 / sqrt(1.0 + square(cap_eta) + square(cap_zeta));
    const ReturnType radial_scaling_factor =
        radius_ * (1.0 / sqrt(3.0) +
                   sphericity_ * (one_over_rho_eta_zeta + one_over_rho_xi_zeta +
                                  one_over_rho_xi_eta - one_over_rho_xi -
                                  one_over_rho_eta - one_over_rho_zeta));

    ReturnType physical_x = radial_scaling_factor * cap_xi;
    ReturnType physical_y = radial_scaling_factor * cap_eta;
    ReturnType physical_z = radial_scaling_factor * cap_zeta;
    return std::array<ReturnType, 3>{
        {std::move(physical_x), std::move(physical_y), std::move(physical_z)}};
  };

  if (use_equiangular_map_) {
    return physical_coordinates(
        tan(M_PI_4 * dereference_wrapper(source_coords[0])),
        tan(M_PI_4 * dereference_wrapper(source_coords[1])),
        tan(M_PI_4 * dereference_wrapper(source_coords[2])));
  }
  return physical_coordinates(dereference_wrapper(source_coords[0]),
                              dereference_wrapper(source_coords[1]),
                              dereference_wrapper(source_coords[2]));
}

boost::optional<std::array<double, 3>> BulgedCube::inverse(
    const std::array<double, 3>& target_coords) const noexcept {
  const double& physical_x = target_coords[0];
  const double& physical_y = target_coords[1];
  const double& physical_z = target_coords[2];
  const double x_sq = square(physical_x);
  const double y_sq = square(physical_y);
  const double z_sq = square(physical_z);
  const double physical_r_squared = x_sq + y_sq + z_sq;
  const auto scaling_factor =
      // NOLINTNEXTLINE(clang-analyzer-core)
      ::scaling_factor(
      RootFunction{radius_, sphericity_, physical_r_squared, x_sq, y_sq, z_sq});
  if (not scaling_factor) {
    return boost::none;
  }
  if (use_equiangular_map_) {
    return {{{2.0 * M_2_PI * atan(physical_x * scaling_factor.get()),
              2.0 * M_2_PI * atan(physical_y * scaling_factor.get()),
              2.0 * M_2_PI * atan(physical_z * scaling_factor.get())}}};
  }
  return {
      {{physical_x * scaling_factor.get(), physical_y * scaling_factor.get(),
        physical_z * scaling_factor.get()}}};
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> BulgedCube::xi_derivative(
    const std::array<T, 3>& source_coords) const noexcept {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const auto derivative_lambda = [this](
      const ReturnType& cap_xi, const ReturnType& cap_eta,
      const ReturnType& cap_zeta, const auto& cap_xi_deriv) {
    const ReturnType one_over_rho_xi_cubed =
        pow<3>(1.0 / sqrt(2.0 + square(cap_xi)));
    const ReturnType one_over_rho_eta = 1.0 / sqrt(2.0 + square(cap_eta));
    const ReturnType one_over_rho_zeta = 1.0 / sqrt(2.0 + square(cap_zeta));
    const ReturnType one_over_rho_xi_eta_cubed =
        pow<3>(1.0 / sqrt(1.0 + square(cap_xi) + square(cap_eta)));
    const ReturnType one_over_rho_xi_zeta_cubed =
        pow<3>(1.0 / sqrt(1.0 + square(cap_xi) + square(cap_zeta)));
    const ReturnType one_over_rho_eta_zeta =
        1.0 / sqrt(1.0 + square(cap_eta) + square(cap_zeta));
    const ReturnType common_factor =
        sphericity_ * radius_ * cap_xi * cap_xi_deriv *
        (one_over_rho_xi_cubed - one_over_rho_xi_eta_cubed -
         one_over_rho_xi_zeta_cubed);

    const ReturnType physical_x =
        radius_ * cap_xi_deriv *
        (1.0 / sqrt(3.0) +
         sphericity_ *
             (((1.0 + square(cap_eta)) * one_over_rho_xi_eta_cubed +
               (1.0 + square(cap_zeta)) * one_over_rho_xi_zeta_cubed -
               2.0 * one_over_rho_xi_cubed) +
              one_over_rho_eta_zeta - one_over_rho_eta - one_over_rho_zeta));
    const ReturnType physical_y = cap_eta * common_factor;
    const ReturnType physical_z = cap_zeta * common_factor;

    return std::array<ReturnType, 3>{{physical_x, physical_y, physical_z}};
  };
  if (use_equiangular_map_) {
    return derivative_lambda(
        tan(M_PI_4 * dereference_wrapper(source_coords[0])),
        tan(M_PI_4 * dereference_wrapper(source_coords[1])),
        tan(M_PI_4 * dereference_wrapper(source_coords[2])),
        ReturnType{M_PI_4 *
                   (1.0 + square(tan(M_PI_4 *
                                     dereference_wrapper(source_coords[0]))))});
  }
  return derivative_lambda(dereference_wrapper(source_coords[0]),
                           dereference_wrapper(source_coords[1]),
                           dereference_wrapper(source_coords[2]), 1.0);
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> BulgedCube::jacobian(
    const std::array<T, 3>& source_coords) const noexcept {
  const auto dX_dxi = xi_derivative(source_coords);
  const auto dX_deta = xi_derivative(
      std::array<std::reference_wrapper<const tt::remove_cvref_wrap_t<T>>, 3>{
          {std::cref(dereference_wrapper(source_coords[1])),
           std::cref(dereference_wrapper(source_coords[0])),
           std::cref(dereference_wrapper(source_coords[2]))}});
  const auto dX_dzeta = xi_derivative(
      std::array<std::reference_wrapper<const tt::remove_cvref_wrap_t<T>>, 3>{
          {std::cref(dereference_wrapper(source_coords[2])),
           std::cref(dereference_wrapper(source_coords[1])),
           std::cref(dereference_wrapper(source_coords[0]))}});
  auto jacobian_matrix =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);

  get<0, 0>(jacobian_matrix) = dX_dxi[0];
  get<0, 1>(jacobian_matrix) = dX_deta[1];
  get<0, 2>(jacobian_matrix) = dX_dzeta[2];
  get<1, 0>(jacobian_matrix) = dX_dxi[1];
  get<1, 1>(jacobian_matrix) = dX_deta[0];
  get<1, 2>(jacobian_matrix) = dX_dzeta[1];
  get<2, 0>(jacobian_matrix) = dX_dxi[2];
  get<2, 1>(jacobian_matrix) = dX_deta[2];
  get<2, 2>(jacobian_matrix) = dX_dzeta[0];
  return jacobian_matrix;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
BulgedCube::inv_jacobian(const std::array<T, 3>& source_coords) const noexcept {
  const auto jac = jacobian(source_coords);
  return determinant_and_inverse(jac).second;
}

void BulgedCube::pup(PUP::er& p) noexcept {
  p | radius_;
  p | sphericity_;
  p | use_equiangular_map_;
  p | is_identity_;
}

bool operator==(const BulgedCube& lhs, const BulgedCube& rhs) noexcept {
  return lhs.radius_ == rhs.radius_ and lhs.sphericity_ == rhs.sphericity_ and
         lhs.use_equiangular_map_ == rhs.use_equiangular_map_ and
         lhs.is_identity_ == rhs.is_identity_;
}

bool operator!=(const BulgedCube& lhs, const BulgedCube& rhs) noexcept {
  return not(lhs == rhs);
}

// Explicit instantiations
/// \cond
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3> BulgedCube::   \
  operator()(const std::array<DTYPE(data), 3>& source_coords) const noexcept; \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>  \
  BulgedCube::jacobian(const std::array<DTYPE(data), 3>& source_coords)       \
      const noexcept;                                                         \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>  \
  BulgedCube::inv_jacobian(const std::array<DTYPE(data), 3>& source_coords)   \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))

#undef DTYPE
#undef INSTANTIATE
/// \endcond
}  // namespace CoordinateMaps
}  // namespace domain
