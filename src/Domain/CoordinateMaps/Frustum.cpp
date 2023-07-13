// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Frustum.hpp"

#include <algorithm>
#include <cmath>
#include <pup.h>

#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Side.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/RootFinding/GslMultiRoot.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Exceptions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain::CoordinateMaps {

Frustum::Frustum(const std::array<std::array<double, 2>, 4>& face_vertices,
                 const double lower_bound, const double upper_bound,
                 OrientationMap<3> orientation_of_frustum,
                 const bool with_equiangular_map,
                 const Distribution zeta_distribution,
                 const std::optional<double> distribution_value,
                 const double sphericity, const double transition_phi,
                 const double opening_angle)
    // clang-tidy: trivially copyable
    : orientation_of_frustum_(std::move(orientation_of_frustum)),  // NOLINT
      with_equiangular_map_(with_equiangular_map),
      is_identity_(face_vertices ==
                       std::array<std::array<double, 2>, 4>{{{{-1.0, -1.0}},
                                                             {{1.0, 1.0}},
                                                             {{-1.0, -1.0}},
                                                             {{1.0, 1.0}}}} and
                   lower_bound == -1.0 and upper_bound == 1.0 and
                   orientation_of_frustum_ == OrientationMap<3>{} and
                   not with_equiangular_map and
                   zeta_distribution == Distribution::Linear),
      zeta_distribution_(zeta_distribution),
      sphericity_(sphericity) {
  ASSERT(sphericity_ >= 0.0 and sphericity_ <= 1.0,
         "The sphericity must be set between 0.0, corresponding to a flat "
         "surface, and 1.0, corresponding to a spherical surface, inclusive. "
         "It is currently set to "
             << sphericity << ".");
  const double& lower_x_lower_base = face_vertices[0][0];
  const double& lower_y_lower_base = face_vertices[0][1];
  const double& upper_x_lower_base = face_vertices[1][0];
  const double& upper_y_lower_base = face_vertices[1][1];
  const double& lower_x_upper_base = face_vertices[2][0];
  const double& lower_y_upper_base = face_vertices[2][1];
  const double& upper_x_upper_base = face_vertices[3][0];
  const double& upper_y_upper_base = face_vertices[3][1];
  ASSERT(upper_x_lower_base > lower_x_lower_base,
         "The lower bound for a coordinate must be numerically less"
         " than the upper bound for that coordinate.");
  ASSERT(upper_y_lower_base > lower_y_lower_base,
         "The lower bound for a coordinate must be numerically less"
         " than the upper bound for that coordinate.");
  ASSERT(upper_x_upper_base > lower_x_upper_base,
         "The lower bound for a coordinate must be numerically less"
         " than the upper bound for that coordinate.");
  ASSERT(upper_y_upper_base > lower_y_upper_base,
         "The lower bound for a coordinate must be numerically less"
         " than the upper bound for that coordinate.");
  ASSERT(upper_bound > lower_bound,
         "The lower bound for a coordinate must be numerically less"
         " than the upper bound for that coordinate.");
  ASSERT(get(determinant(discrete_rotation_jacobian(orientation_of_frustum_))) >
             0.0,
         "Frustum rotations must be done in such a manner that the sign of "
         "the determinant of the discrete rotation is positive. This is to "
         "preserve handedness of the coordinates.");
  sigma_x_ = 0.25 * (lower_x_upper_base + upper_x_upper_base +
                     lower_x_lower_base + upper_x_lower_base);
  delta_x_zeta_ = 0.25 * (lower_x_upper_base + upper_x_upper_base -
                          lower_x_lower_base - upper_x_lower_base);
  delta_x_xi_ = 0.25 * (upper_x_upper_base - lower_x_upper_base +
                        upper_x_lower_base - lower_x_lower_base);
  delta_x_xi_zeta_ = 0.25 * (upper_x_upper_base - lower_x_upper_base -
                             upper_x_lower_base + lower_x_lower_base);
  sigma_y_ = 0.25 * (lower_y_upper_base + upper_y_upper_base +
                     lower_y_lower_base + upper_y_lower_base);
  delta_y_zeta_ = 0.25 * (lower_y_upper_base + upper_y_upper_base -
                          lower_y_lower_base - upper_y_lower_base);
  delta_y_eta_ = 0.25 * (upper_y_upper_base - lower_y_upper_base +
                         upper_y_lower_base - lower_y_lower_base);
  delta_y_eta_zeta_ = 0.25 * (upper_y_upper_base - lower_y_upper_base -
                              upper_y_lower_base + lower_y_lower_base);
  sigma_z_ = 0.5 * (upper_bound + lower_bound);
  delta_z_zeta_ = 0.5 * (upper_bound - lower_bound);
  phi_ = transition_phi;
  half_opening_angle_ = 0.5 * opening_angle;
  one_over_tan_half_opening_angle_ = 1.0 / tan(half_opening_angle_);
  // The radius is taken to be the distance from the origin to the vertex of
  // the Frustum that is furthest away from the origin. For the rectangular
  // BinaryCompactObject Domain, this vertex is assumed to lie on a rectangular
  // prism centered at the origin made up of ten Frustums. The radius is then
  // that of a sphere that circumscribes the prism. For information on how
  // other Domains use the Frustum map, please see that particular Domain's
  // documentation.
  radius_ = sqrt(
      square(std::max({abs(upper_x_upper_base), abs(upper_x_lower_base),
                       abs(lower_x_upper_base), abs(lower_x_lower_base)})) +
      square(std::max({abs(upper_y_upper_base), abs(upper_y_lower_base),
                       abs(lower_y_upper_base), abs(lower_y_lower_base)})) +
      square(std::max({abs(upper_bound), abs(lower_bound)})));
  inner_radius_ = sqrt(
      square(std::max({abs(upper_x_lower_base), abs(lower_x_lower_base)})) +
      square(std::max({abs(upper_y_lower_base), abs(lower_y_lower_base)})) +
      square(lower_bound));

  if (zeta_distribution_ == Distribution::Projective) {
    double w_delta = distribution_value.has_value()
                         ? distribution_value.value()
                         : sqrt(((upper_x_lower_base - lower_x_lower_base) *
                                 (upper_y_lower_base - lower_y_lower_base)) /
                                ((upper_x_upper_base - lower_x_upper_base) *
                                 (upper_y_upper_base - lower_y_upper_base)));
    w_plus_ = w_delta + 1.0;
    w_minus_ = w_delta - 1.0;
  } else {
    w_plus_ = 2.0;
    w_minus_ = 0.0;
  }
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> Frustum::operator()(
    const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;

  const ReturnType& xi = source_coords[0];
  const ReturnType& eta = source_coords[1];
  const ReturnType& zeta = source_coords[2];
  ReturnType cap_zeta;
  if (zeta_distribution_ == Distribution::Projective) {
    cap_zeta = (w_minus_ + w_plus_ * zeta) / (w_plus_ + w_minus_ * zeta);
  } else if (zeta_distribution_ == Distribution::Linear) {
    cap_zeta = zeta;
  } else if (zeta_distribution_ == Distribution::Logarithmic) {
    Interval log_mapping{
        -1.0,
        1.0,
        -1.0,
        1.0,
        zeta_distribution_,
        -(radius_ + inner_radius_) / (radius_ - inner_radius_)};
    cap_zeta = log_mapping(std::array<T, 1>{{zeta}})[0];
  } else {
    ERROR(
        "Only the distributions Linear, Projective, and Logarithmic are "
        "supported.");
  }
  const ReturnType cap_xi_zero = with_equiangular_map_ ? tan(M_PI_4 * xi) : xi;
  const double one_plus_phi_square = 1.0 + phi_ * phi_;
  const ReturnType cap_xi_upper =
      with_equiangular_map_
          ? one_plus_phi_square * one_over_tan_half_opening_angle_ *
                    tan(half_opening_angle_ * (xi + phi_) /
                        one_plus_phi_square) -
                phi_
          : xi;
  const ReturnType cap_xi_transition = 0.5 * (1.0 + cap_zeta) * cap_xi_upper +
                                       0.5 * (1.0 - cap_zeta) * cap_xi_zero;
  const ReturnType cap_eta = with_equiangular_map_ ? tan(M_PI_4 * eta) : eta;

  ReturnType physical_x =
      sigma_x_ + delta_x_xi_ * cap_xi_transition +
      (delta_x_zeta_ + delta_x_xi_zeta_ * cap_xi_transition) * cap_zeta;
  ReturnType physical_y =
      sigma_y_ + delta_y_eta_ * cap_eta +
      (delta_y_zeta_ + delta_y_eta_zeta_ * cap_eta) * cap_zeta;
  ReturnType physical_z = sigma_z_ + delta_z_zeta_ * cap_zeta;
  if (sphericity_ > 0.0) {
    const ReturnType upper_surface_x =
        sigma_x_ + delta_x_xi_ * cap_xi_upper +
        (delta_x_zeta_ + delta_x_xi_zeta_ * cap_xi_upper);
    const ReturnType upper_surface_y =
        sigma_y_ + delta_y_eta_ * cap_eta +
        (delta_y_zeta_ + delta_y_eta_zeta_ * cap_eta);
    const double upper_surface_z = sigma_z_ + delta_z_zeta_;
    const ReturnType upper_surface_r =
        sqrt(square(upper_surface_x) + square(upper_surface_y) +
             (square(upper_surface_z)));
    const ReturnType correction_coefficient = 0.5 * sphericity_ *
                                              (1.0 + cap_zeta) *
                                              (radius_ / upper_surface_r - 1.0);
    physical_x += correction_coefficient * upper_surface_x;
    physical_y += correction_coefficient * upper_surface_y;
    physical_z += correction_coefficient * upper_surface_z;
  }
  std::array<ReturnType, 3> physical_coords{
      {std::move(physical_x), std::move(physical_y), std::move(physical_z)}};
  return discrete_rotation(orientation_of_frustum_, std::move(physical_coords));
}

std::optional<std::array<double, 3>> Frustum::inverse(
    const std::array<double, 3>& target_coords) const {
  // physical coords {x,y,z}
  std::array<double, 3> physical_coords =
      discrete_rotation(orientation_of_frustum_.inverse_map(), target_coords);

  // logical coords {xi,eta,zeta}
  std::array<double, 3> logical_coords{};

  logical_coords[2] = (physical_coords[2] - sigma_z_) / delta_z_zeta_;
  const auto denom0 = delta_x_xi_ + delta_x_xi_zeta_ * logical_coords[2];
  const auto denom1 = delta_y_eta_ + delta_y_eta_zeta_ * logical_coords[2];
  // denom0 and denom1 are always positive inside the frustum.
  if (denom0 < 0.0 or equal_within_roundoff(denom0, 0.0) or denom1 < 0.0 or
      equal_within_roundoff(denom1, 0.0)) {
    return std::nullopt;
  }

  logical_coords[0] =
      (physical_coords[0] - sigma_x_ - delta_x_zeta_ * logical_coords[2]) /
      denom0;
  logical_coords[1] =
      (physical_coords[1] - sigma_y_ - delta_y_zeta_ * logical_coords[2]) /
      denom1;
  if (with_equiangular_map_) {
    logical_coords[0] = atan(logical_coords[0]) / M_PI_4;
    logical_coords[1] = atan(logical_coords[1]) / M_PI_4;
  }
  if (zeta_distribution_ == Distribution::Projective) {
    logical_coords[2] = (-w_minus_ + w_plus_ * logical_coords[2]) /
                        (w_plus_ - w_minus_ * logical_coords[2]);
  } else if (zeta_distribution_ == Distribution::Logarithmic) {
    Interval log_mapping{
        -1.0,
        1.0,
        -1.0,
        1.0,
        zeta_distribution_,
        -(radius_ + inner_radius_) / (radius_ - inner_radius_)};
    logical_coords[2] =
        log_mapping.inverse(std::array<double, 1>{{logical_coords[2]}})
            .value()[0];
  } else {
    ASSERT(zeta_distribution_ == Distribution::Linear,
           "Only the "
           "distributions Linear, Projective, and Logarithmic are supported.");
  }
  if (sphericity_ > 0.0 or phi_ != 0.0) {
    // The physical_coords sometimes have magnitudes slightly
    // larger than radius_ due to roundoff error, this 1.0e-4 margin
    // allows these points to still be inverted. Points much further outside
    // of this value are likely to not be invertible, so we return
    // std::nullopt instead. Also, points below the lower bound are likely not
    // invertible, so return std::nullopt in that case as well.
    if (magnitude(physical_coords) > radius_ + 1.0e-4 or
        physical_coords[2] < (sigma_z_ - delta_z_zeta_) - 1.0e-4) {
      return std::nullopt;
    }
    const double absolute_tolerance = 1.0e-12;
    const double max_absolute_tolerance = 1.0e-10;
    const int maximum_iterations = 20;
    const Verbosity verbosity = Verbosity::Silent;
    const auto method = RootFinder::Method::Newton;
    const double relative_tolerance = 1.0e-12;
    const auto condition = RootFinder::StoppingConditions::Convergence(
        absolute_tolerance, relative_tolerance);
    struct {
      std::array<double, 3> operator()(
          const std::array<double, 3>& source_coords) const {
        // Terminate the rootfind when it diverges too far away from the logical
        // coordinate bounds of [-1, 1]. In this case the target coordinates are
        // likely outside of the bulged frustum. It would be better if we found
        // a way to handle this case more cleanly before the rootfind diverges.
        // Either way, logical coordinates too far outside of [-1, 1] lead to a
        // singular Jacobian, so we have to terminate here anyway.
        if (abs(source_coords[0]) > 3. or abs(source_coords[1]) > 3. or
            abs(source_coords[2]) > 3.) {
          throw convergence_error{
              "Logical coordinates are too far outside of [-1., 1], so the "
              "rootfind is likely diverging."};
        }
        return map(source_coords) - target_coords;
      }
      std::array<std::array<double, 3>, 3> jacobian(
          const std::array<double, 3>& source_coords) const {
        std::array<std::array<double, 3>, 3> jacobian_matrix_array{};
        const auto jacobian_matrix_tnsr = map.jacobian(source_coords);
        for (size_t i = 0; i < 3; i++) {
          for (size_t j = 0; j < 3; j++) {
            gsl::at(gsl::at(jacobian_matrix_array, i), j) =
                jacobian_matrix_tnsr.get(i, j);
          }
        }
        return jacobian_matrix_array;
      }
      const Frustum& map;
      const std::array<double, 3>& target_coords;
    } const rootfunction{*this, target_coords};

    // The `logical_coords` computed above are computed using the inverse map
    // to the flat frustum map, which is known analytically.
    // These `logical_coords` are then used as the initial guess for the
    // inverse of the bulged frustum map. This reduces the number of iterations
    // needed by the root finder. The initial guess is constructed to lie within
    // the logical cube.
    for (auto& coord : logical_coords) {
      if (abs(coord) > 1.0) {
        coord /= abs(coord);
      }
    }
    try {
      logical_coords = RootFinder::gsl_multiroot(
          rootfunction, logical_coords, condition, maximum_iterations,
          verbosity, max_absolute_tolerance, method);

    } catch (const convergence_error& e) {
      return std::nullopt;
    }
  }
  return logical_coords;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> Frustum::jacobian(
    const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType& xi = source_coords[0];
  const ReturnType& eta = source_coords[1];
  const ReturnType& zeta = source_coords[2];
  ReturnType cap_zeta;
  ReturnType cap_zeta_deriv;
  if (zeta_distribution_ == Distribution::Projective) {
    cap_zeta = (w_minus_ + w_plus_ * zeta) / (w_plus_ + w_minus_ * zeta);
    cap_zeta_deriv = (square(w_plus_) - square(w_minus_)) /
                     square(w_plus_ + zeta * w_minus_);
  } else if (zeta_distribution_ == Distribution::Linear) {
    cap_zeta = zeta;
    cap_zeta_deriv = make_with_value<ReturnType>(zeta, 1.0);
  } else if (zeta_distribution_ == Distribution::Logarithmic) {
    Interval log_mapping{
        -1.0,
        1.0,
        -1.0,
        1.0,
        zeta_distribution_,
        -(radius_ + inner_radius_) / (radius_ - inner_radius_)};
    cap_zeta = log_mapping(std::array<T, 1>{{zeta}})[0];
    cap_zeta_deriv = get<0, 0>(log_mapping.jacobian(std::array<T, 1>{{zeta}}));
  } else {
    ERROR(
        "Only the distributions Linear, Projective, and Logarithmic are "
        "supported.");
  }
  const ReturnType& cap_xi_zero = with_equiangular_map_ ? tan(M_PI_4 * xi) : xi;
  const double one_plus_phi_square = 1.0 + phi_ * phi_;
  const ReturnType cap_xi_upper =
      with_equiangular_map_
          ? one_plus_phi_square * one_over_tan_half_opening_angle_ *
                    tan(half_opening_angle_ * (xi + phi_) /
                        one_plus_phi_square) -
                phi_
          : xi;
  const ReturnType cap_xi_transition =
      with_equiangular_map_ ? 0.5 * (1.0 + cap_zeta) * cap_xi_upper +
                                  0.5 * (1.0 - cap_zeta) * cap_xi_zero
                            : xi;

  const ReturnType& cap_eta = with_equiangular_map_ ? tan(M_PI_4 * eta) : eta;
  const ReturnType cap_xi_zero_deriv =
      with_equiangular_map_ ? M_PI_4 * (1.0 + square(cap_xi_zero))
                            : make_with_value<ReturnType>(xi, 1.0);
  const ReturnType cap_xi_upper_deriv =
      with_equiangular_map_
          ? one_over_tan_half_opening_angle_ * half_opening_angle_ *
                (1.0 + square(tan(half_opening_angle_ * (xi + phi_) /
                                  one_plus_phi_square)))
          : make_with_value<ReturnType>(xi, 1.0);

  const ReturnType cap_xi_transition_deriv =
      with_equiangular_map_ ? 0.5 * (1.0 + cap_zeta) * cap_xi_upper_deriv +
                                  0.5 * (1.0 - cap_zeta) * cap_xi_zero_deriv
                            : make_with_value<ReturnType>(xi, 1.0);

  const ReturnType cap_eta_deriv = with_equiangular_map_
                                       ? M_PI_4 * (1.0 + square(cap_eta))
                                       : make_with_value<ReturnType>(eta, 1.0);

  auto jacobian_matrix =
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0);

  // dX_dxi
  const auto mapped_xi =
      orientation_of_frustum_.inverse_map()(Direction<3>::upper_xi());
  const size_t mapped_dim_0 = orientation_of_frustum_.inverse_map()(0);
  jacobian_matrix.get(mapped_dim_0, 0) =
      delta_x_xi_ + delta_x_xi_zeta_ * cap_zeta;
  if (with_equiangular_map_) {
    jacobian_matrix.get(mapped_dim_0, 0) *= cap_xi_transition_deriv;
  }
  if (mapped_xi.side() == Side::Lower) {
    jacobian_matrix.get(mapped_dim_0, 0) *= -1.0;
  }

  // dX_deta
  const auto mapped_eta =
      orientation_of_frustum_.inverse_map()(Direction<3>::upper_eta());
  const size_t mapped_dim_1 = orientation_of_frustum_.inverse_map()(1);
  jacobian_matrix.get(mapped_dim_1, 1) =
      delta_y_eta_ + delta_y_eta_zeta_ * cap_zeta;
  if (with_equiangular_map_) {
    jacobian_matrix.get(mapped_dim_1, 1) *= cap_eta_deriv;
  }
  if (mapped_eta.side() == Side::Lower) {
    jacobian_matrix.get(mapped_dim_1, 1) *= -1.0;
  }

  // dX_dzeta
  std::array<ReturnType, 3> dX_dzeta = discrete_rotation(
      orientation_of_frustum_,
      std::array<ReturnType, 3>{
          {delta_x_zeta_ + delta_x_xi_zeta_ * cap_xi_transition +
               (delta_x_xi_ + delta_x_xi_zeta_ * cap_zeta) * 0.5 *
                   (cap_xi_upper - cap_xi_zero),
           delta_y_zeta_ + delta_y_eta_zeta_ * cap_eta,
           make_with_value<ReturnType>(zeta, delta_z_zeta_)}});

  if (zeta_distribution_ != Distribution::Linear) {
    dX_dzeta[0] *= cap_zeta_deriv;
    dX_dzeta[1] *= cap_zeta_deriv;
    dX_dzeta[2] *= cap_zeta_deriv;
  }

  get<0, 2>(jacobian_matrix) = dX_dzeta[0];
  get<1, 2>(jacobian_matrix) = dX_dzeta[1];
  get<2, 2>(jacobian_matrix) = dX_dzeta[2];

  if (sphericity_ > 0.0) {
    const ReturnType flat_frustum_x = sigma_x_ + delta_x_xi_ * cap_xi_upper +
                                      delta_x_zeta_ +
                                      delta_x_xi_zeta_ * cap_xi_upper;

    const ReturnType flat_frustum_y = sigma_y_ + delta_y_eta_ * cap_eta +
                                      delta_y_zeta_ +
                                      delta_y_eta_zeta_ * cap_eta;

    const double flat_frustum_z = sigma_z_ + delta_z_zeta_;

    const ReturnType one_over_mag_flat =
        1.0 / sqrt(square(flat_frustum_x) + square(flat_frustum_y) +
                   square(flat_frustum_z));

    const ReturnType flat_frustum_x_hat = flat_frustum_x * one_over_mag_flat;

    const ReturnType flat_frustum_y_hat = flat_frustum_y * one_over_mag_flat;

    const ReturnType flat_frustum_z_hat = flat_frustum_z * one_over_mag_flat;

    const ReturnType r_over_mag_flat = radius_ * one_over_mag_flat;

    const double s_over_two = 0.5 * sphericity_;

    // delta_dX_dxi
    std::array<ReturnType, 3> delta_dX_dxi = discrete_rotation(
        orientation_of_frustum_,
        s_over_two * (1.0 + cap_zeta) * (delta_x_xi_ + delta_x_xi_zeta_) *
            std::array<ReturnType, 3>{
                {(r_over_mag_flat * (1.0 - square(flat_frustum_x_hat)) - 1.0),
                 -1.0 * r_over_mag_flat * flat_frustum_y_hat *
                     flat_frustum_x_hat,
                 -1.0 * r_over_mag_flat * flat_frustum_z_hat *
                     flat_frustum_x_hat}});

    if (with_equiangular_map_) {
      delta_dX_dxi[0] *= cap_xi_upper_deriv;
      delta_dX_dxi[1] *= cap_xi_upper_deriv;
      delta_dX_dxi[2] *= cap_xi_upper_deriv;
    }

    get<0, 0>(jacobian_matrix) += delta_dX_dxi[0];
    get<1, 0>(jacobian_matrix) += delta_dX_dxi[1];
    get<2, 0>(jacobian_matrix) += delta_dX_dxi[2];

    // delta_dX_deta
    std::array<ReturnType, 3> delta_dX_deta = discrete_rotation(
        orientation_of_frustum_,
        s_over_two * (1.0 + cap_zeta) * (delta_y_eta_ + delta_y_eta_zeta_) *
            std::array<ReturnType, 3>{
                {-1.0 * r_over_mag_flat * flat_frustum_x_hat *
                     flat_frustum_y_hat,
                 r_over_mag_flat * (1.0 - square(flat_frustum_y_hat)) - 1.0,
                 -1.0 * r_over_mag_flat * flat_frustum_z_hat *
                     flat_frustum_y_hat}});

    if (with_equiangular_map_) {
      delta_dX_deta[0] *= cap_eta_deriv;
      delta_dX_deta[1] *= cap_eta_deriv;
      delta_dX_deta[2] *= cap_eta_deriv;
    }

    get<0, 1>(jacobian_matrix) += delta_dX_deta[0];
    get<1, 1>(jacobian_matrix) += delta_dX_deta[1];
    get<2, 1>(jacobian_matrix) += delta_dX_deta[2];

    // delta_dX_dzeta
    std::array<ReturnType, 3> delta_dX_dzeta = discrete_rotation(
        orientation_of_frustum_,
        s_over_two * (r_over_mag_flat - 1.0) *
            std::array<ReturnType, 3>{
                {flat_frustum_x, flat_frustum_y,
                 make_with_value<ReturnType>(zeta, flat_frustum_z)}});

    if (zeta_distribution_ != Distribution::Linear) {
      delta_dX_dzeta[0] *= cap_zeta_deriv;
      delta_dX_dzeta[1] *= cap_zeta_deriv;
      delta_dX_dzeta[2] *= cap_zeta_deriv;
    }

    get<0, 2>(jacobian_matrix) += delta_dX_dzeta[0];
    get<1, 2>(jacobian_matrix) += delta_dX_dzeta[1];
    get<2, 2>(jacobian_matrix) += delta_dX_dzeta[2];
  }
  return jacobian_matrix;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> Frustum::inv_jacobian(
    const std::array<T, 3>& source_coords) const {
  const auto jac = jacobian(source_coords);
  return determinant_and_inverse(jac).second;
}

void Frustum::pup(PUP::er& p) {
  size_t version = 2;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version >= 0) {
    p | orientation_of_frustum_;
    p | with_equiangular_map_;
    p | is_identity_;
    if (version < 2 /*is unpacking*/) {
      bool with_projective_map = false;  // unused
      p | with_projective_map;
      zeta_distribution_ =
          with_projective_map ? Distribution::Projective : Distribution::Linear;
    }
    p | sigma_x_;
    p | delta_x_zeta_;
    p | delta_x_xi_;
    p | delta_x_xi_zeta_;
    p | sigma_y_;
    p | delta_y_zeta_;
    p | delta_y_eta_;
    p | delta_y_eta_zeta_;
    p | sigma_z_;
    p | delta_z_zeta_;
    p | w_plus_;
    p | w_minus_;
    p | sphericity_;
    p | radius_;
    p | phi_;
  }
  if (version >= 1) {
    p | half_opening_angle_;
    p | one_over_tan_half_opening_angle_;
  } else {
    half_opening_angle_ = M_PI_4;
    one_over_tan_half_opening_angle_ = 1.0;
  }
  if (version >= 2) {
    p | inner_radius_;
    p | zeta_distribution_;
  } else {
    // Only `Distribution::Logarithmic` uses `inner_radius`, which
    // was not implemented until version 2, so setting inner_radius = NaN
    // should not affect maps unpacked from old versions. `inner_radius` is
    // computed from frustum vertices so it is not necessary to compare
    // `inner_radius` values between maps in the equality operator. Note that
    // this means that a Frustum will have `inner_radius` == NaN when unpacking
    // data from an old version.
    inner_radius_ = std::numeric_limits<double>::signaling_NaN();
  }
}

bool operator==(const Frustum& lhs, const Frustum& rhs) {
  // Note that the inner radii are not compared, as `inner_radius` is computed
  // the Frustum vertices and as quantities derived from
  // vertices (e.g. `sigma_x`, `delta_x_xi`, etc.) are compared between the
  // Frustums and equality between these derived quantities implies equality
  // of their vertices. `inner_radius` might sometimes be NaN when unpacking
  // Frustums from old data. For more details see the comment in Frustum::pup
  // above.
  // `radius` is also not compared for similar reasons.
  return lhs.orientation_of_frustum_ == rhs.orientation_of_frustum_ and
         lhs.with_equiangular_map_ == rhs.with_equiangular_map_ and
         lhs.is_identity_ == rhs.is_identity_ and
         lhs.zeta_distribution_ == rhs.zeta_distribution_ and
         lhs.sigma_x_ == rhs.sigma_x_ and
         lhs.delta_x_zeta_ == rhs.delta_x_zeta_ and
         lhs.delta_x_xi_ == rhs.delta_x_xi_ and
         lhs.delta_x_xi_zeta_ == rhs.delta_x_xi_zeta_ and
         lhs.sigma_y_ == rhs.sigma_y_ and
         lhs.delta_y_zeta_ == rhs.delta_y_zeta_ and
         lhs.delta_y_eta_ == lhs.delta_y_eta_ and
         lhs.delta_y_eta_zeta_ == lhs.delta_y_eta_zeta_ and
         lhs.sigma_z_ == rhs.sigma_z_ and
         lhs.delta_z_zeta_ == rhs.delta_z_zeta_ and
         lhs.w_plus_ == rhs.w_plus_ and lhs.w_minus_ == rhs.w_minus_ and
         lhs.sphericity_ == rhs.sphericity_ and lhs.phi_ == rhs.phi_ and
         lhs.half_opening_angle_ == rhs.half_opening_angle_ and
         lhs.one_over_tan_half_opening_angle_ ==
             rhs.one_over_tan_half_opening_angle_;
}

bool operator!=(const Frustum& lhs, const Frustum& rhs) {
  return not(lhs == rhs);
}

// Explicit instantiations
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>                \
  Frustum::operator()(const std::array<DTYPE(data), 3>& source_coords) const; \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>  \
  Frustum::jacobian(const std::array<DTYPE(data), 3>& source_coords) const;   \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>  \
  Frustum::inv_jacobian(const std::array<DTYPE(data), 3>& source_coords)      \
      const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))
#undef DTYPE
#undef INSTANTIATE
}  // namespace domain::CoordinateMaps
