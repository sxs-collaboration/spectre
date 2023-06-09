// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/SphericalTorus.hpp"

#include <pup.h>

#include "Options/ParseError.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain::CoordinateMaps {

SphericalTorus::SphericalTorus(const double r_min, const double r_max,
                               const double min_polar_angle,
                               const double fraction_of_torus,
                               const Options::Context& context)
    : r_min_(r_min),
      r_max_(r_max),
      pi_over_2_minus_theta_min_(M_PI_2 - min_polar_angle),
      fraction_of_torus_(fraction_of_torus) {
  if (r_min_ <= 0.0) {
    PARSE_ERROR(context, "Minimum radius must be positive.");
  }
  if (r_max_ <= r_min_) {
    PARSE_ERROR(context, "Maximum radius must be greater than minimum radius.");
  }
  if (pi_over_2_minus_theta_min_ <= 0.0) {
    PARSE_ERROR(context, "Minimum polar angle should be less than pi/2.");
  }
  if (pi_over_2_minus_theta_min_ >= 0.5 * M_PI) {
    PARSE_ERROR(context, "Minimum polar angle should be positive.");
  }
  if (fraction_of_torus_ <= 0.0) {
    PARSE_ERROR(context, "Fraction of torus included must be positive.");
  }
  if (fraction_of_torus_ > 1.0) {
    PARSE_ERROR(context, "Fraction of torus included must be at most 1.");
  }
}

SphericalTorus::SphericalTorus(const std::array<double, 2>& radial_range,
                               const double min_polar_angle,
                               const double fraction_of_torus,
                               const Options::Context& context)
    : SphericalTorus(radial_range[0], radial_range[1], min_polar_angle,
                     fraction_of_torus, context) {}

//
// **** WARNING : below we are using the following convention for naming
//                spherical coordinate variables internally
//  * r     : radius
//  * theta : azimuthal angle
//  * phi   : elevation angle measured from equator, which is equal to (\pi/2 -
//            polar angle) where the polar angle is measured from +z axis.
//
template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> SphericalTorus::operator()(
    const std::array<T, 3>& source_coords) const {
  const auto r = radius(source_coords[0]);
  const auto theta = M_PI * fraction_of_torus_ * source_coords[2];
  const auto phi = pi_over_2_minus_theta_min_ * source_coords[1];

  return {{r * cos(theta) * cos(phi), r * sin(theta) * cos(phi), r * sin(phi)}};
}

std::optional<std::array<double, 3>> SphericalTorus::inverse(
    const std::array<double, 3>& target_coords) const {
  const double r =
      std::hypot(target_coords[0], target_coords[1], target_coords[2]);
  const double theta = std::atan2(target_coords[1], target_coords[0]);
  const double phi = std::atan2(target_coords[2],
                                std::hypot(target_coords[0], target_coords[1]));

  return {{radius_inverse(r), phi / pi_over_2_minus_theta_min_,
           theta / (M_PI * fraction_of_torus_)}};
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
SphericalTorus::jacobian(const std::array<T, 3>& source_coords) const {
  using UnwrappedT = tt::remove_cvref_wrap_t<T>;
  auto jacobian = make_with_value<tnsr::Ij<UnwrappedT, 3, Frame::NoFrame>>(
      dereference_wrapper(source_coords[0]), 0.0);

  // In order to reduce number of memory allocations we use some slots of
  // jacobian for storing temp variables as below.

  auto& sin_theta = get<1, 1>(jacobian);
  auto& cos_theta = get<1, 2>(jacobian);
  get<2, 2>(jacobian) = M_PI * fraction_of_torus_ * source_coords[2];
  cos_theta = cos(get<2, 2>(jacobian));
  sin_theta = sin(get<2, 2>(jacobian));

  auto& sin_phi = get<2, 0>(jacobian);
  auto& cos_phi = get<2, 1>(jacobian);
  get<2, 2>(jacobian) = pi_over_2_minus_theta_min_ * source_coords[1];
  cos_phi = cos(get<2, 2>(jacobian));
  sin_phi = sin(get<2, 2>(jacobian));

  auto& r = get<2, 2>(jacobian);
  radius(make_not_null(&r), source_coords[0]);

  // Note : execution order matters here since we are overwriting each temp
  // variables with jacobian values corresponding to the slot.
  get<0, 0>(jacobian) = 0.5 * (r_max_ - r_min_) * cos_theta * cos_phi;
  get<0, 1>(jacobian) = -pi_over_2_minus_theta_min_ * r * cos_theta * sin_phi;
  get<0, 2>(jacobian) = -M_PI * fraction_of_torus_ * r * sin_theta * cos_phi;
  get<1, 0>(jacobian) = 0.5 * (r_max_ - r_min_) * sin_theta * cos_phi;
  get<1, 1>(jacobian) = -pi_over_2_minus_theta_min_ * r * sin_theta * sin_phi;
  get<1, 2>(jacobian) = M_PI * fraction_of_torus_ * r * cos_theta * cos_phi;
  get<2, 0>(jacobian) = 0.5 * (r_max_ - r_min_) * sin_phi;
  get<2, 1>(jacobian) = pi_over_2_minus_theta_min_ * r * cos_phi;
  get<2, 2>(jacobian) = 0.0;

  return jacobian;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
SphericalTorus::inv_jacobian(const std::array<T, 3>& source_coords) const {
  using UnwrappedT = tt::remove_cvref_wrap_t<T>;
  auto inv_jacobian = make_with_value<tnsr::Ij<UnwrappedT, 3, Frame::NoFrame>>(
      dereference_wrapper(source_coords[0]), 0.0);

  // In order to reduce number of memory allocations we use some slots of
  // jacobian for storing temp variables as below.

  auto& sin_theta = get<1, 1>(inv_jacobian);
  auto& cos_theta = get<2, 1>(inv_jacobian);
  get<2, 2>(inv_jacobian) = M_PI * fraction_of_torus_ * source_coords[2];
  cos_theta = cos(get<2, 2>(inv_jacobian));
  sin_theta = sin(get<2, 2>(inv_jacobian));

  auto& sin_phi = get<0, 2>(inv_jacobian);
  auto& cos_phi = get<1, 2>(inv_jacobian);
  get<2, 2>(inv_jacobian) = pi_over_2_minus_theta_min_ * source_coords[1];
  cos_phi = cos(get<2, 2>(inv_jacobian));
  sin_phi = sin(get<2, 2>(inv_jacobian));

  auto& r = get<2, 2>(inv_jacobian);
  radius(make_not_null(&r), source_coords[0]);

  // Note : execution order matters here since we are overwriting each temp
  // variables with jacobian values corresponding to the slot.
  get<0, 0>(inv_jacobian) = 2.0 / (r_max_ - r_min_) * cos_theta * cos_phi;
  get<0, 1>(inv_jacobian) = 2.0 / (r_max_ - r_min_) * sin_theta * cos_phi;
  get<1, 0>(inv_jacobian) =
      -(1.0 / pi_over_2_minus_theta_min_) * cos_theta * sin_phi / r;
  get<2, 0>(inv_jacobian) =
      -(1.0 / (M_PI * fraction_of_torus_)) * sin_theta / (r * cos_phi);
  get<1, 1>(inv_jacobian) =
      -(1.0 / pi_over_2_minus_theta_min_) * sin_theta * sin_phi / r;
  get<2, 1>(inv_jacobian) =
      (1.0 / (M_PI * fraction_of_torus_)) * cos_theta / (r * cos_phi);
  get<0, 2>(inv_jacobian) = 2.0 / (r_max_ - r_min_) * sin_phi;
  get<1, 2>(inv_jacobian) = (1.0 / pi_over_2_minus_theta_min_) * cos_phi / r;
  get<2, 2>(inv_jacobian) = 0.0;

  return inv_jacobian;
}

template <typename T>
tnsr::Ijj<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
SphericalTorus::hessian(const std::array<T, 3>& source_coords) const {
  using UnwrappedT = tt::remove_cvref_wrap_t<T>;
  auto hessian = make_with_value<tnsr::Ijj<UnwrappedT, 3, Frame::NoFrame>>(
      dereference_wrapper(source_coords[0]), 0.0);

  // In order to reduce number of memory allocations we use some slots of
  // hessian for storing temp variables as below.

  // these two slots are zeros (not used)
  auto& theta_factor = get<2, 0, 0>(hessian);
  auto& phi_factor = get<2, 0, 2>(hessian);
  theta_factor = M_PI * fraction_of_torus_;
  phi_factor = pi_over_2_minus_theta_min_;

  // these two slots are zeros (not used)
  auto& cos_theta = get<2, 1, 2>(hessian);
  auto& sin_theta = get<2, 2, 2>(hessian);
  get<0, 0, 0>(hessian) = theta_factor * source_coords[2];
  cos_theta = cos(get<0, 0, 0>(hessian));
  sin_theta = sin(get<0, 0, 0>(hessian));

  // these two slots are NOT zeros, but will be overwritten with hessian later
  auto& cos_phi = get<2, 0, 1>(hessian);
  auto& sin_phi = get<2, 1, 1>(hessian);
  get<0, 0, 0>(hessian) = phi_factor * source_coords[1];
  cos_phi = cos(get<0, 0, 0>(hessian));
  sin_phi = sin(get<0, 0, 0>(hessian));

  // these two slots are zeros (not used)
  auto& r_factor = get<0, 0, 0>(hessian);
  auto& r = get<1, 0, 0>(hessian);
  r_factor = 0.5 * (r_max_ - r_min_);
  radius(make_not_null(&r), source_coords[0]);

  // Note : execution order matters here
  get<0, 0, 1>(hessian) = -r_factor * phi_factor * cos_theta * sin_phi;
  get<0, 0, 2>(hessian) = -r_factor * theta_factor * sin_theta * cos_phi;
  get<0, 1, 1>(hessian) = -square(phi_factor) * r * cos_theta * cos_phi;
  get<0, 1, 2>(hessian) = phi_factor * theta_factor * r * sin_theta * sin_phi;
  get<0, 2, 2>(hessian) = -square(theta_factor) * r * cos_theta * cos_phi;
  get<1, 0, 1>(hessian) = -r_factor * phi_factor * sin_theta * sin_phi;
  get<1, 0, 2>(hessian) = r_factor * theta_factor * cos_theta * cos_phi;
  get<1, 1, 1>(hessian) = -square(phi_factor) * r * sin_theta * cos_phi;
  get<1, 1, 2>(hessian) = -phi_factor * theta_factor * r * cos_theta * sin_phi;
  get<1, 2, 2>(hessian) = -square(theta_factor) * r * sin_theta * cos_phi;
  get<2, 0, 1>(hessian) = r_factor * phi_factor * cos_phi;
  get<2, 1, 1>(hessian) = -square(phi_factor) * r * sin_phi;

  // remove temp vars and restore to zero
  get<0, 0, 0>(hessian) = 0.0;
  get<1, 0, 0>(hessian) = 0.0;
  get<2, 0, 0>(hessian) = 0.0;
  get<2, 0, 2>(hessian) = 0.0;
  get<2, 1, 2>(hessian) = 0.0;
  get<2, 2, 2>(hessian) = 0.0;

  return hessian;
}

template <typename T>
tnsr::Ijk<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
SphericalTorus::derivative_of_inv_jacobian(
    const std::array<T, 3>& source_coords) const {
  using UnwrappedT = tt::remove_cvref_wrap_t<T>;
  auto result = make_with_value<tnsr::Ijk<UnwrappedT, 3, Frame::NoFrame>>(
      dereference_wrapper(source_coords[0]), 0.0);

  // In order to reduce number of memory allocations we use some slots of
  // `result` for storing temp variables as below.

  auto& cos_theta = get<1, 2, 2>(result);
  auto& sin_theta = get<2, 2, 0>(result);
  get<0, 0, 0>(result) = M_PI * fraction_of_torus_ * source_coords[2];
  cos_theta = cos(get<0, 0, 0>(result));
  sin_theta = sin(get<0, 0, 0>(result));

  auto& cos_phi = get<2, 2, 1>(result);
  auto& sin_phi = get<2, 2, 2>(result);
  get<0, 0, 0>(result) = pi_over_2_minus_theta_min_ * source_coords[1];
  cos_phi = cos(get<0, 0, 0>(result));
  sin_phi = sin(get<0, 0, 0>(result));

  auto& theta_factor = get<0, 2, 0>(result);
  auto& phi_factor = get<0, 2, 2>(result);
  theta_factor = M_PI * fraction_of_torus_;
  phi_factor = pi_over_2_minus_theta_min_;

  auto& r = get<0, 0, 0>(result);
  auto& r_factor = get<0, 1, 0>(result);
  radius(make_not_null(&r), source_coords[0]);
  r_factor = 0.5 * (r_max_ - r_min_);

  get<0, 0, 1>(result) = -phi_factor / r_factor * cos_theta * sin_phi;
  get<0, 0, 2>(result) = -theta_factor / r_factor * sin_theta * cos_phi;
  get<0, 1, 1>(result) = -phi_factor / r_factor * sin_theta * sin_phi;
  get<0, 1, 2>(result) = theta_factor / r_factor * cos_theta * cos_phi;
  get<0, 2, 1>(result) = phi_factor / r_factor * cos_phi;
  get<1, 0, 0>(result) =
      r_factor / phi_factor * cos_theta * sin_phi / square(r);
  get<1, 0, 1>(result) = -cos_theta * cos_phi / r;
  get<1, 0, 2>(result) = theta_factor / phi_factor * sin_theta * sin_phi / r;
  get<1, 1, 0>(result) =
      r_factor / phi_factor * sin_theta * sin_phi / square(r);
  get<1, 1, 1>(result) = -sin_theta * cos_phi / r;
  get<1, 1, 2>(result) = -theta_factor / phi_factor * cos_theta * sin_phi / r;
  get<1, 2, 0>(result) = -r_factor / phi_factor * cos_phi / square(r);
  get<1, 2, 1>(result) = -sin_phi / r;
  get<2, 0, 0>(result) =
      r_factor / theta_factor * sin_theta / (square(r) * cos_phi);
  get<2, 0, 1>(result) =
      -phi_factor / theta_factor * sin_theta * sin_phi / (r * square(cos_phi));
  get<2, 0, 2>(result) = -cos_theta / (r * cos_phi);

  get<2, 1, 0>(result) =
      -r_factor / theta_factor * cos_theta / (square(r) * cos_phi);
  get<2, 1, 1>(result) =
      phi_factor / theta_factor * cos_theta * sin_phi / (r * square(cos_phi));
  get<2, 1, 2>(result) = -sin_theta / (r * cos_phi);

  // remove temp vars and restore to zero
  get<0, 0, 0>(result) = 0.0;
  get<0, 1, 0>(result) = 0.0;
  get<0, 2, 0>(result) = 0.0;
  get<0, 2, 2>(result) = 0.0;
  get<1, 2, 2>(result) = 0.0;
  get<2, 2, 0>(result) = 0.0;
  get<2, 2, 1>(result) = 0.0;
  get<2, 2, 2>(result) = 0.0;

  return result;
}

void SphericalTorus::pup(PUP::er& p) {
  size_t version = 0;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version >= 0) {
    p | r_min_;
    p | r_max_;
    p | pi_over_2_minus_theta_min_;
    p | fraction_of_torus_;
  }
}

template <typename T>
tt::remove_cvref_wrap_t<T> SphericalTorus::radius(const T& x) const {
  return 0.5 * r_min_ * (1.0 - x) + 0.5 * r_max_ * (1.0 + x);
}

template <typename T>
void SphericalTorus::radius(const gsl::not_null<tt::remove_cvref_wrap_t<T>*> r,
                            const T& x) const {
  *r = 0.5 * r_min_ * (1.0 - x) + 0.5 * r_max_ * (1.0 + x);
}

template <typename T>
tt::remove_cvref_wrap_t<T> SphericalTorus::radius_inverse(const T& r) const {
  return ((r - r_min_) - (r_max_ - r)) / (r_max_ - r_min_);
}

bool operator==(const SphericalTorus& lhs, const SphericalTorus& rhs) {
  return lhs.r_min_ == rhs.r_min_ and lhs.r_max_ == rhs.r_max_ and
         lhs.pi_over_2_minus_theta_min_ == rhs.pi_over_2_minus_theta_min_ and
         lhs.fraction_of_torus_ == rhs.fraction_of_torus_;
}

bool operator!=(const SphericalTorus& lhs, const SphericalTorus& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>                \
  SphericalTorus::operator()(const std::array<DTYPE(data), 3>& source_coords) \
      const;                                                                  \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>  \
  SphericalTorus::jacobian(const std::array<DTYPE(data), 3>& source_coords)   \
      const;                                                                  \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>  \
  SphericalTorus::inv_jacobian(                                               \
      const std::array<DTYPE(data), 3>& source_coords) const;                 \
  template tnsr::Ijj<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  SphericalTorus::hessian(const std::array<DTYPE(data), 3>& source_coords)    \
      const;                                                                  \
  template tnsr::Ijk<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  SphericalTorus::derivative_of_inv_jacobian(                                 \
      const std::array<DTYPE(data), 3>& source_coords) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))
#undef INSTANTIATE
#undef DTYPE

}  // namespace domain::CoordinateMaps
