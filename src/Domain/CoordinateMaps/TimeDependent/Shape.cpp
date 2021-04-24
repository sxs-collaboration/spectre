// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/TimeDependent/Shape.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <pup_stl.h>
#include <string>

#include "ApparentHorizons/SpherepackIterator.hpp"
#include "ApparentHorizons/TagsTypeAliases.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/ShapeMapTransitionFunction.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"

namespace domain::CoordinateMaps::TimeDependent {

template <typename T>
std::array<T, 2> cartesian_to_spherical(
    const std::array<T, 3>& cartesian) noexcept {
  const auto& [x, y, z] = cartesian;
  return {atan2(hypot(x, y), z), atan2(y, x)};
}

Shape::Shape(
    const std::array<double, 3>& center, const size_t l_max, const size_t m_max,
    std::unique_ptr<ShapeMapTransitionFunctions::ShapeMapTransitionFunction>
        transition_func,
    std::string function_of_time_name) noexcept
    : f_of_t_name_(std::move(function_of_time_name)),
      center_(center),
      l_max_(l_max),
      m_max_(m_max),
      ylm_(l_max, m_max),
      transition_func_(std::move(transition_func)) {
  ASSERT(l_max >= 2, "The shape map requires l_max >= 2 but l_max = " << l_max);
  ASSERT(m_max >= 2, "The shape map requires m_max >= 2 but m_max = " << m_max);
  ASSERT(l_max >= m_max, "The shape map requires l_max >= m_max but l_max = "
                             << l_max << ", m_max = " << m_max);
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> Shape::operator()(
    const std::array<T, 3>& source_coords, const double time,
    const FunctionsOfTimeMap& functions_of_time) const noexcept {
  ASSERT(functions_of_time.find(f_of_t_name_) != functions_of_time.end(),
         "Could not find function of time: '"
             << f_of_t_name_ << "' in functions of time. Known functions are "
             << keys_of(functions_of_time));

  const auto centered_coords = center_coordinates(source_coords);
  auto theta_phis = cartesian_to_spherical(centered_coords);
  const auto interpolation_info = ylm_.set_up_interpolation_info(theta_phis);
  const DataVector coefs = functions_of_time.at(f_of_t_name_)->func(time)[0];
  check_coefficients(coefs);
  // re-use allocation
  auto& distorted_radii = get<0>(theta_phis);
  // evaluate the spherical harmonic expansion at the angles of `source_coords`
  ylm_.interpolate_from_coefs(make_not_null(&distorted_radii), coefs,
                              interpolation_info);

  // this should be taken care of by the control system but is very hard to
  // debug
#ifdef SPECTRE_DEBUG
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType shift_radii =
      distorted_radii * transition_func_->operator()(centered_coords);
  if constexpr (std::is_same_v<ReturnType, double>) {
    ASSERT(shift_radii < 1., "Coordinates mapped through the center!");
  } else {
    for (const auto& radius : shift_radii) {
      ASSERT(radius < 1., "Coordinates mapped through the center!");
    }
  }
#endif  // SPECTRE_DEBUG

  return center_ + centered_coords *
                       (1. - distorted_radii *
                                 transition_func_->operator()(centered_coords));
}

std::optional<std::array<double, 3>> Shape::inverse(
    const std::array<double, 3>& target_coords, const double time,
    const FunctionsOfTimeMap& functions_of_time) const noexcept {
  ASSERT(functions_of_time.find(f_of_t_name_) != functions_of_time.end(),
         "Could not find function of time: '"
             << f_of_t_name_ << "' in functions of time. Known functions are "
             << keys_of(functions_of_time));

  const std::array<double, 3> centered_coords =
      center_coordinates(target_coords);
  const std::array<double, 2> theta_phis =
      cartesian_to_spherical(centered_coords);
  const DataVector coefs = functions_of_time.at(f_of_t_name_)->func(time)[0];
  check_coefficients(coefs);
  const double distorted_radii = ylm_.interpolate_from_coefs(coefs, theta_phis);
  const std::optional<double> original_radius_over_radius =
      transition_func_->original_radius_over_radius(centered_coords,
                                                    distorted_radii);
  if (not original_radius_over_radius.has_value()) {
    return std::nullopt;
  }
  return center_ + centered_coords * original_radius_over_radius.value();
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> Shape::frame_velocity(
    const std::array<T, 3>& source_coords, const double time,
    const FunctionsOfTimeMap& functions_of_time) const noexcept {
  ASSERT(functions_of_time.find(f_of_t_name_) != functions_of_time.end(),
         "Could not find function of time: '"
             << f_of_t_name_ << "' in functions of time. Known functions are "
             << keys_of(functions_of_time));
  const auto centered_coords = center_coordinates(source_coords);
  auto theta_phis = cartesian_to_spherical(centered_coords);
  const auto interpolation_info = ylm_.set_up_interpolation_info(theta_phis);
  const auto coef_derivs =
      functions_of_time.at(f_of_t_name_)->func_and_deriv(time)[1];
  check_coefficients(coef_derivs);
  // re-use allocation
  auto& radii_velocities = get<0>(theta_phis);
  ylm_.interpolate_from_coefs(make_not_null(&radii_velocities), coef_derivs,
                              interpolation_info);
  return -centered_coords * radii_velocities *
         transition_func_->operator()(centered_coords);
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> Shape::jacobian(
    const std::array<T, 3>& source_coords, const double time,
    const FunctionsOfTimeMap& functions_of_time) const noexcept {
  ASSERT(functions_of_time.find(f_of_t_name_) != functions_of_time.end(),
         "Could not find function of time: '"
             << f_of_t_name_ << "' in functions of time. Known functions are "
             << keys_of(functions_of_time));

  const auto centered_coords = center_coordinates(source_coords);

  // The distorted radii are calculated analogously to the call operator
  auto theta_phis = cartesian_to_spherical(centered_coords);

  // The Cartesian gradient cannot be represented exactly by `l_max_` and
  // `m_max_` which causes an aliasing error. We need an additional order to
  // represent it. This is in theory not needed for the distorted_radii
  // calculation but saves calculating the `interpolation_info` twice.
  const YlmSpherepack extended_ylm(l_max_ + 1, m_max_ + 1);
  const auto interpolation_info =
      extended_ylm.set_up_interpolation_info(theta_phis);

  const DataVector coefs = functions_of_time.at(f_of_t_name_)->func(time)[0];
  check_coefficients(coefs);
  DataVector extended_coefs(extended_ylm.spectral_size(), 0.);

  // Copy over the coefficients. The additional coefficients of order `l_max_
  // +1` are zero and will only have an effect in the interpolation of the
  // cartesian gradient.
  SpherepackIterator extended_iter(l_max_ + 1, m_max_ + 1);
  SpherepackIterator iter(l_max_, m_max_);
  for (size_t l = 2; l <= l_max_; ++l) {
    const int m_max = std::min(l, m_max_);
    for (int m = -m_max; m <= m_max; ++m) {
      iter.set(l, m);
      extended_iter.set(l, m);
      extended_coefs[extended_iter()] = coefs[iter()];
    }
  }

  // Re-use allocation
  auto& distorted_radii = get<0>(theta_phis);
  extended_ylm.interpolate_from_coefs(make_not_null(&distorted_radii),
                                      extended_coefs, interpolation_info);
  // Calculates the Pfaffian derivative at the internal collocation points of
  // YlmSpherePack. We can't interpolate these directly as they are not smooth
  // across the poles, so we convert them to the Cartesian gradients first,
  // which are smooth.
  const auto angular_gradient =
      extended_ylm.gradient_from_coefs(extended_coefs);

  tnsr::i<DataVector, 3, Frame::Inertial> cartesian_gradient(
      extended_ylm.physical_size());

  // Re-use allocations
  std::array<DataVector, 2> collocation_theta_phis{};
  collocation_theta_phis[0].set_data_ref(&get<2>(cartesian_gradient));
  collocation_theta_phis[1].set_data_ref(&get<1>(cartesian_gradient));
  collocation_theta_phis = extended_ylm.theta_phi_points();

  const auto& col_thetas = collocation_theta_phis[0];
  const auto& col_phis = collocation_theta_phis[1];

  // The Cartesian derivative is the Pfaffian derivative multiplied by the
  // inverse Jacobian matrix. Some optimizations here may be possible by
  // introducing temporaries for some of the sin/cos which are computed twice,
  // if the compiler CSE doesn't take care of it.
  get<0>(cartesian_gradient) =
      (cos(col_thetas) * cos(col_phis) * get<0>(angular_gradient) -
       sin(col_phis) * get<1>(angular_gradient));

  get<1>(cartesian_gradient) =
      (cos(col_thetas) * sin(col_phis) * get<0>(angular_gradient) +
       cos(col_phis) * get<1>(angular_gradient));

  get<2>(cartesian_gradient) = -sin(col_thetas) * get<0>(angular_gradient);

  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> result(
      get_size(centered_coords[0]));

  // re-use allocation
  auto& target_gradient_x = get<2, 0>(result);
  auto& target_gradient_y = get<2, 1>(result);
  auto& target_gradient_z = get<2, 2>(result);

  // interpolate the cartesian gradient to the thetas and phis of the
  // `source_coords`
  extended_ylm.interpolate(make_not_null(&target_gradient_x),
                           get<0>(cartesian_gradient).data(),
                           interpolation_info);
  extended_ylm.interpolate(make_not_null(&target_gradient_y),
                           get<1>(cartesian_gradient).data(),
                           interpolation_info);
  extended_ylm.interpolate(make_not_null(&target_gradient_z),
                           get<2>(cartesian_gradient).data(),
                           interpolation_info);

  // re-use allocation
  auto& transition_func = get<1>(theta_phis);
  transition_func = transition_func_->operator()(centered_coords);
  // leave it to the domain specific transition function to divide by the radii
  // as it can safely do this
  const auto transition_func_over_radius =
      transition_func_->map_over_radius(centered_coords);

  const auto transition_func_gradient =
      transition_func_->gradient(centered_coords);

  const auto& [x_transition_gradient, y_transition_gradient,
               z_transition_gradient] = transition_func_gradient;
  const auto& [x_centered, y_centered, z_centered] = centered_coords;

  get<0, 0>(result) =
      -x_centered * (x_transition_gradient * distorted_radii +
                     target_gradient_x * transition_func_over_radius);
  get<0, 1>(result) =
      -x_centered * (y_transition_gradient * distorted_radii +
                     target_gradient_y * transition_func_over_radius);
  get<0, 2>(result) =
      -x_centered * (z_transition_gradient * distorted_radii +
                     target_gradient_z * transition_func_over_radius);
  get<1, 0>(result) =
      -y_centered * (x_transition_gradient * distorted_radii +
                     target_gradient_x * transition_func_over_radius);
  get<1, 1>(result) =
      -y_centered * (y_transition_gradient * distorted_radii +
                     target_gradient_y * transition_func_over_radius);
  get<1, 2>(result) =
      -y_centered * (z_transition_gradient * distorted_radii +
                     target_gradient_z * transition_func_over_radius);
  get<2, 0>(result) =
      -z_centered * (x_transition_gradient * distorted_radii +
                     target_gradient_x * transition_func_over_radius);
  get<2, 1>(result) =
      -z_centered * (y_transition_gradient * distorted_radii +
                     target_gradient_y * transition_func_over_radius);
  get<2, 2>(result) =
      -z_centered * (z_transition_gradient * distorted_radii +
                     target_gradient_z * transition_func_over_radius);

  get<0, 0>(result) += 1. - distorted_radii * transition_func;
  get<1, 1>(result) += 1. - distorted_radii * transition_func;
  get<2, 2>(result) += 1. - distorted_radii * transition_func;

  return result;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> Shape::inv_jacobian(
    const std::array<T, 3>& source_coords, const double time,
    const FunctionsOfTimeMap& functions_of_time) const noexcept {
  return determinant_and_inverse(
             jacobian(source_coords, time, functions_of_time))
      .second;
}

void Shape::check_coefficients(
    [[maybe_unused]] const DataVector& coefs) const noexcept {
#ifdef SPECTRE_DEBUG
  // The expected format of the coefficients passed from the control system can
  // be changed depending on what turns out to be most convenient for the
  // control system
  ASSERT(coefs.size() == ylm_.spectral_size(),
         "Spectral coefficients are expected to be in YlmSpherepack format "
         "with size 2 * (l_max + 1) * (m_max + 1) = "
             << ylm_.spectral_size() << ", but have size " << coefs.size());

  SpherepackIterator iter(l_max_, m_max_);
  const std::vector<std::pair<size_t, int>> mono_and_dipole{
      {0, 0}, {1, 0}, {1, -1}, {1, 1}};
  for (const auto& [l, m] : mono_and_dipole) {
    iter.set(l, m);
    ASSERT(
        coefs[iter()] == 0.,
        "The shape map expects the values of the mono- and dipole (l = 0, 1) "
        "to be zero, but the coefficients for l = "
            << l << ", m = " << m << " have value " << coefs[iter()]);
  }
#endif  // SPECTRE_DEBUG
}

bool operator==(const Shape& lhs, const Shape& rhs) noexcept {
  return lhs.f_of_t_name_ == rhs.f_of_t_name_ and lhs.center_ == rhs.center_ and
         lhs.l_max_ == rhs.l_max_ and lhs.m_max_ == rhs.m_max_ and
         *lhs.transition_func_ == *rhs.transition_func_;
}

bool operator!=(const Shape& lhs, const Shape& rhs) noexcept {
  return not(lhs == rhs);
}

void Shape::pup(PUP::er& p) noexcept {
  p | l_max_;
  p | m_max_;
  p | center_;
  p | f_of_t_name_;
  p | transition_func_;

  if (p.isUnpacking()) {
    ylm_ = YlmSpherepack(l_max_, m_max_);
  }
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                                  \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>                \
  Shape::operator()(const std::array<DTYPE(data), 3>& source_coords,          \
                    double time, const FunctionsOfTimeMap& functions_of_time) \
      const;                                                                  \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>                \
  Shape::frame_velocity(const std::array<DTYPE(data), 3>& source_coords,      \
                        double time,                                          \
                        const FunctionsOfTimeMap& functions_of_time) const;   \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>  \
  Shape::jacobian(const std::array<DTYPE(data), 3>& source_coords,            \
                  double time, const FunctionsOfTimeMap& functions_of_time)   \
      const;                                                                  \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>  \
  Shape::inv_jacobian(const std::array<DTYPE(data), 3>& source_coords,        \
                      double time,                                            \
                      const FunctionsOfTimeMap& functions_of_time) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))
#undef DTYPE
#undef INSTANTIATE

}  // namespace domain::CoordinateMaps::TimeDependent
