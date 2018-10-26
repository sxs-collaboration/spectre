// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GrMhd/MagnetizedFmDisk.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <pup.h>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace grmhd {
namespace AnalyticData {

MagnetizedFmDisk::MagnetizedFmDisk(
    const double bh_mass, const double bh_dimless_spin,
    const double inner_edge_radius, const double max_pressure_radius,
    const double polytropic_constant, const double polytropic_exponent,
    const double threshold_density, const double inverse_plasma_beta,
    const size_t normalization_grid_res) noexcept
    : RelativisticEuler::Solutions::FishboneMoncriefDisk(
          bh_mass, bh_dimless_spin, inner_edge_radius, max_pressure_radius,
          polytropic_constant, polytropic_exponent),
      threshold_density_(threshold_density),
      inverse_plasma_beta_(inverse_plasma_beta),
      normalization_grid_res_(normalization_grid_res),
      kerr_schild_coords_{bh_mass, bh_dimless_spin} {
  ASSERT(threshold_density_ > 0.0 and threshold_density_ < 1.0,
         "The threshold density must be in the range (0, 1). The value given "
         "was "
             << threshold_density_ << ".");
  ASSERT(inverse_plasma_beta_ >= 0.0,
         "The inverse plasma beta must be non-negative. The value given "
         "was "
             << inverse_plasma_beta_ << ".");
  ASSERT(normalization_grid_res_ >= 4,
         "The grid resolution used in the magnetic field normalization must be "
         "at least 4 points. The value given was "
             << normalization_grid_res_);

  // The normalization of the magnetic field is determined by the condition
  //
  // plasma beta = pgas_max / pmag_max
  //
  // where pgas_max and pmag_max are the maximum values of the gas and magnetic
  // pressure over the domain, respectively. pgas_max is obtained by
  // evaluating the pressure of the Fishbone-Moncrief solution at the
  // corresponding radius. In order to compare with other groups, pmag_max is
  // obtained by precomputing the unnormalized magnetic field and then
  // finding the maximum of the comoving field squared on a given grid.
  // Following the CodeComparisonProject files, we choose to find the maximum
  // on a spherical grid as described below.

  // Set up a Kerr ("spherical Kerr-Schild") grid of constant phi = 0,
  // whose parameterization in Cartesian Kerr-Schild coordinates is
  // x(r, theta) = r sin_theta,
  // y(r, theta) = a sin_theta,
  // z(r, theta) = r cos_theta.
  const double r_init = inner_edge_radius_ * bh_mass_;
  const double r_fin = max_pressure_radius_ * bh_mass_;
  const double dr = (r_fin - r_init) / normalization_grid_res_;
  const double dtheta = M_PI / normalization_grid_res_;
  const double cartesian_n_pts = square(normalization_grid_res_);
  Index<2> extents(normalization_grid_res_, normalization_grid_res_);
  auto grid =
      make_with_value<tnsr::I<DataVector, 3>>(DataVector(cartesian_n_pts), 0.0);
  for (size_t i = 0; i < normalization_grid_res_; ++i) {
    double r_i = r_init + static_cast<double>(i) * dr;
    for (size_t j = 0; j < normalization_grid_res_; ++j) {
      double sin_theta_j = sin(static_cast<double>(j) * dtheta);
      auto index = collapsed_index(Index<2>{i, j}, extents);
      get<0>(grid)[index] = r_i * sin_theta_j;
      get<1>(grid)[index] = bh_spin_a_ * sin_theta_j;
      get<2>(grid)[index] = r_i * cos(static_cast<double>(j) * dtheta);
    }
  }

  const auto b_field = unnormalized_magnetic_field(grid);
  const auto spatial_metric = get<
      gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(variables(
      grid,
      tmpl::list<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>{}));

  const tnsr::I<double, 3, Frame::Inertial> x_max{
      {{max_pressure_radius_, bh_spin_a_, 0.0}}};

  const double b_squared_max = max(
      get(dot_product(b_field, b_field, spatial_metric)) /
          square(get(get<hydro::Tags::LorentzFactor<DataVector>>(variables(
              grid, tmpl::list<hydro::Tags::LorentzFactor<DataVector>>{})))) +
      square(get(dot_product(
          b_field,
          get<hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>>(
              variables(grid, tmpl::list<hydro::Tags::SpatialVelocity<
                                  DataVector, 3, Frame::Inertial>>{})),
          spatial_metric))));
  ASSERT(b_squared_max > 0.0, "Max b squared is zero.");
  b_field_normalization_ =
      sqrt(2.0 *
           get(get<hydro::Tags::Pressure<double>>(
               variables(x_max, tmpl::list<hydro::Tags::Pressure<double>>{}))) *
           inverse_plasma_beta_ / b_squared_max);
}

void MagnetizedFmDisk::pup(PUP::er& p) noexcept {
  RelativisticEuler::Solutions::FishboneMoncriefDisk::pup(p);
  p | threshold_density_;
  p | inverse_plasma_beta_;
  p | b_field_normalization_;
  p | normalization_grid_res_;
  p | kerr_schild_coords_;
}

template <typename DataType>
tnsr::I<DataType, 3, Frame::Inertial>
MagnetizedFmDisk::unnormalized_magnetic_field(
    const tnsr::I<DataType, 3, Frame::Inertial>& x) const noexcept {
  auto magnetic_field =
      make_with_value<tnsr::I<DataType, 3, Frame::NoFrame>>(x, 0.0);

  // The maximum pressure (and hence the maximum rest mass density) is located
  // on the ring x^2 + y^2 = r_max^2 + a^2, z = 0.
  // Note that `x` may or may not include points on this ring.
  const tnsr::I<double, 3, Frame::Inertial> x_max{
      {{max_pressure_radius_, bh_spin_a_, 0.0}}};
  const double threshold_rest_mass_density =
      threshold_density_ *
      get(get<hydro::Tags::RestMassDensity<double>>(variables(
          x_max, tmpl::list<hydro::Tags::RestMassDensity<double>>{})));

  const double inner_edge_potential =
      fm_disk::potential(square(inner_edge_radius_), 1.0);

  // A_phi \propto rho - rho_threshold. Normalization comes later.
  const auto mag_potential =
      [ this, &threshold_rest_mass_density, &inner_edge_potential ](
          const double& r, const double& sin_theta_squared) noexcept {
    // enthalpy = exp(Win - W(r,theta)), as in the Fishbone-Moncrief disk
    return get(equation_of_state_.rest_mass_density_from_enthalpy(
               Scalar<double>{
                   exp(inner_edge_potential -
                       fm_disk::potential(square(r), sin_theta_squared))})) -
           threshold_rest_mass_density;
  };

  // The magnetic field is present only within the disk, where the
  // rest mass density is greater than the threshold rest mass density.
  const DataType rest_mass_density =
      get(get<hydro::Tags::RestMassDensity<DataType>>(
          variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})));
  for (size_t s = 0; s < get_size(get<0>(x)); ++s) {
    if (get_element(rest_mass_density, s) > threshold_rest_mass_density) {
      // The magnetic field is defined in terms of the Faraday tensor in Kerr
      // (r, theta, phi) coordinates. We need to get B^r, B^theta, B^phi first,
      // then we transform to Cartesian coordinates.
      const double z_squared = square(get_element(get<2>(x), s));
      const double a_squared = bh_spin_a_ * bh_spin_a_;

      double sin_theta_squared =
          square(get_element(get<0>(x), s)) + square(get_element(get<1>(x), s));
      double r_squared = 0.5 * (sin_theta_squared + z_squared - a_squared);
      r_squared += sqrt(square(r_squared) + a_squared * z_squared);
      sin_theta_squared /= (r_squared + a_squared);

      // B^i is proportional to derivatives of the magnetic potential. One has
      //
      // B^r = sqrt(gamma) * \partial_\theta A_\phi
      // B^\theta = -sqrt(gamma) * \partial_r A_\phi
      //
      // where sqrt(gamma) = sqrt( sigma (sigma + 2Mr) ) * sin\theta.
      const double radius = sqrt(r_squared);
      const double sin_theta = sqrt(sin_theta_squared);
      double prefactor = r_squared + square(bh_spin_a_) * z_squared / r_squared;
      prefactor *= (prefactor + 2.0 * bh_mass_ * radius);

      // As done by SpEC and other groups, we approximate the derivatives
      // with a 2nd-order centered finite difference expression.
      // For simplicity, we set \delta r = \delta\theta = small number.
      const double small = 0.0001 * bh_mass_;
      prefactor = 2.0 * small * sqrt(prefactor) * sin_theta;
      prefactor = 1.0 / prefactor;

      // Since the metric, and thus the field, depend on theta through
      // sin^2(theta) only, we use chain rule in B^\theta and write
      //
      // d/d\theta = 2 * sin\theta * cos\theta * d/d(sin^2(theta)),
      //
      // where cos\theta = z/r.
      get_element(get<0>(magnetic_field), s) =
          2.0 * prefactor * sin_theta * get_element(get<2>(x), s) *
          (mag_potential(radius, sin_theta_squared + small) -
           mag_potential(radius, sin_theta_squared - small)) /
          radius;
      get_element(get<1>(magnetic_field), s) =
          prefactor * (mag_potential(radius - small, sin_theta_squared) -
                       mag_potential(radius + small, sin_theta_squared));
      // phi component of the field vanishes identically.
    }
  }
  return kerr_schild_coords_.cartesian_from_spherical_ks(
      std::move(magnetic_field), x);
}

template <typename DataType, bool NeedSpacetime>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>
MagnetizedFmDisk::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<
        hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>> /*meta*/,
    const IntermediateVariables<DataType, NeedSpacetime>& /*vars*/,
    const size_t /*index*/) const noexcept {
  auto result = unnormalized_magnetic_field(x);
  for (size_t i = 0; i < 3; ++i) {
    result.get(i) *= b_field_normalization_;
  }
  return result;
}

bool operator==(const MagnetizedFmDisk& lhs,
                const MagnetizedFmDisk& rhs) noexcept {
  using fm_disk = MagnetizedFmDisk::fm_disk;
  return *static_cast<const fm_disk*>(&lhs) ==
             *static_cast<const fm_disk*>(&rhs) and
         lhs.threshold_density_ == rhs.threshold_density_ and
         lhs.inverse_plasma_beta_ == rhs.inverse_plasma_beta_ and
         lhs.b_field_normalization_ == rhs.b_field_normalization_ and
         lhs.normalization_grid_res_ == rhs.normalization_grid_res_;
}

bool operator!=(const MagnetizedFmDisk& lhs,
                const MagnetizedFmDisk& rhs) noexcept {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define NEED_SPACETIME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                            \
  template tuples::TaggedTuple<                                         \
      hydro::Tags::MagneticField<DTYPE(data), 3, Frame::Inertial>>      \
  MagnetizedFmDisk::variables(                                          \
      const tnsr::I<DTYPE(data), 3>& x,                                 \
      tmpl::list<hydro::Tags::MagneticField<DTYPE(data), 3,             \
                                            Frame::Inertial>> /*meta*/, \
      const FishboneMoncriefDisk::IntermediateVariables<                \
          DTYPE(data), NEED_SPACETIME(data)>& vars,                     \
      const size_t) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector), (true, false))

#undef DTYPE
#undef NEED_SPACETIME
#undef INSTANTIATE
}  // namespace AnalyticData
}  // namespace grmhd
/// \endcond
