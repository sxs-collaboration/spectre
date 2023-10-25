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
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace grmhd::AnalyticData {

MagnetizedFmDisk::MagnetizedFmDisk(
    const double bh_mass, const double bh_dimless_spin,
    const double inner_edge_radius, const double max_pressure_radius,
    const double polytropic_constant, const double polytropic_exponent,
    const double threshold_density, const double inverse_plasma_beta,
    const size_t normalization_grid_res)
    : fm_disk_(bh_mass, bh_dimless_spin, inner_edge_radius, max_pressure_radius,
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
  const double r_init = fm_disk_.inner_edge_radius_ * fm_disk_.bh_mass_;
  const double r_fin = fm_disk_.max_pressure_radius_ * fm_disk_.bh_mass_;
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
      get<1>(grid)[index] = fm_disk_.bh_spin_a_ * sin_theta_j;
      get<2>(grid)[index] = r_i * cos(static_cast<double>(j) * dtheta);
    }
  }

  const auto b_field = unnormalized_magnetic_field(grid);
  const auto unmagnetized_vars = variables(
      grid, tmpl::list<gr::Tags::SpatialMetric<DataVector, 3>,
                       hydro::Tags::LorentzFactor<DataVector>,
                       hydro::Tags::SpatialVelocity<DataVector, 3>>{});
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3>>(unmagnetized_vars);

  const tnsr::I<double, 3> x_max{
      {{fm_disk_.max_pressure_radius_, 0.0, 0.0}}};

  const double b_squared_max = max(
      get(dot_product(b_field, b_field, spatial_metric)) /
          square(get(
              get<hydro::Tags::LorentzFactor<DataVector>>(unmagnetized_vars))) +
      square(get(dot_product(
          b_field,
          get<hydro::Tags::SpatialVelocity<DataVector, 3>>(unmagnetized_vars),
          spatial_metric))));
  ASSERT(b_squared_max > 0.0, "Max b squared is zero.");
  b_field_normalization_ =
      sqrt(2.0 *
           get(get<hydro::Tags::Pressure<double>>(
               variables(x_max, tmpl::list<hydro::Tags::Pressure<double>>{}))) *
           inverse_plasma_beta_ / b_squared_max);
}

std::unique_ptr<evolution::initial_data::InitialData>
MagnetizedFmDisk::get_clone() const {
  return std::make_unique<MagnetizedFmDisk>(*this);
}

MagnetizedFmDisk::MagnetizedFmDisk(CkMigrateMessage* msg) : fm_disk_(msg) {}

void MagnetizedFmDisk::pup(PUP::er& p) {
  p | fm_disk_;
  p | threshold_density_;
  p | inverse_plasma_beta_;
  p | b_field_normalization_;
  p | normalization_grid_res_;
  p | kerr_schild_coords_;
}

template <typename DataType>
tnsr::I<DataType, 3> MagnetizedFmDisk::unnormalized_magnetic_field(
    const tnsr::I<DataType, 3>& x) const {
  auto magnetic_field =
      make_with_value<tnsr::I<DataType, 3, Frame::NoFrame>>(x, 0.0);

  auto x_ks = x;

  // The maximum pressure (and hence the maximum rest mass density) is located
  // on the ring x^2 + y^2 = r_max^2, z = 0.
  // Note that `x` may or may not include points on this ring.
  const tnsr::I<double, 3> x_max{
      {{fm_disk_.max_pressure_radius_, 0.0, 0.0}}};
  const double threshold_rest_mass_density =
      threshold_density_ *
      get(get<hydro::Tags::RestMassDensity<double>>(variables(
          x_max, tmpl::list<hydro::Tags::RestMassDensity<double>>{})));

  const double inner_edge_potential =
      fm_disk_.potential(square(fm_disk_.inner_edge_radius_), 1.0);

  // A_phi \propto rho - rho_threshold. Normalization comes later.
  const auto mag_potential = [this, &threshold_rest_mass_density,
                              &inner_edge_potential](
                                 const double r,
                                 const double sin_theta_squared) {
    // enthalpy = exp(Win - W(r,theta)), as in the Fishbone-Moncrief disk
    return get(equation_of_state().rest_mass_density_from_enthalpy(
               Scalar<double>{
                   exp(inner_edge_potential -
                       fm_disk_.potential(square(r), sin_theta_squared))})) -
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
      double sin_theta_squared =
          square(get_element(get<0>(x), s)) + square(get_element(get<1>(x), s));
      double r_squared = sin_theta_squared + z_squared;
      sin_theta_squared /= r_squared;

      // B^i is proportional to derivatives of the magnetic potential. One has
      //
      // B^r = sqrt(gamma) * \partial_\theta A_\phi
      // B^\theta = -sqrt(gamma) * \partial_r A_\phi
      //
      // where sqrt(gamma) = sqrt( sigma (sigma + 2Mr) ) * sin\theta.
      const double radius = sqrt(r_squared);
      const double sin_theta = sqrt(sin_theta_squared);
      double prefactor =
          r_squared + square(fm_disk_.bh_spin_a_) * z_squared / r_squared;
      prefactor *= (prefactor + 2.0 * fm_disk_.bh_mass_ * radius);

      // As done by SpEC and other groups, we approximate the derivatives
      // with a 2nd-order centered finite difference expression.
      // For simplicity, we set \delta r = \delta\theta = small number.
      const double small = 0.0001 * fm_disk_.bh_mass_;
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

      // Need x in KS coordinates instead of SKS coordinates
      // to do transformation below. copy x into x_sks and alter at each
      // element index s.
      const double sks_to_ks_factor =
          sqrt(r_squared + square(fm_disk_.bh_spin_a_)) / radius;
      get_element(x_ks.get(0), s) = get_element(x.get(0), s) * sks_to_ks_factor;
      get_element(x_ks.get(1), s) = get_element(x.get(1), s) * sks_to_ks_factor;
      get_element(x_ks.get(2), s) = get_element(x.get(2), s);
    }
  }
  // magnetic field in KS_coords
  // change back to SKS_coords
  const auto magnetic_field_ks =
      kerr_schild_coords_.cartesian_from_spherical_ks(std::move(magnetic_field),
                                                      x_ks);

  using inv_jacobian =
      gr::Solutions::SphericalKerrSchild::internal_tags::inv_jacobian<
          DataType, Frame::Inertial>;

  FmDisk::IntermediateVariables<DataType> vars(x);

  const auto inv_jacobians =
      get<inv_jacobian>(fm_disk_.background_spacetime_.variables(
          x, 0.0, tmpl::list<inv_jacobian>{},
          make_not_null(&vars.sph_kerr_schild_cache)));

  auto magnetic_field_sks =
      make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.0);

  for (size_t j = 0; j < 3; ++j) {
    for (size_t i = 0; i < 3; ++i) {
      magnetic_field_sks.get(j) +=
          inv_jacobians.get(j, i) * magnetic_field_ks.get(i);
    }
  }

  return magnetic_field_sks;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>
MagnetizedFmDisk::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/,
    gsl::not_null<FmDisk::IntermediateVariables<DataType>*> /*vars*/) const {
  auto result = unnormalized_magnetic_field(x);
  for (size_t i = 0; i < 3; ++i) {
    result.get(i) *= b_field_normalization_;
  }
  return result;
}

PUP::able::PUP_ID MagnetizedFmDisk::my_PUP_ID = 0;

bool operator==(const MagnetizedFmDisk& lhs, const MagnetizedFmDisk& rhs) {
  return lhs.fm_disk_ == rhs.fm_disk_ and
         lhs.threshold_density_ == rhs.threshold_density_ and
         lhs.inverse_plasma_beta_ == rhs.inverse_plasma_beta_ and
         lhs.b_field_normalization_ == rhs.b_field_normalization_ and
         lhs.normalization_grid_res_ == rhs.normalization_grid_res_;
}

bool operator!=(const MagnetizedFmDisk& lhs, const MagnetizedFmDisk& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template tuples::TaggedTuple<hydro::Tags::MagneticField<DTYPE(data), 3>>   \
  MagnetizedFmDisk::variables(                                               \
      const tnsr::I<DTYPE(data), 3>& x,                                      \
      tmpl::list<hydro::Tags::MagneticField<DTYPE(data), 3>> /*meta*/,       \
      gsl::not_null<                                                         \
          FmDisk::FishboneMoncriefDisk::IntermediateVariables<DTYPE(data)>*> \
          vars) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
}  // namespace grmhd::AnalyticData
