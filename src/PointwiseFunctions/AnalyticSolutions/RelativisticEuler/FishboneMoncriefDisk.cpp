// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/FishboneMoncriefDisk.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <pup.h>

#include "DataStructures/DataVector.hpp"                   // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/EquationsOfState/PolytropicFluid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace RelativisticEuler {
namespace Solutions {

FishboneMoncriefDisk::FishboneMoncriefDisk(
    const double black_hole_mass, const double black_hole_spin,
    const double inner_edge_radius, const double max_pressure_radius,
    const double polytropic_constant, const double polytropic_exponent) noexcept
    : black_hole_mass_(black_hole_mass),
      black_hole_spin_(black_hole_spin),
      inner_edge_radius_(inner_edge_radius),
      max_pressure_radius_(max_pressure_radius),
      polytropic_constant_(polytropic_constant),
      polytropic_exponent_(polytropic_exponent) {
  ASSERT(black_hole_mass_ > 0.0,
         "The black hole mass must be positive. The value given was "
             << black_hole_mass_ << ".");
  ASSERT(black_hole_spin_ >= 0.0 and black_hole_spin_ < 1.0,
         "The black hole spin magnitude must be in the range [0, 1). "
         "The value given was "
             << black_hole_spin_ << ".");
  ASSERT(polytropic_constant_ > 0.0,
         "The polytropic constant must be positive. The value given was "
             << polytropic_constant_ << ".");
  ASSERT(polytropic_exponent_ > 1.0,
         "The polytropic exponent must be greater than 1. The value given was "
             << polytropic_exponent_ << ".");
}

void FishboneMoncriefDisk::pup(PUP::er& p) noexcept {
  p | black_hole_mass_;
  p | black_hole_spin_;
  p | inner_edge_radius_;
  p | max_pressure_radius_;
  p | polytropic_constant_;
  p | polytropic_exponent_;
}

template <typename DataType>
DataType FishboneMoncriefDisk::sigma(const DataType& r_sqrd,
                                     const DataType& sin_theta_sqrd) const
    noexcept {
  return r_sqrd + square(black_hole_spin_) * (1.0 - sin_theta_sqrd);
}

template <typename DataType>
DataType FishboneMoncriefDisk::inv_ucase_a(const DataType& r_sqrd,
                                           const DataType& sin_theta_sqrd,
                                           const DataType& delta) const
    noexcept {
  const double a_sqrd = square(black_hole_spin_);
  const DataType r_sqrd_plus_a_sqrd = r_sqrd + a_sqrd;
  return 1.0 / (square(r_sqrd_plus_a_sqrd) - delta * a_sqrd * sin_theta_sqrd);
}

template <typename DataType>
DataType FishboneMoncriefDisk::four_velocity_t_sqrd(
    const DataType& r_sqrd, const DataType& sin_theta_sqrd,
    const double angular_momentum) const noexcept {
  const DataType delta =
      r_sqrd - 2.0 * black_hole_mass_ * sqrt(r_sqrd) + square(black_hole_spin_);
  const DataType prefactor = 0.5 / (inv_ucase_a(r_sqrd, sin_theta_sqrd, delta) *
                                    sigma(r_sqrd, sin_theta_sqrd));
  return prefactor * (1.0 + sqrt(1.0 +
                                 square(angular_momentum) * delta /
                                     (square(prefactor) * sin_theta_sqrd))) /
         delta;
}

template <typename DataType>
DataType FishboneMoncriefDisk::angular_velocity(
    const DataType& r_sqrd, const DataType& sin_theta_sqrd,
    const double angular_momentum) const noexcept {
  const DataType r = sqrt(r_sqrd);
  return inv_ucase_a(r_sqrd, sin_theta_sqrd,
                     DataType{r_sqrd - 2.0 * black_hole_mass_ * r +
                              square(black_hole_spin_)}) *
         (angular_momentum * sigma(r_sqrd, sin_theta_sqrd) /
              (four_velocity_t_sqrd(r_sqrd, sin_theta_sqrd, angular_momentum) *
               sin_theta_sqrd) +
          2.0 * black_hole_mass_ * r * black_hole_spin_);
}

template <typename DataType>
DataType FishboneMoncriefDisk::potential(const DataType& r_sqrd,
                                         const DataType& sin_theta_sqrd,
                                         const double angular_momentum) const
    noexcept {
  return angular_momentum *
             angular_velocity(r_sqrd, sin_theta_sqrd, angular_momentum) -
         log(sqrt(
             four_velocity_t_sqrd(r_sqrd, sin_theta_sqrd, angular_momentum)));
}

template <typename DataType>
tuples::tagged_tuple_from_typelist<
    FishboneMoncriefDisk::variables_tags<DataType>>
FishboneMoncriefDisk::variables(
    const tnsr::I<DataType, 3>& x, const double t,
    FishboneMoncriefDisk::variables_tags<DataType> /*meta*/) const noexcept {
  const double a_sqrd = black_hole_spin_ * black_hole_spin_;

  // Boyer-Lindquist (r, theta) coords in terms of Kerr-Schild coords (x, y, z)
  DataType z_sqrd = square(x.get(2));
  const auto r_sqrd = [&x, &z_sqrd, &a_sqrd ]() noexcept {
    const DataType temp =
        0.5 * (square(x.get(0)) + square(x.get(1)) + z_sqrd - a_sqrd);
    return DataType{temp + sqrt(square(temp) + a_sqrd * z_sqrd)};
  }
  ();
  const DataType sin_theta_sqrd = 1.0 - std::move(z_sqrd) / r_sqrd;

  // angular momentum per unit inertial mass in terms of r_max
  const double angular_momentum = [&a_sqrd, this ]() noexcept {
    const double sqrt_m = sqrt(black_hole_mass_);
    const double a_sqrt_m = black_hole_spin_ * sqrt_m;
    const double& rmax = max_pressure_radius_;
    const double sqrt_rmax = sqrt(rmax);
    const double rmax_sqrt_rmax = rmax * sqrt_rmax;
    const double rmax_sqrd = rmax * rmax;
    return sqrt_m * (rmax_sqrt_rmax + a_sqrt_m) *
           (a_sqrd - 2.0 * a_sqrt_m * sqrt_rmax + rmax_sqrd) /
           (2.0 * a_sqrt_m * rmax_sqrt_rmax +
            (rmax - 3.0 * black_hole_mass_) * rmax_sqrd);
  }
  ();

  const auto kerr_schild_metric = gr::Solutions::KerrSchild{
      black_hole_mass_,
      {{0.0, 0.0, black_hole_spin_}},
      {{0.0, 0.0,
        0.0}}}.variables(x, t, gr::Solutions::KerrSchild::tags<DataType>{});

  EquationsOfState::PolytropicFluid<true> polytrope(polytropic_constant_,
                                                    polytropic_exponent_);

  auto result = make_with_value<tuples::tagged_tuple_from_typelist<
      FishboneMoncriefDisk::variables_tags<DataType>>>(x, 0.0);

  get(get<hydro::Tags::SpecificEnthalpy<DataType>>(result)) = 1.0;

  const double inner_edge_potential =
      potential(square(inner_edge_radius_), 1.0, angular_momentum);

  // fill the disk with matter
  for (size_t s = 0; s < get_size(r_sqrd); ++s) {
    const double r_sqrd_s = get_element(r_sqrd, s);
    const double sin_theta_sqrd_s = get_element(sin_theta_sqrd, s);

    // the disk won't extend closer to the axis than r sin theta = rin
    // so no need to evaluate the potential there
    if (sqrt(r_sqrd_s * sin_theta_sqrd_s) >= inner_edge_radius_) {
      const double potential_s =
          potential(r_sqrd_s, sin_theta_sqrd_s, angular_momentum);

      // the fluid can only be where W(r, theta) < W_in
      if (potential_s < inner_edge_potential) {
        get_element(get(get<hydro::Tags::SpecificEnthalpy<DataType>>(result)),
                    s) = exp(inner_edge_potential - potential_s);

        get_element(get(get<hydro::Tags::RestMassDensity<DataType>>(result)),
                    s) =
            get(polytrope.rest_mass_density_from_enthalpy(
                Scalar<double>{get_element(
                    get(get<hydro::Tags::SpecificEnthalpy<DataType>>(result)),
                    s)}));

        get_element(get(get<hydro::Tags::Pressure<DataType>>(result)), s) =
            get(polytrope.pressure_from_density(Scalar<double>{get_element(
                get(get<hydro::Tags::RestMassDensity<DataType>>(result)), s)}));

        get_element(
            get(get<hydro::Tags::SpecificInternalEnergy<DataType>>(result)),
            s) =
            get(polytrope.specific_internal_energy_from_density(
                Scalar<double>{get_element(
                    get(get<hydro::Tags::RestMassDensity<DataType>>(result)),
                    s)}));

        const double angular_velocity_s =
            angular_velocity(r_sqrd_s, sin_theta_sqrd_s, angular_momentum);

        auto transport_velocity_s = make_array<3>(0.0);
        transport_velocity_s[0] -=
            angular_velocity_s * get_element(x.get(1), s);
        transport_velocity_s[1] +=
            angular_velocity_s * get_element(x.get(0), s);

        for (size_t i = 0; i < 3; ++i) {
          get_element(
              get<hydro::Tags::SpatialVelocity<DataType, 3>>(result).get(i),
              s) +=
              gsl::at(transport_velocity_s, i) +
              get_element(get<gr::Tags::Shift<3, Frame::Inertial, DataType>>(
                              kerr_schild_metric)
                              .get(i),
                          s);
        }
      }
    }
  }  // End fill the disk with matter

  const DataType one_over_lapse =
      1.0 / get(get<gr::Tags::Lapse<DataType>>(kerr_schild_metric));
  for (size_t i = 0; i < 3; ++i) {
    get<hydro::Tags::SpatialVelocity<DataType, 3>>(result).get(i) *=
        one_over_lapse;
  }

  return result;
}

template <typename DataType>
tuples::tagged_tuple_from_typelist<
    FishboneMoncriefDisk::dt_variables_tags<DataType>>
FishboneMoncriefDisk::dt_variables(
    const tnsr::I<DataType, 3>& x, const double /*t*/,
    FishboneMoncriefDisk::dt_variables_tags<DataType> /*meta*/) const noexcept {
  return make_with_value<tuples::tagged_tuple_from_typelist<
      FishboneMoncriefDisk::dt_variables_tags<DataType>>>(x, 0.0);
}

}  // namespace Solutions
}  // namespace RelativisticEuler

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template tuples::tagged_tuple_from_typelist<                               \
      RelativisticEuler::Solutions::FishboneMoncriefDisk::variables_tags<    \
          DTYPE(data)>>                                                      \
  RelativisticEuler::Solutions::FishboneMoncriefDisk::variables(             \
      const tnsr::I<DTYPE(data), 3>& x, const double t,                      \
      RelativisticEuler::Solutions::FishboneMoncriefDisk::variables_tags<    \
          DTYPE(data)> /*meta*/) const noexcept;                             \
  template tuples::tagged_tuple_from_typelist<                               \
      RelativisticEuler::Solutions::FishboneMoncriefDisk::dt_variables_tags< \
          DTYPE(data)>>                                                      \
  RelativisticEuler::Solutions::FishboneMoncriefDisk::dt_variables(          \
      const tnsr::I<DTYPE(data), 3>& x, const double /*t*/,                  \
      RelativisticEuler::Solutions::FishboneMoncriefDisk::dt_variables_tags< \
          DTYPE(data)> /*meta*/) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
/// \endcond
