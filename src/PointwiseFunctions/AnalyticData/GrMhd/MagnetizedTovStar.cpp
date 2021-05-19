// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GrMhd/MagnetizedTovStar.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace grmhd::AnalyticData {
MagnetizedTovStar::MagnetizedTovStar(
    const double central_rest_mass_density, const double polytropic_constant,
    const double polytropic_exponent, const size_t pressure_exponent,
    const double cutoff_pressure_fraction,
    const double vector_potential_amplitude) noexcept
    : tov_star(central_rest_mass_density, polytropic_constant,
               polytropic_exponent),
      pressure_exponent_(pressure_exponent),
      cutoff_pressure_(cutoff_pressure_fraction *
                       get(equation_of_state().pressure_from_density(
                           Scalar<double>{central_rest_mass_density}))),
      vector_potential_amplitude_(vector_potential_amplitude) {}

void MagnetizedTovStar::pup(PUP::er& p) noexcept {
  tov_star::pup(p);
  p | pressure_exponent_;
  p | cutoff_pressure_;
  p | vector_potential_amplitude_;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>
MagnetizedTovStar::variables(
    const tnsr::I<DataType, 3>& coords,
    tmpl::list<
        hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>> /*meta*/,
    const RadialVariables<DataType>& radial_vars) const noexcept {
  const size_t num_pts = get_size(get<0>(coords));
  auto magnetic_field =
      make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(num_pts, 0.0);
  using std::max;
  const auto sqrt_det_spatial_metric =
      get<gr::Tags::SqrtDetSpatialMetric<DataType>>(variables(
          coords, tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataType>>{},
          radial_vars));
  for (size_t i = 0; i < num_pts; ++i) {
    const double pressure = get_element(get(radial_vars.pressure), i);
    if (LIKELY(get_element(radial_vars.radial_coordinate, i) > 1.0e-16)) {
      if (pressure < cutoff_pressure_) {
        continue;
      }

      const double x = get_element(get<0>(coords), i);
      const double y = get_element(get<1>(coords), i);
      const double z = get_element(get<2>(coords), i);
      const double radius = get_element(radial_vars.radial_coordinate, i);
      const double dr_pressure = get_element(radial_vars.dr_pressure, i);
      const double pressure_term =
          pow(pressure - cutoff_pressure_, pressure_exponent_);
      const double deriv_pressure_term =
          pressure_exponent_ *
          pow(pressure - cutoff_pressure_,
              static_cast<int>(pressure_exponent_) - 1) *
          dr_pressure;

      get_element(get<0>(magnetic_field), i) =
          x * z / radius * deriv_pressure_term;

      get_element(get<1>(magnetic_field), i) =
          y * z / radius * deriv_pressure_term;

      get_element(get<2>(magnetic_field), i) =
          (-2.0 * pressure_term +
           (square(x) + square(y)) / radius * deriv_pressure_term);
    } else {
      get_element(get<2>(magnetic_field), i) =
          (-2.0 * pow(pressure - cutoff_pressure_, pressure_exponent_));
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    magnetic_field.get(i) *=
        vector_potential_amplitude_ / get(sqrt_det_spatial_metric);
  }
  return magnetic_field;
}

bool operator==(const MagnetizedTovStar& lhs,
                const MagnetizedTovStar& rhs) noexcept {
  return static_cast<const typename MagnetizedTovStar::tov_star&>(lhs) ==
             static_cast<const typename MagnetizedTovStar::tov_star&>(rhs) and
         lhs.pressure_exponent_ == rhs.pressure_exponent_ and
         lhs.cutoff_pressure_ == rhs.cutoff_pressure_ and
         lhs.vector_potential_amplitude_ == rhs.vector_potential_amplitude_;
}

bool operator!=(const MagnetizedTovStar& lhs,
                const MagnetizedTovStar& rhs) noexcept {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                            \
  template tuples::TaggedTuple<                                         \
      hydro::Tags::MagneticField<DTYPE(data), 3, Frame::Inertial>>      \
  MagnetizedTovStar::variables(                                         \
      const tnsr::I<DTYPE(data), 3>& coords,                            \
      tmpl::list<hydro::Tags::MagneticField<DTYPE(data), 3,             \
                                            Frame::Inertial>> /*meta*/, \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef INSTANTIATE
#undef DTYPE
}  // namespace grmhd::AnalyticData
