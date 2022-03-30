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
    const double polytropic_exponent,
    const gr::Solutions::TovCoordinates coordinate_system,
    const size_t pressure_exponent, const double cutoff_pressure_fraction,
    const double vector_potential_amplitude)
    : tov_star(central_rest_mass_density, polytropic_constant,
               polytropic_exponent, coordinate_system),
      pressure_exponent_(pressure_exponent),
      cutoff_pressure_(cutoff_pressure_fraction *
                       get(equation_of_state().pressure_from_density(
                           Scalar<double>{central_rest_mass_density}))),
      vector_potential_amplitude_(vector_potential_amplitude) {}

MagnetizedTovStar::MagnetizedTovStar(CkMigrateMessage* msg) : tov_star(msg) {}

void MagnetizedTovStar::pup(PUP::er& p) {
  tov_star::pup(p);
  p | pressure_exponent_;
  p | cutoff_pressure_;
  p | vector_potential_amplitude_;
}

namespace magnetized_tov_detail {
template <typename DataType, StarRegion Region>
void MagnetizedTovVariables<DataType, Region>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> magnetic_field,
    const gsl::not_null<Cache*> cache,
    hydro::Tags::MagneticField<DataType, 3> /*meta*/) const {
  const size_t num_pts = get_size(get<0>(coords));
  const auto& pressure_profile =
      get(cache->get_var(*this, hydro::Tags::Pressure<DataType>{}));
  const auto& dr_pressure_profile = get(cache->get_var(
      *this,
      RelativisticEuler::Solutions::tov_detail::Tags::DrPressure<DataType>{}));
  const auto& sqrt_det_spatial_metric =
      get(cache->get_var(*this, gr::Tags::SqrtDetSpatialMetric<DataType>{}));
  for (size_t i = 0; i < num_pts; ++i) {
    const double pressure = get_element(pressure_profile, i);
    if (LIKELY(get_element(radius, i) > 1.0e-16)) {
      if (pressure < cutoff_pressure) {
        get_element(get<0>(*magnetic_field), i) = 0.0;
        get_element(get<1>(*magnetic_field), i) = 0.0;
        get_element(get<2>(*magnetic_field), i) = 0.0;
        continue;
      }

      const double x = get_element(get<0>(coords), i);
      const double y = get_element(get<1>(coords), i);
      const double z = get_element(get<2>(coords), i);
      const double radius_i = get_element(radius, i);
      const double dr_pressure = get_element(dr_pressure_profile, i);
      const double pressure_term =
          pow(pressure - cutoff_pressure, pressure_exponent);
      const double deriv_pressure_term =
          pressure_exponent *
          pow(pressure - cutoff_pressure,
              static_cast<int>(pressure_exponent) - 1) *
          dr_pressure;

      get_element(get<0>(*magnetic_field), i) =
          x * z / radius_i * deriv_pressure_term;

      get_element(get<1>(*magnetic_field), i) =
          y * z / radius_i * deriv_pressure_term;

      get_element(get<2>(*magnetic_field), i) =
          (-2.0 * pressure_term +
           (square(x) + square(y)) / radius_i * deriv_pressure_term);
    } else {
      get_element(get<0>(*magnetic_field), i) = 0.0;
      get_element(get<1>(*magnetic_field), i) = 0.0;
      get_element(get<2>(*magnetic_field), i) =
          (-2.0 * pow(pressure - cutoff_pressure, pressure_exponent));
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    magnetic_field->get(i) *=
        vector_potential_amplitude / sqrt_det_spatial_metric;
  }
}
}  // namespace magnetized_tov_detail

PUP::able::PUP_ID MagnetizedTovStar::my_PUP_ID = 0;

bool operator==(const MagnetizedTovStar& lhs, const MagnetizedTovStar& rhs) {
  return static_cast<const typename MagnetizedTovStar::tov_star&>(lhs) ==
             static_cast<const typename MagnetizedTovStar::tov_star&>(rhs) and
         lhs.pressure_exponent_ == rhs.pressure_exponent_ and
         lhs.cutoff_pressure_ == rhs.cutoff_pressure_ and
         lhs.vector_potential_amplitude_ == rhs.vector_potential_amplitude_;
}

bool operator!=(const MagnetizedTovStar& lhs, const MagnetizedTovStar& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define REGION(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                \
  template class magnetized_tov_detail::MagnetizedTovVariables<DTYPE(data), \
                                                               REGION(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector),
                        (magnetized_tov_detail::StarRegion::Center,
                         magnetized_tov_detail::StarRegion::Interior,
                         magnetized_tov_detail::StarRegion::Exterior))

#undef INSTANTIATE
#undef DTYPE
#undef REGION
}  // namespace grmhd::AnalyticData
