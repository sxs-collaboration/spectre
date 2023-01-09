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
#include "PointwiseFunctions/AnalyticData/GrMhd/InitialMagneticFields/Poloidal.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/InitialMagneticFields/Toroidal.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace grmhd::AnalyticData {
MagnetizedTovStar::MagnetizedTovStar(
    const double central_rest_mass_density,
    std::unique_ptr<MagnetizedTovStar::equation_of_state_type>
        equation_of_state,
    const RelativisticEuler::Solutions::TovCoordinates coordinate_system,
    const size_t poloidal_pressure_exponent,
    const double poloidal_cutoff_pressure,
    const double poloidal_vector_potential_amplitude,
    const size_t toroidal_pressure_exponent,
    const double toroidal_cutoff_pressure,
    const double toroidal_vector_potential_amplitude)
    : tov_star(central_rest_mass_density, std::move(equation_of_state),
               coordinate_system),
      poloidal_pressure_exponent_(poloidal_pressure_exponent),
      poloidal_cutoff_pressure_(
          poloidal_cutoff_pressure *
          get(this->equation_of_state().pressure_from_density(
              Scalar<double>{central_rest_mass_density}))),
      poloidal_vector_potential_amplitude_(poloidal_vector_potential_amplitude),
      toroidal_pressure_exponent_(toroidal_pressure_exponent),
      toroidal_cutoff_pressure_(
          toroidal_cutoff_pressure *
          get(this->equation_of_state().pressure_from_density(
              Scalar<double>{central_rest_mass_density}))),
      toroidal_vector_potential_amplitude_(
          toroidal_vector_potential_amplitude) {}

std::unique_ptr<evolution::initial_data::InitialData>
MagnetizedTovStar::get_clone() const {
  return std::make_unique<MagnetizedTovStar>(*this);
}

MagnetizedTovStar::MagnetizedTovStar(CkMigrateMessage* msg) : tov_star(msg) {}

void MagnetizedTovStar::pup(PUP::er& p) {
  tov_star::pup(p);
  p | poloidal_pressure_exponent_;
  p | poloidal_cutoff_pressure_;
  p | poloidal_vector_potential_amplitude_;
  p | toroidal_pressure_exponent_;
  p | toroidal_cutoff_pressure_;
  p | toroidal_vector_potential_amplitude_;
}

namespace magnetized_tov_detail {
template <typename DataType, StarRegion Region>
void MagnetizedTovVariables<DataType, Region>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> magnetic_field,
    const gsl::not_null<Cache*> cache,
    hydro::Tags::MagneticField<DataType, 3> /*meta*/) const {
  const auto& pressure =
      cache->get_var(*this, hydro::Tags::Pressure<DataType>{});
  const auto& deriv_pressure =
      cache->get_var(*this, ::Tags::deriv<hydro::Tags::Pressure<DataType>,
                                          tmpl::size_t<3>, Frame::Inertial>{});
  const auto& sqrt_det_spatial_metric =
      cache->get_var(*this, gr::Tags::SqrtDetSpatialMetric<DataType>{});

  const auto poloidal_field = get<hydro::Tags::MagneticField<DataType, 3>>(
      InitialMagneticFields::Poloidal(poloidal_pressure_exponent,
                                      poloidal_cutoff_pressure,
                                      poloidal_vector_potential_amplitude)
          .variables(coords, pressure, sqrt_det_spatial_metric,
                     deriv_pressure));

  const auto toroidal_field = get<hydro::Tags::MagneticField<DataType, 3>>(
      InitialMagneticFields::Toroidal(toroidal_pressure_exponent,
                                      toroidal_cutoff_pressure,
                                      toroidal_vector_potential_amplitude)
          .variables(coords, pressure, sqrt_det_spatial_metric,
                     deriv_pressure));

  for (size_t i = 0; i < 3; ++i) {
    (*magnetic_field).get(i) = poloidal_field.get(i) + toroidal_field.get(i);
  }
}
}  // namespace magnetized_tov_detail

PUP::able::PUP_ID MagnetizedTovStar::my_PUP_ID = 0;

bool operator==(const MagnetizedTovStar& lhs, const MagnetizedTovStar& rhs) {
  return static_cast<const typename MagnetizedTovStar::tov_star&>(lhs) ==
             static_cast<const typename MagnetizedTovStar::tov_star&>(rhs) and
         lhs.poloidal_pressure_exponent_ == rhs.poloidal_pressure_exponent_ and
         lhs.poloidal_cutoff_pressure_ == rhs.poloidal_cutoff_pressure_ and
         lhs.poloidal_vector_potential_amplitude_ ==
             rhs.poloidal_vector_potential_amplitude_ and
         lhs.toroidal_pressure_exponent_ == rhs.toroidal_pressure_exponent_ and
         lhs.toroidal_cutoff_pressure_ == rhs.toroidal_cutoff_pressure_ and
         lhs.toroidal_vector_potential_amplitude_ ==
             rhs.toroidal_vector_potential_amplitude_;
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
