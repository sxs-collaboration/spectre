// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Worldtube/InitialData/ZerothOrderPuncture.hpp"

#include <cstddef>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/PunctureField.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeodesicAcceleration.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace CurvedScalarWave::AnalyticData {

ZerothOrderPuncture::ZerothOrderPuncture(
    const std::array<double, 3> particle_position,
    const std::array<double, 3> particle_velocity, const double particle_charge,
    const Options::Context& /*context*/)
    : particle_position_(particle_position),
      particle_velocity_(particle_velocity),
      particle_charge_(particle_charge) {
  const auto background_vars = kerr_schild_.variables(
      particle_position_, 0.,
      tmpl::list<gr::Tags::SpacetimeChristoffelSecondKind<double, 3,
                                                          Frame::Inertial>>{});
  const auto& christoffel =
      get<gr::Tags::SpacetimeChristoffelSecondKind<double, 3, Frame::Inertial>>(
          background_vars);
  geodesic_acceleration_ =
      gr::geodesic_acceleration(particle_velocity_, christoffel);
}

tuples::TaggedTuple<CurvedScalarWave::Tags::Psi, CurvedScalarWave::Tags::Pi,
                    CurvedScalarWave::Tags::Phi<3>>
ZerothOrderPuncture::variables(const tnsr::I<DataVector, 3>& x,
                               tags /*meta*/) const {
  auto centered_coords = x;
  for (size_t i = 0; i < 3; ++i) {
    centered_coords.get(i) -= particle_position_.get(i);
  }
  Variables<tmpl::list<CurvedScalarWave::Tags::Psi,
                       ::Tags::dt<CurvedScalarWave::Tags::Psi>,
                       ::Tags::deriv<CurvedScalarWave::Tags::Psi,
                                     tmpl::size_t<3>, Frame::Inertial>>>
      puncture(get<0>(centered_coords).size());
  CurvedScalarWave::Worldtube::puncture_field_0(
      make_not_null(&puncture), centered_coords, particle_position_,
      particle_velocity_, geodesic_acceleration_, 1.);
  puncture *= particle_charge_;
  const auto background_vars =
      kerr_schild_.variables(x, 0.,
                             tmpl::list<gr::Tags::Shift<DataVector, 3>,
                                        gr::Tags::Lapse<DataVector>>{});
  const auto& shift = get<gr::Tags::Shift<DataVector, 3>>(background_vars);
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(background_vars);
  const auto shift_dot_dpsi = dot_product(
      shift, get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                               Frame::Inertial>>(puncture));

  return tuples::TaggedTuple<CurvedScalarWave::Tags::Psi,
                             CurvedScalarWave::Tags::Pi,
                             CurvedScalarWave::Tags::Phi<3>>{
      get<CurvedScalarWave::Tags::Psi>(puncture),
      (get(shift_dot_dpsi) -
       get(get<::Tags::dt<CurvedScalarWave::Tags::Psi>>(puncture))) /
          get(lapse),
      get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                        Frame::Inertial>>(puncture)};
}

void ZerothOrderPuncture::pup(PUP::er& p) {
  p | particle_position_;
  p | particle_velocity_;
  p | geodesic_acceleration_;
  p | particle_charge_;
  p | kerr_schild_;
}

bool operator==(const ZerothOrderPuncture& lhs,
                const ZerothOrderPuncture& rhs) {
  return lhs.particle_position_ == rhs.particle_position_ and
         lhs.particle_velocity_ == rhs.particle_velocity_ and
         lhs.geodesic_acceleration_ == rhs.geodesic_acceleration_ and
         lhs.particle_charge_ == rhs.particle_charge_ and
         lhs.kerr_schild_ == rhs.kerr_schild_;
}
bool operator!=(const ZerothOrderPuncture& lhs,
                const ZerothOrderPuncture& rhs) {
  return not(lhs == rhs);
}
}  // namespace CurvedScalarWave::AnalyticData
