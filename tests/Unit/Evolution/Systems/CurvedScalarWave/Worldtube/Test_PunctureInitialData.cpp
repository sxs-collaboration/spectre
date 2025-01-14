// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/InitialData/ZerothOrderPuncture.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/PunctureField.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeodesicAcceleration.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

namespace CurvedScalarWave {
namespace {

SPECTRE_TEST_CASE("Unit.Evolution.Systems.CurvedScalarWave.PunctureInitialData",
                  "[Unit][Evolution]") {
  const std::array<double, 3> pos_array{{12., 13., 0.}};
  const std::array<double, 3> vel_array{{0.1, 0.2, 0.}};
  const double charge = 0.23;

  const AnalyticData::ZerothOrderPuncture zeroth_order_puncture(
      pos_array, vel_array, charge);
  const auto copy = zeroth_order_puncture;
  CHECK(zeroth_order_puncture == copy);
  test_serialization(zeroth_order_puncture);

  const tnsr::I<double, 3> pos(pos_array);
  const tnsr::I<double, 3> vel(vel_array);

  const std::uniform_real_distribution sample_dist(-100., 100.);
  const size_t num_points = 1000;
  MAKE_GENERATOR(gen);

  const auto sample_coords =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), sample_dist, DataVector(num_points));

  const auto vars = zeroth_order_puncture.variables(sample_coords, {});
  auto sample_coords_centered = sample_coords;
  for (size_t i = 0; i < 3; ++i) {
    sample_coords_centered.get(i) -= pos.get(i);
  }

  // central black hole of mass 1
  const gr::Solutions::KerrSchild ks{
      1., {0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}};
  const auto background_vars_christoffel = ks.variables(
      pos, 0.,
      tmpl::list<gr::Tags::SpacetimeChristoffelSecondKind<double, 3,
                                                          Frame::Inertial>>{});
  const auto& christoffel =
      get<gr::Tags::SpacetimeChristoffelSecondKind<double, 3, Frame::Inertial>>(
          background_vars_christoffel);

  const auto background_vars =
      ks.variables(sample_coords, 0.,
                   tmpl::list<gr::Tags::Shift<DataVector, 3>,
                              gr::Tags::Lapse<DataVector>>{});
  const auto& shift = get<gr::Tags::Shift<DataVector, 3>>(background_vars);
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(background_vars);
  const auto geodesic_acceleration =
      gr::geodesic_acceleration(vel, christoffel);
  Variables<tmpl::list<CurvedScalarWave::Tags::Psi,
                       ::Tags::dt<CurvedScalarWave::Tags::Psi>,
                       ::Tags::deriv<CurvedScalarWave::Tags::Psi,
                                     tmpl::size_t<3>, Frame::Inertial>>>
      expected_puncture_field(num_points);

  Worldtube::puncture_field_0(make_not_null(&expected_puncture_field),
                              sample_coords_centered, pos, vel,
                              geodesic_acceleration, 1.);
  expected_puncture_field *= charge;

  const auto shift_dot_phi = dot_product(shift, get<Tags::Phi<3>>(vars));

  const auto dt_psi =
      get(shift_dot_phi) - get(lapse) * get(get<Tags::Pi>(vars));

  CHECK_ITERABLE_APPROX(get<Tags::Psi>(vars),
                        get<Tags::Psi>(expected_puncture_field));
  CHECK_ITERABLE_APPROX(
      dt_psi, get(get<::Tags::dt<Tags::Psi>>(expected_puncture_field)));
  const auto& di_psi =
      get<::Tags::deriv<Tags::Psi, tmpl::size_t<3>, Frame::Inertial>>(
          expected_puncture_field);
  for (size_t i = 0; i < 3; ++i) {
    CHECK_ITERABLE_APPROX(di_psi.get(i), get<Tags::Phi<3>>(vars).get(i));
  }
}
}  // namespace
}  // namespace CurvedScalarWave
