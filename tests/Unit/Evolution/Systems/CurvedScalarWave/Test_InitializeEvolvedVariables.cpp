// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <limits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Initialize.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/ScalarWave/System.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/AnalyticData/CurvedWaveEquation/PureSphericalHarmonic.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphericalKerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/RegularSphericalWave.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"

namespace {
template <typename BackgroundSpacetime, typename InitialData>
void test_initialize_evolved_variables(
    const gsl::not_null<std::mt19937*> generator,
    const BackgroundSpacetime& background_spacetime,
    const InitialData& initial_data) {
  static_assert(BackgroundSpacetime::volume_dim == InitialData::volume_dim,
                "The dimensions of the background spacetime and the initial "
                "data do not match.");
  static constexpr size_t Dim = BackgroundSpacetime::volume_dim;
  using evolved_var_tag = typename CurvedScalarWave::System<Dim>::variables_tag;
  const size_t grid_size = 100;
  std::uniform_real_distribution dist{-10., 10.};
  const auto random_coords = make_with_random_values<tnsr::I<DataVector, Dim>>(
      generator, make_not_null(&dist), DataVector{grid_size});
  const double initial_time = 0.;

  const auto lapse_and_shift = background_spacetime.variables(
      random_coords, initial_time,
      tmpl::list<gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<DataVector, Dim>>{});
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(lapse_and_shift);
  const auto& shift = get<gr::Tags::Shift<DataVector, Dim>>(lapse_and_shift);

  auto box = db::create<db::AddSimpleTags<
      ::Tags::Time, domain::Tags::Coordinates<Dim, Frame::Inertial>,
      Tags::AnalyticData<InitialData>, gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<DataVector, Dim>, evolved_var_tag>>(
      initial_time, random_coords, initial_data, lapse, shift,
      typename evolved_var_tag::type{grid_size});
  db::mutate_apply<
      CurvedScalarWave::Initialization::InitializeEvolvedVariables<Dim>>(
      make_not_null(&box));

  if constexpr (tmpl::list_contains_v<typename InitialData::tags,
                                      CurvedScalarWave::Tags::Psi>) {
    // it is a curved solution, no adjustment
    const auto unadjusted_initial_data =
        variables_from_tagged_tuple(evolution::Initialization::initial_data(
            initial_data, random_coords, initial_time,
            typename evolved_var_tag::tags_list{}));
    CHECK_VARIABLES_APPROX(db::get<evolved_var_tag>(box),
                           unadjusted_initial_data);
  } else {
    // it is a flat solution, need to adjust
    const auto unadjusted_initial_data =
        evolution::Initialization::initial_data(
            initial_data, random_coords, initial_time,
            typename ScalarWave::System<Dim>::variables_tag::tags_list{});
    const auto& unadjusted_psi =
        get<ScalarWave::Tags::Psi>(unadjusted_initial_data);
    const auto& unadjusted_pi =
        get<ScalarWave::Tags::Pi>(unadjusted_initial_data);
    const auto& unadjusted_phi =
        get<ScalarWave::Tags::Phi<Dim>>(unadjusted_initial_data);
    Scalar<DataVector> adjusted_pi(grid_size);
    const auto shift_dot_dpsi = dot_product(shift, unadjusted_phi);
    get(adjusted_pi) = (get(shift_dot_dpsi) + get(unadjusted_pi)) / get(lapse);

    CHECK_ITERABLE_APPROX(unadjusted_psi,
                          db::get<CurvedScalarWave::Tags::Psi>(box));
    CHECK_ITERABLE_APPROX(unadjusted_phi,
                          db::get<CurvedScalarWave::Tags::Phi<Dim>>(box));
    CHECK_ITERABLE_APPROX(adjusted_pi,
                          db::get<CurvedScalarWave::Tags::Pi>(box));
    if constexpr (std::is_same_v<BackgroundSpacetime,
                                 gr::Solutions::Minkowski<Dim>>) {
      CHECK_ITERABLE_APPROX(adjusted_pi, unadjusted_pi);
    }
  }
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.CurvedScalarWave.InitializeEvolvedVariables",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);

  const ScalarWave::Solutions::PlaneWave<1> plane_wave_1d{
      {{1.0}},
      {{0.0}},
      std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(11., 1.4,
                                                                    0.3)};
  const ScalarWave::Solutions::PlaneWave<2> plane_wave_2d{
      {{1.0, 1.0}},
      {{0.0, 0.0}},
      std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(11., 1.4,
                                                                    0.3)};
  const ScalarWave::Solutions::PlaneWave<3> plane_wave_3d{
      {{1.0, 1.0, 1.0}},
      {{0.0, 0.0, 0.0}},
      std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(11., 1.4,
                                                                    0.3)};
  const ScalarWave::Solutions::RegularSphericalWave regular_spherical_wave{
      std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(11., 1.4,
                                                                    0.3)};
  const CurvedScalarWave::AnalyticData::PureSphericalHarmonic
      pure_spherical_harmonic{10., 1., {4, -3}};

  const gr::Solutions::Minkowski<1> minkowski_1d{};
  const gr::Solutions::Minkowski<2> minkowski_2d{};
  const gr::Solutions::Minkowski<3> minkowski_3d{};
  const gr::Solutions::KerrSchild kerr_schild(0.7, {{0.1, -0.2, 0.3}},
                                              {{0.7, 0.6, -2.}});
  const gr::Solutions::SphericalKerrSchild spherical_kerr_schild(
      0.7, {{0.1, -0.2, 0.3}}, {{0.7, 0.6, -2.}});

  test_initialize_evolved_variables(make_not_null(&generator), minkowski_1d,
                                    plane_wave_1d);
  test_initialize_evolved_variables(make_not_null(&generator), minkowski_2d,
                                    plane_wave_2d);
  test_initialize_evolved_variables(make_not_null(&generator), minkowski_3d,
                                    plane_wave_3d);
  test_initialize_evolved_variables(make_not_null(&generator), minkowski_3d,
                                    regular_spherical_wave);
  test_initialize_evolved_variables(make_not_null(&generator), minkowski_3d,
                                    pure_spherical_harmonic);

  test_initialize_evolved_variables(make_not_null(&generator), kerr_schild,
                                    plane_wave_3d);
  test_initialize_evolved_variables(make_not_null(&generator), kerr_schild,
                                    regular_spherical_wave);
  test_initialize_evolved_variables(make_not_null(&generator), kerr_schild,
                                    pure_spherical_harmonic);

  test_initialize_evolved_variables(make_not_null(&generator),
                                    spherical_kerr_schild, plane_wave_3d);
  test_initialize_evolved_variables(
      make_not_null(&generator), spherical_kerr_schild, regular_spherical_wave);
  test_initialize_evolved_variables(make_not_null(&generator),
                                    spherical_kerr_schild,
                                    pure_spherical_harmonic);
}
}  // namespace
