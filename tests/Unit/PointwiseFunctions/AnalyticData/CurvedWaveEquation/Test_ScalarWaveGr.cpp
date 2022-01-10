// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <memory>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarWave/System.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticData/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticData/CurvedWaveEquation/ScalarWaveGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/RegularSphericalWave.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<MathFunction<1, Frame::Inertial>,
                   tmpl::list<MathFunctions::Gaussian<1, Frame::Inertial>>>>;
  };
};

template <typename ScalarFieldData, typename BackgroundData>
void test_tag_retrieval(const CurvedScalarWave::AnalyticData::ScalarWaveGr<
                        ScalarFieldData, BackgroundData>& curved_wave_data) {
  const tnsr::I<DataVector, 3> x{std::array<DataVector, 3>{
      {DataVector({0., 1., 2., 3.}), DataVector({0., 0., 0., 0.}),
       DataVector({0., 0., 0., 0.})}}};

  // Evaluate solution
  TestHelpers::AnalyticData::test_tag_retrieval(
      curved_wave_data, x,
      typename CurvedScalarWave::AnalyticData::ScalarWaveGr<
          ScalarFieldData, BackgroundData>::tags());
}

template <typename ScalarFieldData, typename BackgroundData>
void test_no_hole(const CurvedScalarWave::AnalyticData::ScalarWaveGr<
                      ScalarFieldData, BackgroundData>& curved_wave_data,
                  const ScalarFieldData& flat_wave_solution) {
  const tnsr::I<DataVector, 3> x{std::array<DataVector, 3>{
      {DataVector({0., 1., 2., 3.}), DataVector({0., 0., 0., 0.}),
       DataVector({0., 0., 0., 0.})}}};
  auto vars = curved_wave_data.variables(
      x, tmpl::list<CurvedScalarWave::Tags::Psi, CurvedScalarWave::Tags::Pi,
                    CurvedScalarWave::Tags::Phi<3>>{});
  auto flat_vars = flat_wave_solution.variables(
      x, 0.,
      tmpl::list<ScalarWave::Tags::Psi, ScalarWave::Tags::Pi,
                 ScalarWave::Tags::Phi<3>>{});
  CHECK_ITERABLE_APPROX(get(get<CurvedScalarWave::Tags::Psi>(vars)),
                        get(get<ScalarWave::Tags::Psi>(flat_vars)));
  CHECK_ITERABLE_APPROX(get(get<CurvedScalarWave::Tags::Pi>(vars)),
                        get(get<ScalarWave::Tags::Pi>(flat_vars)));
  CHECK_ITERABLE_APPROX(get<0>(get<CurvedScalarWave::Tags::Phi<3>>(vars)),
                        get<0>(get<ScalarWave::Tags::Phi<3>>(flat_vars)));
  CHECK_ITERABLE_APPROX(get<1>(get<CurvedScalarWave::Tags::Phi<3>>(vars)),
                        get<1>(get<ScalarWave::Tags::Phi<3>>(flat_vars)));
  CHECK_ITERABLE_APPROX(get<2>(get<CurvedScalarWave::Tags::Phi<3>>(vars)),
                        get<2>(get<ScalarWave::Tags::Phi<3>>(flat_vars)));
}

template <typename ScalarFieldData>
void test_kerr(
    const CurvedScalarWave::AnalyticData::ScalarWaveGr<
        ScalarFieldData, gr::Solutions::KerrSchild>& curved_wave_data,
    const gr::Solutions::KerrSchild& bh_solution,
    const ScalarFieldData& flat_wave_solution) {
  const tnsr::I<DataVector, 3> x{std::array<DataVector, 3>{
      {DataVector({0., 1., 2., 3.}), DataVector({0., 0., 0., 0.}),
       DataVector({0., 0., 0., 0.})}}};
  auto vars = curved_wave_data.variables(
      x, tmpl::list<CurvedScalarWave::Tags::Psi, CurvedScalarWave::Tags::Pi,
                    CurvedScalarWave::Tags::Phi<3>>{});

  const auto flat_wave_vars = flat_wave_solution.variables(
      x, 0., ScalarWave::System<3>::variables_tag::tags_list{});

  const auto kerr_variables = bh_solution.variables(
      x, 0., gr::Solutions::KerrSchild::tags<DataVector>{});

  // construct the expected vars in-situ
  const auto& local_psi = get<ScalarWave::Tags::Psi>(flat_wave_vars);
  const auto& local_phi = get<ScalarWave::Tags::Phi<3>>(flat_wave_vars);
  auto local_pi = make_with_value<Scalar<DataVector>>(x, 0.);
  {
    const auto shift_dot_dpsi = dot_product(
        get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(kerr_variables),
        get<ScalarWave::Tags::Phi<3>>(flat_wave_vars));
    get(local_pi) =
        (get(shift_dot_dpsi) + get(get<ScalarWave::Tags::Pi>(flat_wave_vars))) /
        get(get<gr::Tags::Lapse<DataVector>>(kerr_variables));
  }

  CHECK_ITERABLE_APPROX(get(get<CurvedScalarWave::Tags::Psi>(vars)),
                        get(local_psi));
  CHECK_ITERABLE_APPROX(get(get<CurvedScalarWave::Tags::Pi>(vars)),
                        get(local_pi));
  CHECK_ITERABLE_APPROX(get<0>(get<CurvedScalarWave::Tags::Phi<3>>(vars)),
                        get<0>(local_phi));
  CHECK_ITERABLE_APPROX(get<1>(get<CurvedScalarWave::Tags::Phi<3>>(vars)),
                        get<1>(local_phi));
  CHECK_ITERABLE_APPROX(get<2>(get<CurvedScalarWave::Tags::Phi<3>>(vars)),
                        get<2>(local_phi));
}

template <typename ScalarFieldData>
void test_kerr_schild_vars(
    const CurvedScalarWave::AnalyticData::ScalarWaveGr<
        ScalarFieldData, gr::Solutions::KerrSchild>& curved_wave_data,
    const gr::Solutions::KerrSchild& bh_solution) {
  const tnsr::I<DataVector, 3> x{std::array<DataVector, 3>{
      {DataVector({0., 1., 2., 3.}), DataVector({0., 0., 0., 0.}),
       DataVector({0., 0., 0., 0.})}}};

  tuples::tagged_tuple_from_typelist<
      typename gr::Solutions::KerrSchild::tags<DataVector>>
      vars = curved_wave_data.variables(
          x, typename gr::Solutions::KerrSchild::tags<DataVector>{});

  // Now get Kerr background vars directly
  const auto kerr_variables = bh_solution.variables(
      x, 0., gr::Solutions::KerrSchild::tags<DataVector>{});

  tmpl::for_each<gr::Solutions::KerrSchild::tags<DataVector>>(
      [&vars, &kerr_variables](auto x_) {
        using tag = typename decltype(x_)::type;
        CHECK_ITERABLE_APPROX(get<tag>(vars), get<tag>(kerr_variables));
      });
}

void test_construct_from_options() {
  {  // test with wave profile = spherical
    const auto curved_wave_data_constructed_from_opts =
        TestHelpers::test_creation<
            CurvedScalarWave::AnalyticData::ScalarWaveGr<
                ScalarWave::Solutions::RegularSphericalWave,
                gr::Solutions::KerrSchild>,
            Metavariables>(
            "Background:\n"
            "  Mass: 0.7\n"
            "  Spin: [0.1, -0.2, 0.3]\n"
            "  Center: [0.7,0.6,-2.]\n"
            "ScalarField:\n"
            "  Profile:\n"
            "    Gaussian:\n"
            "      Amplitude: 11.\n"
            "      Width: 1.4\n"
            "      Center: 0.3");

    // construct internals of data constructed from options
    const gr::Solutions::KerrSchild bh_solution(0.7, {{0.1, -0.2, 0.3}},
                                                {{0.7, 0.6, -2.}});
    const ScalarWave::Solutions::RegularSphericalWave flat_wave_solution{
        std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(11., 1.4,
                                                                      0.3)};

    CHECK(curved_wave_data_constructed_from_opts ==
          CurvedScalarWave::AnalyticData::ScalarWaveGr<
              ScalarWave::Solutions::RegularSphericalWave,
              gr::Solutions::KerrSchild>(
              bh_solution,
              // cannot use `flat_wave_solution` in construction here
              // as it has a `std::unique_ptr` member
              ScalarWave::Solutions::RegularSphericalWave(
                  std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(
                      11., 1.4, 0.3))));

    test_kerr(curved_wave_data_constructed_from_opts, bh_solution,
              flat_wave_solution);
  }
  {  // test with wave profile = plane
    const auto curved_wave_data_constructed_from_opts =
        TestHelpers::test_creation<
            CurvedScalarWave::AnalyticData::ScalarWaveGr<
                ScalarWave::Solutions::PlaneWave<3>, gr::Solutions::KerrSchild>,
            Metavariables>(
            "Background:\n"
            "  Mass: 0.7\n"
            "  Spin: [0.1, -0.2, 0.3]\n"
            "  Center: [0.7,0.6,-2.]\n"
            "ScalarField:\n"
            "  WaveVector: [1.0, 1.0, 1.0]\n"
            "  Center: [0.0, 0.0, 0.0]\n"
            "  Profile:\n"
            "    Gaussian:\n"
            "      Amplitude: 11.\n"
            "      Width: 1.4\n"
            "      Center: 0.3");

    // construct internals of data constructed from options
    const gr::Solutions::KerrSchild bh_solution(0.7, {{0.1, -0.2, 0.3}},
                                                {{0.7, 0.6, -2.}});
    const ScalarWave::Solutions::PlaneWave<3> flat_wave_solution{
        {{1.0, 1.0, 1.0}},
        {{0.0, 0.0, 0.0}},
        std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(11., 1.4,
                                                                      0.3)};

    CHECK(curved_wave_data_constructed_from_opts ==
          CurvedScalarWave::AnalyticData::ScalarWaveGr<
              ScalarWave::Solutions::PlaneWave<3>, gr::Solutions::KerrSchild>(
              bh_solution,
              // cannot use `flat_wave_solution` in construction here
              // as it has a `std::unique_ptr` member
              ScalarWave::Solutions::PlaneWave<3>(
                  {{1.0, 1.0, 1.0}}, {{0.0, 0.0, 0.0}},
                  std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(
                      11., 1.4, 0.3))));

    test_kerr(curved_wave_data_constructed_from_opts, bh_solution,
              flat_wave_solution);
  }
  {  // test flat spacetime with wave profile = plane in 3D
    const auto curved_wave_data_constructed_from_opts =
        TestHelpers::test_creation<CurvedScalarWave::AnalyticData::ScalarWaveGr<
                                       ScalarWave::Solutions::PlaneWave<3>,
                                       gr::Solutions::Minkowski<3>>,
                                   Metavariables>(
            "Background:\n"
            "ScalarField:\n"
            "  WaveVector: [1.0, 1.0, 1.0]\n"
            "  Center: [0.0, 0.0, 0.0]\n"
            "  Profile:\n"
            "    Gaussian:\n"
            "      Amplitude: 11.\n"
            "      Width: 1.4\n"
            "      Center: 0.3");

    // construct internals of data constructed from options
    const gr::Solutions::Minkowski<3> flat_space_solution{};

    CHECK(curved_wave_data_constructed_from_opts ==
          CurvedScalarWave::AnalyticData::ScalarWaveGr<
              ScalarWave::Solutions::PlaneWave<3>, gr::Solutions::Minkowski<3>>(
              flat_space_solution,
              // cannot use a `flat_wave_solution` object in construction here
              // as it has a `std::unique_ptr` member
              ScalarWave::Solutions::PlaneWave<3>(
                  {{1.0, 1.0, 1.0}}, {{0.0, 0.0, 0.0}},
                  std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(
                      11., 1.4, 0.3))));
  }
  {  // test flat spacetime with wave profile = plane in 2D
    const auto curved_wave_data_constructed_from_opts =
        TestHelpers::test_creation<CurvedScalarWave::AnalyticData::ScalarWaveGr<
                                       ScalarWave::Solutions::PlaneWave<2>,
                                       gr::Solutions::Minkowski<2>>,
                                   Metavariables>(
            "Background:\n"
            "ScalarField:\n"
            "  WaveVector: [1.0, 1.0]\n"
            "  Center: [0.0, 0.0]\n"
            "  Profile:\n"
            "    Gaussian:\n"
            "      Amplitude: 11.\n"
            "      Width: 1.4\n"
            "      Center: 0.3");

    // construct internals of data constructed from options
    const gr::Solutions::Minkowski<2> flat_space_solution{};

    CHECK(curved_wave_data_constructed_from_opts ==
          CurvedScalarWave::AnalyticData::ScalarWaveGr<
              ScalarWave::Solutions::PlaneWave<2>, gr::Solutions::Minkowski<2>>(
              flat_space_solution,
              // cannot use a `flat_wave_solution` object in construction here
              // as it has a `std::unique_ptr` member
              ScalarWave::Solutions::PlaneWave<2>(
                  {{1.0, 1.0}}, {{0.0, 0.0}},
                  std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(
                      11., 1.4, 0.3))));
  }
  {  // test flat spacetime with wave profile = plane in 1D
    const auto curved_wave_data_constructed_from_opts =
        TestHelpers::test_creation<CurvedScalarWave::AnalyticData::ScalarWaveGr<
                                       ScalarWave::Solutions::PlaneWave<1>,
                                       gr::Solutions::Minkowski<1>>,
                                   Metavariables>(
            "Background:\n"
            "ScalarField:\n"
            "  WaveVector: [1.0]\n"
            "  Center: [0.0]\n"
            "  Profile:\n"
            "    Gaussian:\n"
            "      Amplitude: 11.\n"
            "      Width: 1.4\n"
            "      Center: 0.3");

    // construct internals of data constructed from options
    const gr::Solutions::Minkowski<1> flat_space_solution{};

    CHECK(curved_wave_data_constructed_from_opts ==
          CurvedScalarWave::AnalyticData::ScalarWaveGr<
              ScalarWave::Solutions::PlaneWave<1>, gr::Solutions::Minkowski<1>>(
              flat_space_solution,
              // cannot use a `flat_wave_solution` object in construction here
              // as it has a `std::unique_ptr` member
              ScalarWave::Solutions::PlaneWave<1>(
                  {{1.0}}, {{0.0}},
                  std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(
                      11., 1.4, 0.3))));
  }
}

void test_serialize(const double mass, const std::array<double, 3>& spin,
                    const std::array<double, 3>& center) {
  Parallel::register_factory_classes_with_charm<Metavariables>();
  {  // test with wave profile = spherical
    CurvedScalarWave::AnalyticData::ScalarWaveGr<
        ScalarWave::Solutions::RegularSphericalWave, gr::Solutions::KerrSchild>
        curved_wave_data(
            gr::Solutions::KerrSchild(mass, spin, center),
            ScalarWave::Solutions::RegularSphericalWave(
                std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(
                    11., 1.4, 0.3)));
    const auto snd_curved_wave_data =
        serialize_and_deserialize(curved_wave_data);
    CHECK(curved_wave_data == snd_curved_wave_data);

    // test internals of deserialized data
    const gr::Solutions::KerrSchild bh_solution(mass, spin, center);
    const ScalarWave::Solutions::RegularSphericalWave flat_wave_solution{
        std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(11., 1.4,
                                                                      0.3)};

    test_kerr(snd_curved_wave_data, bh_solution, flat_wave_solution);
  }
  {  // test with wave profile = plane
    CurvedScalarWave::AnalyticData::ScalarWaveGr<
        ScalarWave::Solutions::PlaneWave<3>, gr::Solutions::KerrSchild>
        curved_wave_data(
            gr::Solutions::KerrSchild(mass, spin, center),
            ScalarWave::Solutions::PlaneWave<3>(
                {{1.0, 1.0, 1.0}}, {{0.0, 0.0, 0.0}},
                std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(
                    11., 1.4, 0.3)));
    const auto snd_curved_wave_data =
        serialize_and_deserialize(curved_wave_data);
    CHECK(curved_wave_data == snd_curved_wave_data);

    // test internals of deserialized data
    const gr::Solutions::KerrSchild bh_solution(mass, spin, center);
    const ScalarWave::Solutions::PlaneWave<3> flat_wave_solution{
        {{1.0, 1.0, 1.0}},
        {{0.0, 0.0, 0.0}},
        std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(11., 1.4,
                                                                      0.3)};

    test_kerr(snd_curved_wave_data, bh_solution, flat_wave_solution);
  }
}

void test_move(const double mass, const std::array<double, 3>& spin,
               const std::array<double, 3>& center) {
  {  // test with wave profile = spherical
    CurvedScalarWave::AnalyticData::ScalarWaveGr<
        ScalarWave::Solutions::RegularSphericalWave, gr::Solutions::KerrSchild>
        curved_wave_data(
            gr::Solutions::KerrSchild(mass, spin, center),
            ScalarWave::Solutions::RegularSphericalWave(
                std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(
                    11., 1.4, 0.3)));
    // since it can't actually be copied
    CurvedScalarWave::AnalyticData::ScalarWaveGr<
        ScalarWave::Solutions::RegularSphericalWave, gr::Solutions::KerrSchild>
        copy_of_curved_wave_data(
            gr::Solutions::KerrSchild(mass, spin, center),
            ScalarWave::Solutions::RegularSphericalWave(
                std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(
                    11., 1.4, 0.3)));
    test_move_semantics(std::move(curved_wave_data), copy_of_curved_wave_data);
  }
  {  // test with wave profile = plane
    CurvedScalarWave::AnalyticData::ScalarWaveGr<
        ScalarWave::Solutions::PlaneWave<3>, gr::Solutions::KerrSchild>
        curved_wave_data(
            gr::Solutions::KerrSchild(mass, spin, center),
            ScalarWave::Solutions::PlaneWave<3>(
                {{1.0, 1.0, 1.0}}, {{0.0, 0.0, 0.0}},
                std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(
                    11., 1.4, 0.3)));
    // since it can't actually be copied
    CurvedScalarWave::AnalyticData::ScalarWaveGr<
        ScalarWave::Solutions::PlaneWave<3>, gr::Solutions::KerrSchild>
        copy_of_curved_wave_data(
            gr::Solutions::KerrSchild(mass, spin, center),
            ScalarWave::Solutions::PlaneWave<3>(
                {{1.0, 1.0, 1.0}}, {{0.0, 0.0, 0.0}},
                std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(
                    11., 1.4, 0.3)));
    test_move_semantics(std::move(curved_wave_data), copy_of_curved_wave_data);
  }
}
}  // namespace

void test_scalarwave_bh() {
  const double mass = 0.7;
  const std::array<double, 3> spin{{0.1, -0.2, 0.3}};
  const std::array<double, 3> center{{0.3, 0.2, 0.4}};
  const gr::Solutions::KerrSchild bh_solution(mass, spin, center);

  test_construct_from_options();
  test_serialize(mass, spin, center);
  test_move(mass, spin, center);

  using background_tag = gr::Solutions::KerrSchild;

  {  // test with wave profile = spherical
    using sw_tag = ScalarWave::Solutions::RegularSphericalWave;
    const CurvedScalarWave::AnalyticData::ScalarWaveGr<sw_tag, background_tag>
        curved_wave_data_bh_far_away{
            background_tag(mass, spin, {{1.e20, 1., 1.}}),
            sw_tag(
                std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(
                    11., 1.5, 0.))};
    const CurvedScalarWave::AnalyticData::ScalarWaveGr<sw_tag, background_tag>
        curved_wave_data{
            background_tag(mass, spin, center),
            sw_tag(
                std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(
                    11., 1.5, 0.))};
    const sw_tag flat_wave_solution{
        std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(11., 1.5,
                                                                      0.)};

    test_tag_retrieval(curved_wave_data_bh_far_away);
    test_tag_retrieval(curved_wave_data);
    test_no_hole(curved_wave_data_bh_far_away, flat_wave_solution);
    test_kerr(curved_wave_data, bh_solution, flat_wave_solution);
    test_kerr_schild_vars(curved_wave_data, bh_solution);
  }
  {  // test with wave profile = plane
    using sw_tag = ScalarWave::Solutions::PlaneWave<3>;
    const CurvedScalarWave::AnalyticData::ScalarWaveGr<sw_tag, background_tag>
        curved_wave_data_bh_far_away{
            background_tag(mass, spin, {{1.e20, 1., 1.}}),
            sw_tag(
                {{1., 1., 1.}}, {{0., 0., 0.}},
                std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(
                    11., 1.5, 0.))};
    const CurvedScalarWave::AnalyticData::ScalarWaveGr<sw_tag, background_tag>
        curved_wave_data{
            background_tag(mass, spin, center),
            sw_tag(
                {{1., 1., 1.}}, {{0., 0., 0.}},
                std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(
                    11., 1.5, 0.))};
    const sw_tag flat_wave_solution{
        {{1., 1., 1.}},
        {{0., 0., 0.}},
        std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(11., 1.5,
                                                                      0.)};

    test_tag_retrieval(curved_wave_data_bh_far_away);
    test_tag_retrieval(curved_wave_data);
    test_no_hole(curved_wave_data_bh_far_away, flat_wave_solution);
    test_kerr(curved_wave_data, bh_solution, flat_wave_solution);
    test_kerr_schild_vars(curved_wave_data, bh_solution);
  }
}

void test_scalarwave_minkowski() {
  // test with wave profile = plane in flat space in 3D
  using flat_background_tag = gr::Solutions::Minkowski<3>;
  using sw_tag = ScalarWave::Solutions::PlaneWave<3>;
  const CurvedScalarWave::AnalyticData::ScalarWaveGr<sw_tag,
                                                     flat_background_tag>
      curved_wave_data{
          flat_background_tag{},
          sw_tag({{1., 1., 1.}}, {{0., 0., 0.}},
                 std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(
                     11., 1.5, 0.))};
  const sw_tag flat_wave_solution{
      {{1., 1., 1.}},
      {{0., 0., 0.}},
      std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(11., 1.5,
                                                                    0.)};

  test_tag_retrieval(curved_wave_data);
  test_no_hole(curved_wave_data, flat_wave_solution);
}

SPECTRE_TEST_CASE("Unit.AnalyticData.CurvedWaveEquation.ScalarWaveGr",
                  "[PointwiseFunctions][Unit]") {
  test_scalarwave_bh();
  test_scalarwave_minkowski();
}

// Check restrictions on input options below

// [[OutputRegex, Mass must be non-negative]]
SPECTRE_TEST_CASE("Unit.AnalyticData.CurvedWaveEquation.ScalarWaveGrMass",
                  "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  CurvedScalarWave::AnalyticData::ScalarWaveGr<
      ScalarWave::Solutions::RegularSphericalWave, gr::Solutions::KerrSchild>
      solution(
          gr::Solutions::KerrSchild(-0.2, {{0.1, -0.2, 0.3}},
                                    {{0.3, 0.2, 0.4}}),
          ScalarWave::Solutions::RegularSphericalWave(
              std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(
                  1., 1., 0.)));
}

// [[OutputRegex, In string:.*At line 3 column 11:.Value -0.5 is below the
// lower bound of 0]]
SPECTRE_TEST_CASE("Unit.AnalyticData.CurvedWaveEquation.ScalarWaveGrOptM",
                  "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  TestHelpers::test_creation<CurvedScalarWave::AnalyticData::ScalarWaveGr<
                                 ScalarWave::Solutions::RegularSphericalWave,
                                 gr::Solutions::KerrSchild>,
                             Metavariables>(
      "Background:\n"
      "  Mass: -0.5\n"
      "  Spin: [0.1, -0.2, 0.3]\n"
      "  Center: [0.7,0.6,-2.]\n"
      "ScalarField:\n"
      "  Profile:\n"
      "    Gaussian:\n"
      "      Amplitude: 11.\n"
      "      Width: 1.4\n"
      "      Center: 0.3");
}

// [[OutputRegex, Spin magnitude must be < 1]]
SPECTRE_TEST_CASE("Unit.AnalyticData.CurvedWaveEquation.ScalarWaveGrSpin",
                  "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  CurvedScalarWave::AnalyticData::ScalarWaveGr<
      ScalarWave::Solutions::RegularSphericalWave, gr::Solutions::KerrSchild>
      solution(
          gr::Solutions::KerrSchild(0.2, {{1.0, 1.0, 1.0}}, {{0.3, 0.2, 0.4}}),
          ScalarWave::Solutions::RegularSphericalWave(
              std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(
                  1., 1., 0.)));
}
