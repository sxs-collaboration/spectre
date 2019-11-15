// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <memory>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/ScalarWave/System.hpp"      // IWYU pragma: keep
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticData/TestHelpers.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticData/CurvedWaveEquation/ScalarWaveKerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
template <typename ScalarFieldSolution>
struct ScalarWaveKerr {
  using type =
      CurvedScalarWave::AnalyticData::ScalarWaveKerrSchild<ScalarFieldSolution>;
  static constexpr OptionString help{"A scalar wave in Kerr spacetime"};
};

template <typename ScalarFieldSolution>
void test_tag_retrieval(
    const CurvedScalarWave::AnalyticData::ScalarWaveKerrSchild<
        ScalarFieldSolution>& curved_wave_data) noexcept {
  const tnsr::I<DataVector, 3> x{std::array<DataVector, 3>{
      {DataVector({0., 1., 2., 3.}), DataVector({0., 0., 0., 0.}),
       DataVector({0., 0., 0., 0.})}}};

  // Evaluate solution
  TestHelpers::AnalyticData::test_tag_retrieval(
      curved_wave_data, x,
      typename CurvedScalarWave::AnalyticData::ScalarWaveKerrSchild<
          ScalarFieldSolution>::tags());
}

template <typename ScalarFieldSolution>
void test_no_hole(const CurvedScalarWave::AnalyticData::ScalarWaveKerrSchild<
                      ScalarFieldSolution>& curved_wave_data,
                  const ScalarFieldSolution& flat_wave_solution) noexcept {
  const tnsr::I<DataVector, 3> x{std::array<DataVector, 3>{
      {DataVector({0., 1., 2., 3.}), DataVector({0., 0., 0., 0.}),
       DataVector({0., 0., 0., 0.})}}};
  auto vars = curved_wave_data.variables(
      x, tmpl::list<CurvedScalarWave::Pi, CurvedScalarWave::Phi<3>,
                    CurvedScalarWave::Psi>{});
  auto flat_vars = flat_wave_solution.variables(
      x, 0., tmpl::list<ScalarWave::Pi, ScalarWave::Phi<3>, ScalarWave::Psi>{});
  CHECK_ITERABLE_APPROX(get(get<CurvedScalarWave::Psi>(vars)),
                        get(get<ScalarWave::Psi>(flat_vars)));
  CHECK_ITERABLE_APPROX(get(get<CurvedScalarWave::Pi>(vars)),
                        get(get<ScalarWave::Pi>(flat_vars)));
  CHECK_ITERABLE_APPROX(get<0>(get<CurvedScalarWave::Phi<3>>(vars)),
                        get<0>(get<ScalarWave::Phi<3>>(flat_vars)));
  CHECK_ITERABLE_APPROX(get<1>(get<CurvedScalarWave::Phi<3>>(vars)),
                        get<1>(get<ScalarWave::Phi<3>>(flat_vars)));
  CHECK_ITERABLE_APPROX(get<2>(get<CurvedScalarWave::Phi<3>>(vars)),
                        get<2>(get<ScalarWave::Phi<3>>(flat_vars)));
}

template <typename ScalarFieldSolution>
void test_kerr(const CurvedScalarWave::AnalyticData::ScalarWaveKerrSchild<
                   ScalarFieldSolution>& curved_wave_data,
               const gr::Solutions::KerrSchild& bh_solution,
               const ScalarFieldSolution& flat_wave_solution) noexcept {
  const tnsr::I<DataVector, 3> x{std::array<DataVector, 3>{
      {DataVector({0., 1., 2., 3.}), DataVector({0., 0., 0., 0.}),
       DataVector({0., 0., 0., 0.})}}};
  auto vars = curved_wave_data.variables(
      x, tmpl::list<CurvedScalarWave::Pi, CurvedScalarWave::Phi<3>,
                    CurvedScalarWave::Psi>{});

  const auto flat_wave_vars = flat_wave_solution.variables(
      x, 0., ScalarWave::System<3>::variables_tag::tags_list{});
  const auto flat_wave_dt_vars = flat_wave_solution.variables(
      x, 0.,
      tmpl::list<::Tags::dt<ScalarWave::Pi>, ::Tags::dt<ScalarWave::Phi<3>>,
                 ::Tags::dt<ScalarWave::Psi>>{});

  const auto kerr_variables = bh_solution.variables(
      x, 0., gr::Solutions::KerrSchild::tags<DataVector>{});

  // construct the vars in-situ
  const auto& local_psi = get<ScalarWave::Psi>(flat_wave_vars);
  const auto& local_phi = get<ScalarWave::Phi<3>>(flat_wave_vars);
  auto local_pi = make_with_value<Scalar<DataVector>>(x, 0.);
  {
    const auto shift_dot_dpsi = dot_product(
        get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(kerr_variables),
        get<ScalarWave::Phi<3>>(flat_wave_vars));
    get(local_pi) = (get(shift_dot_dpsi) -
                     get(get<::Tags::dt<ScalarWave::Psi>>(flat_wave_dt_vars))) /
                    get(get<gr::Tags::Lapse<DataVector>>(kerr_variables));
  }

  CHECK_ITERABLE_APPROX(get(get<CurvedScalarWave::Psi>(vars)), get(local_psi));
  CHECK_ITERABLE_APPROX(get(get<CurvedScalarWave::Pi>(vars)), get(local_pi));
  CHECK_ITERABLE_APPROX(get<0>(get<CurvedScalarWave::Phi<3>>(vars)),
                        get<0>(local_phi));
  CHECK_ITERABLE_APPROX(get<1>(get<CurvedScalarWave::Phi<3>>(vars)),
                        get<1>(local_phi));
  CHECK_ITERABLE_APPROX(get<2>(get<CurvedScalarWave::Phi<3>>(vars)),
                        get<2>(local_phi));
}

template <typename ScalarFieldSolution>
void test_kerr_schild_vars(
    const CurvedScalarWave::AnalyticData::ScalarWaveKerrSchild<
        ScalarFieldSolution>& curved_wave_data,
    const gr::Solutions::KerrSchild& bh_solution) noexcept {
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

void test_construct_from_options() noexcept {
  {  // test with wave profile = spherical
    Options<
        tmpl::list<ScalarWaveKerr<ScalarWave::Solutions::RegularSphericalWave>>>
        opts("");
    opts.parse(
        "ScalarWaveKerr:\n"
        "  BlackHoleMass: 0.7\n"
        "  BlackHoleSpin: [0.1,-0.2,0.3]\n"
        "  BlackHoleCenter: [0.7,0.6,-2.]\n"
        "  WaveProfile:\n"
        "    Gaussian:\n"
        "      Amplitude: 11.\n"
        "      Width: 1.4\n"
        "      Center: 0.3");
    const auto curved_wave_data_constructed_from_opts =
        opts.get<ScalarWaveKerr<ScalarWave::Solutions::RegularSphericalWave>>();
    CHECK(curved_wave_data_constructed_from_opts ==
          CurvedScalarWave::AnalyticData::ScalarWaveKerrSchild<
              ScalarWave::Solutions::RegularSphericalWave>(
              0.7, {{0.1, -0.2, 0.3}}, {{0.7, 0.6, -2.}},
              std::make_unique<MathFunctions::Gaussian>(11., 1.4, 0.3)));

    // test internals of data constructed from options
    const gr::Solutions::KerrSchild bh_solution(0.7, {{0.1, -0.2, 0.3}},
                                                {{0.7, 0.6, -2.}});
    const ScalarWave::Solutions::RegularSphericalWave flat_wave_solution{
        std::make_unique<MathFunctions::Gaussian>(11., 1.4, 0.3)};

    test_kerr(curved_wave_data_constructed_from_opts, bh_solution,
              flat_wave_solution);
  }
  {  // test with wave profile = plane
    Options<tmpl::list<ScalarWaveKerr<ScalarWave::Solutions::PlaneWave<3>>>>
        opts("");
    opts.parse(
        "ScalarWaveKerr:\n"
        "  BlackHoleMass: 0.7\n"
        "  BlackHoleSpin: [0.1,-0.2,0.3]\n"
        "  BlackHoleCenter: [0.7,0.6,-2.]\n"
        "  WaveVector: [1.0, 1.0, 1.0]\n"
        "  WaveCenter: [0.0, 0.0, 0.0]\n"
        "  WaveProfile:\n"
        "    Gaussian:\n"
        "      Amplitude: 11.\n"
        "      Width: 1.4\n"
        "      Center: 0.3");
    const auto curved_wave_data_constructed_from_opts =
        opts.get<ScalarWaveKerr<ScalarWave::Solutions::PlaneWave<3>>>();
    CHECK(curved_wave_data_constructed_from_opts ==
          CurvedScalarWave::AnalyticData::ScalarWaveKerrSchild<
              ScalarWave::Solutions::PlaneWave<3>>(
              0.7, {{0.1, -0.2, 0.3}}, {{0.7, 0.6, -2.}}, {{1.0, 1.0, 1.0}},
              {{0.0, 0.0, 0.0}},
              std::make_unique<MathFunctions::Gaussian>(11., 1.4, 0.3)));

    // test internals of data constructed from options
    const gr::Solutions::KerrSchild bh_solution(0.7, {{0.1, -0.2, 0.3}},
                                                {{0.7, 0.6, -2.}});
    const ScalarWave::Solutions::PlaneWave<3> flat_wave_solution{
        {{1.0, 1.0, 1.0}},
        {{0.0, 0.0, 0.0}},
        std::make_unique<MathFunctions::Gaussian>(11., 1.4, 0.3)};

    test_kerr(curved_wave_data_constructed_from_opts, bh_solution,
              flat_wave_solution);
  }
}

void test_serialize(const double mass, const std::array<double, 3>& spin,
                    const std::array<double, 3>& center) noexcept {
  {  // test with wave profile = spherical
    CurvedScalarWave::AnalyticData::ScalarWaveKerrSchild<
        ScalarWave::Solutions::RegularSphericalWave>
        curved_wave_data(
            mass, spin, center,
            std::make_unique<MathFunctions::Gaussian>(11., 1.4, 0.3));
    Parallel::register_derived_classes_with_charm<MathFunction<1>>();
    const auto snd_curved_wave_data =
        serialize_and_deserialize(curved_wave_data);
    CHECK(curved_wave_data == snd_curved_wave_data);

    // test internals of deserialized data
    const gr::Solutions::KerrSchild bh_solution(mass, spin, center);
    const ScalarWave::Solutions::RegularSphericalWave flat_wave_solution{
        std::make_unique<MathFunctions::Gaussian>(11., 1.4, 0.3)};

    test_kerr(snd_curved_wave_data, bh_solution, flat_wave_solution);
  }
  {  // test with wave profile = plane
    CurvedScalarWave::AnalyticData::ScalarWaveKerrSchild<
        ScalarWave::Solutions::PlaneWave<3>>
        curved_wave_data(
            mass, spin, center, {{1.0, 1.0, 1.0}}, {{0.0, 0.0, 0.0}},
            std::make_unique<MathFunctions::Gaussian>(11., 1.4, 0.3));
    Parallel::register_derived_classes_with_charm<MathFunction<1>>();
    const auto snd_curved_wave_data =
        serialize_and_deserialize(curved_wave_data);
    CHECK(curved_wave_data == snd_curved_wave_data);

    // test internals of deserialized data
    const gr::Solutions::KerrSchild bh_solution(mass, spin, center);
    const ScalarWave::Solutions::PlaneWave<3> flat_wave_solution{
        {{1.0, 1.0, 1.0}},
        {{0.0, 0.0, 0.0}},
        std::make_unique<MathFunctions::Gaussian>(11., 1.4, 0.3)};

    test_kerr(snd_curved_wave_data, bh_solution, flat_wave_solution);
  }
}

void test_move(const double mass, const std::array<double, 3>& spin,
               const std::array<double, 3>& center) noexcept {
  {  // test with wave profile = spherical
    CurvedScalarWave::AnalyticData::ScalarWaveKerrSchild<
        ScalarWave::Solutions::RegularSphericalWave>
        curved_wave_data(
            mass, spin, center,
            std::make_unique<MathFunctions::Gaussian>(11., 1.4, 0.3));
    // since it can't actually be copied
    CurvedScalarWave::AnalyticData::ScalarWaveKerrSchild<
        ScalarWave::Solutions::RegularSphericalWave>
        copy_of_curved_wave_data(
            mass, spin, center,
            std::make_unique<MathFunctions::Gaussian>(11., 1.4, 0.3));
    test_move_semantics(std::move(curved_wave_data), copy_of_curved_wave_data);
  }
  {  // test with wave profile = plane
    CurvedScalarWave::AnalyticData::ScalarWaveKerrSchild<
        ScalarWave::Solutions::PlaneWave<3>>
        curved_wave_data(
            mass, spin, center, {{1.0, 1.0, 1.0}}, {{0.0, 0.0, 0.0}},
            std::make_unique<MathFunctions::Gaussian>(11., 1.4, 0.3));
    // since it can't actually be copied
    CurvedScalarWave::AnalyticData::ScalarWaveKerrSchild<
        ScalarWave::Solutions::PlaneWave<3>>
        copy_of_curved_wave_data(
            mass, spin, center, {{1.0, 1.0, 1.0}}, {{0.0, 0.0, 0.0}},
            std::make_unique<MathFunctions::Gaussian>(11., 1.4, 0.3));
    test_move_semantics(std::move(curved_wave_data), copy_of_curved_wave_data);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.AnalyticData.CurvedWaveEquation.ScalarWaveKerrSchild",
                  "[PointwiseFunctions][Unit]") {
  const double mass = 0.7;
  const std::array<double, 3> spin{{0.1, -0.2, 0.3}};
  const std::array<double, 3> center{{0.3, 0.2, 0.4}};
  const gr::Solutions::KerrSchild bh_solution(mass, spin, center);

  test_construct_from_options();
  test_serialize(mass, spin, center);
  test_move(mass, spin, center);

  {  // test with wave profile = spherical
    using sw_tag = ScalarWave::Solutions::RegularSphericalWave;
    const CurvedScalarWave::AnalyticData::ScalarWaveKerrSchild<sw_tag>
        curved_wave_data_bh_far_away{
            mass,
            spin,
            {{1.e20, 1., 1.}},
            std::make_unique<MathFunctions::Gaussian>(11., 1.5, 0.)};
    const CurvedScalarWave::AnalyticData::ScalarWaveKerrSchild<sw_tag>
        curved_wave_data{
            mass, spin, center,
            std::make_unique<MathFunctions::Gaussian>(11., 1.5, 0.)};
    const sw_tag flat_wave_solution{
        std::make_unique<MathFunctions::Gaussian>(11., 1.5, 0.)};

    test_tag_retrieval(curved_wave_data_bh_far_away);
    test_tag_retrieval(curved_wave_data);
    test_no_hole(curved_wave_data_bh_far_away, flat_wave_solution);
    test_kerr(curved_wave_data, bh_solution, flat_wave_solution);
    test_kerr_schild_vars(curved_wave_data, bh_solution);
  }
  {  // test with wave profile = plane
    using sw_tag = ScalarWave::Solutions::PlaneWave<3>;
    const CurvedScalarWave::AnalyticData::ScalarWaveKerrSchild<sw_tag>
        curved_wave_data_bh_far_away{
            mass,
            spin,
            {{1.e20, 1., 1.}},
            {{1., 1., 1.}},
            {{0., 0., 0.}},
            std::make_unique<MathFunctions::Gaussian>(11., 1.5, 0.)};
    const CurvedScalarWave::AnalyticData::ScalarWaveKerrSchild<sw_tag>
        curved_wave_data{
            mass,
            spin,
            center,
            {{1., 1., 1.}},
            {{0., 0., 0.}},
            std::make_unique<MathFunctions::Gaussian>(11., 1.5, 0.)};
    const sw_tag flat_wave_solution{
        {{1., 1., 1.}},
        {{0., 0., 0.}},
        std::make_unique<MathFunctions::Gaussian>(11., 1.5, 0.)};

    test_tag_retrieval(curved_wave_data_bh_far_away);
    test_tag_retrieval(curved_wave_data);
    test_no_hole(curved_wave_data_bh_far_away, flat_wave_solution);
    test_kerr(curved_wave_data, bh_solution, flat_wave_solution);
    test_kerr_schild_vars(curved_wave_data, bh_solution);
  }
}

/// Check restrictions on input options below

// [[OutputRegex, Mass must be non-negative]]
SPECTRE_TEST_CASE(
    "Unit.AnalyticData.CurvedWaveEquation.ScalarWaveKerrSchildMass",
    "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  CurvedScalarWave::AnalyticData::ScalarWaveKerrSchild<
      ScalarWave::Solutions::RegularSphericalWave>
      solution(-0.2, {{0.1, -0.2, 0.3}}, {{0.3, 0.2, 0.4}},
               std::make_unique<MathFunctions::Gaussian>(1., 1., 0.));
}

// [[OutputRegex, In string:.*At line 2 column 18:.Value -0.5 is below the
// lower bound of 0]]
SPECTRE_TEST_CASE(
    "Unit.AnalyticData.CurvedWaveEquation.ScalarWaveKerrSchildOptM",
    "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  Options<
      tmpl::list<ScalarWaveKerr<ScalarWave::Solutions::RegularSphericalWave>>>
      opts("");
  opts.parse(
      "ScalarWaveKerr:\n"
      "  BlackHoleMass: -0.5\n"
      "  BlackHoleSpin: [0.1,0.2,0.3]\n"
      "  BlackHoleCenter: [1.0,3.0,2.0]\n"
      "  WaveProfile:\n"
      "    Gaussian:\n"
      "      Amplitude: 1.\n"
      "      Width: 1.\n"
      "      Center: 0.");
  opts.get<ScalarWaveKerr<ScalarWave::Solutions::RegularSphericalWave>>();
}

// [[OutputRegex, Spin magnitude must be < 1]]
SPECTRE_TEST_CASE(
    "Unit.AnalyticData.CurvedWaveEquation.ScalarWaveKerrSchildSpin",
    "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  CurvedScalarWave::AnalyticData::ScalarWaveKerrSchild<
      ScalarWave::Solutions::RegularSphericalWave>
      solution(0.2, {{1.0, 1.0, 1.0}}, {{0.3, 0.2, 0.4}},
               std::make_unique<MathFunctions::Gaussian>(1., 1., 0.));
}
