// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrapGh.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_forward_declare Tags::deriv

namespace {
template <typename SolutionType>
void test_generalized_harmonic_solution(
    const SolutionType& solution,
    const GeneralizedHarmonic::Solutions::WrapGh<SolutionType>
        wrapped_solution) {
  const DataVector data_vector{5.0, 4.0};
  const tnsr::I<DataVector, wrapped_solution.volume_dim, Frame::Inertial> x{
      data_vector};
  const double t = std::numeric_limits<double>::signaling_NaN();

  // Get quantities from an analytic solution of Einstein's equations
  const auto vars = solution.variables(
      x, t, typename SolutionType::template tags<DataVector>{});
  const auto lapse = get<gr::Tags::Lapse<DataVector>>(vars);
  const auto dt_lapse = get<Tags::dt<gr::Tags::Lapse<DataVector>>>(vars);
  const auto d_lapse = get<
      Tags::deriv<gr::Tags::Lapse<DataVector>,
                  tmpl::size_t<wrapped_solution.volume_dim>, Frame::Inertial>>(
      vars);
  const auto shift = get<gr::Tags::Shift<wrapped_solution.volume_dim,
                                         Frame::Inertial, DataVector>>(vars);
  const auto dt_shift =
      get<Tags::dt<gr::Tags::Shift<wrapped_solution.volume_dim, Frame::Inertial,
                                   DataVector>>>(vars);
  const auto d_shift = get<Tags::deriv<
      gr::Tags::Shift<wrapped_solution.volume_dim, Frame::Inertial, DataVector>,
      tmpl::size_t<wrapped_solution.volume_dim>, Frame::Inertial>>(vars);
  const auto g =
      get<gr::Tags::SpatialMetric<wrapped_solution.volume_dim, Frame::Inertial,
                                  DataVector>>(vars);
  const auto dt_g =
      get<Tags::dt<gr::Tags::SpatialMetric<wrapped_solution.volume_dim,
                                           Frame::Inertial, DataVector>>>(vars);
  const auto d_g = get<
      Tags::deriv<gr::Tags::SpatialMetric<wrapped_solution.volume_dim,
                                          Frame::Inertial, DataVector>,
                  tmpl::size_t<wrapped_solution.volume_dim>, Frame::Inertial>>(
      vars);
  const auto inv_g =
      get<gr::Tags::InverseSpatialMetric<wrapped_solution.volume_dim,
                                         Frame::Inertial, DataVector>>(vars);
  const auto extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<wrapped_solution.volume_dim,
                                       Frame::Inertial, DataVector>>(vars);
  const auto sqrt_det_g = get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(vars);

  // Get quantities from the same solution, wrapped in a
  // WrapGh
  const auto wrapped_lapse =
      get<gr::Tags::Lapse<DataVector>>(wrapped_solution.variables(
          x, t, tmpl::list<gr::Tags::Lapse<DataVector>>{}));
  const auto wrapped_dt_lapse =
      get<Tags::dt<gr::Tags::Lapse<DataVector>>>(wrapped_solution.variables(
          x, t, tmpl::list<Tags::dt<gr::Tags::Lapse<DataVector>>>{}));
  const auto wrapped_d_lapse = get<
      Tags::deriv<gr::Tags::Lapse<DataVector>,
                  tmpl::size_t<wrapped_solution.volume_dim>, Frame::Inertial>>(
      wrapped_solution.variables(
          x, t,
          tmpl::list<Tags::deriv<gr::Tags::Lapse<DataVector>,
                                 tmpl::size_t<wrapped_solution.volume_dim>,
                                 Frame::Inertial>>{}));
  const auto wrapped_shift = get<gr::Tags::Shift<wrapped_solution.volume_dim,
                                                 Frame::Inertial, DataVector>>(
      wrapped_solution.variables(
          x, t,
          tmpl::list<gr::Tags::Shift<wrapped_solution.volume_dim,
                                     Frame::Inertial, DataVector>>{}));
  const auto wrapped_dt_shift = get<Tags::dt<gr::Tags::Shift<
      wrapped_solution.volume_dim, Frame::Inertial, DataVector>>>(
      wrapped_solution.variables(
          x, t,
          tmpl::list<Tags::dt<gr::Tags::Shift<
              wrapped_solution.volume_dim, Frame::Inertial, DataVector>>>{}));
  const auto wrapped_d_shift = get<Tags::deriv<
      gr::Tags::Shift<wrapped_solution.volume_dim, Frame::Inertial, DataVector>,
      tmpl::size_t<wrapped_solution.volume_dim>, Frame::Inertial>>(
      wrapped_solution.variables(
          x, t,
          tmpl::list<Tags::deriv<gr::Tags::Shift<wrapped_solution.volume_dim,
                                                 Frame::Inertial, DataVector>,
                                 tmpl::size_t<wrapped_solution.volume_dim>,
                                 Frame::Inertial>>{}));
  const auto wrapped_g = get<gr::Tags::SpatialMetric<
      wrapped_solution.volume_dim, Frame::Inertial, DataVector>>(
      wrapped_solution.variables(
          x, t,
          tmpl::list<gr::Tags::SpatialMetric<wrapped_solution.volume_dim,
                                             Frame::Inertial, DataVector>>{}));
  const auto wrapped_dt_g = get<Tags::dt<gr::Tags::SpatialMetric<
      wrapped_solution.volume_dim, Frame::Inertial, DataVector>>>(
      wrapped_solution.variables(
          x, t,
          tmpl::list<Tags::dt<gr::Tags::SpatialMetric<
              wrapped_solution.volume_dim, Frame::Inertial, DataVector>>>{}));
  const auto wrapped_d_g = get<
      Tags::deriv<gr::Tags::SpatialMetric<wrapped_solution.volume_dim,
                                          Frame::Inertial, DataVector>,
                  tmpl::size_t<wrapped_solution.volume_dim>, Frame::Inertial>>(
      wrapped_solution.variables(
          x, t,
          tmpl::list<Tags::deriv<
              gr::Tags::SpatialMetric<wrapped_solution.volume_dim,
                                      Frame::Inertial, DataVector>,
              tmpl::size_t<wrapped_solution.volume_dim>, Frame::Inertial>>{}));
  const auto wrapped_inv_g = get<gr::Tags::InverseSpatialMetric<
      wrapped_solution.volume_dim, Frame::Inertial, DataVector>>(
      wrapped_solution.variables(
          x, t,
          tmpl::list<gr::Tags::InverseSpatialMetric<
              wrapped_solution.volume_dim, Frame::Inertial, DataVector>>{}));
  const auto wrapped_extrinsic_curvature = get<gr::Tags::ExtrinsicCurvature<
      wrapped_solution.volume_dim, Frame::Inertial, DataVector>>(
      wrapped_solution.variables(
          x, t,
          tmpl::list<gr::Tags::ExtrinsicCurvature<
              wrapped_solution.volume_dim, Frame::Inertial, DataVector>>{}));
  const auto wrapped_sqrt_det_g =
      get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(
          wrapped_solution.variables(
              x, t, tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataVector>>{}));

  // Check that the solution and the wrapped solution returned the same
  // results
  CHECK(lapse == wrapped_lapse);
  CHECK(dt_lapse == wrapped_dt_lapse);
  CHECK(d_lapse == wrapped_d_lapse);

  CHECK(shift == wrapped_shift);
  CHECK(dt_shift == wrapped_dt_shift);
  CHECK(d_shift == wrapped_d_shift);

  CHECK(g == wrapped_g);
  CHECK(dt_g == wrapped_dt_g);
  CHECK(d_g == wrapped_d_g);

  // Check that the wrapped solution returns the correct psi, pi, phi
  const auto psi = gr::spacetime_metric(lapse, shift, g);
  const auto phi =
      GeneralizedHarmonic::phi(lapse, d_lapse, shift, d_shift, g, d_g);
  const auto pi =
      GeneralizedHarmonic::pi(lapse, dt_lapse, shift, dt_shift, g, dt_g, phi);

  const auto wrapped_gh_vars = wrapped_solution.variables(
      x, t,
      tmpl::list<gr::Tags::SpacetimeMetric<wrapped_solution.volume_dim,
                                           Frame::Inertial, DataVector>,
                 GeneralizedHarmonic::Tags::Pi<wrapped_solution.volume_dim,
                                               Frame::Inertial>,
                 GeneralizedHarmonic::Tags::Phi<wrapped_solution.volume_dim,
                                                Frame::Inertial>>{});
  CHECK(psi == get<gr::Tags::SpacetimeMetric<wrapped_solution.volume_dim,
                                             Frame::Inertial, DataVector>>(
                   wrapped_gh_vars));
  CHECK(pi ==
        get<GeneralizedHarmonic::Tags::Pi<wrapped_solution.volume_dim,
                                          Frame::Inertial>>(wrapped_gh_vars));
  CHECK(phi ==
        get<GeneralizedHarmonic::Tags::Phi<wrapped_solution.volume_dim,
                                           Frame::Inertial>>(wrapped_gh_vars));

  test_serialization(wrapped_solution);
  // test operator !=
  CHECK_FALSE(wrapped_solution != wrapped_solution);
}

struct WrapGh {
  using type =
      GeneralizedHarmonic::Solutions::WrapGh<gr::Solutions::KerrSchild>;
  static constexpr OptionString help{"A wrapped generalized harmonic solution"};
};

void test_construct_from_options() {
  Options<tmpl::list<WrapGh>> opts("");
  opts.parse(
      "WrapGh:\n"
      "  Mass: 0.5\n"
      "  Spin: [0.1,0.2,0.3]\n"
      "  Center: [1.0,3.0,2.0]");
  const double mass = 0.5;
  const std::array<double, 3> spin{{0.1, 0.2, 0.3}};
  const std::array<double, 3> center{{1.0, 3.0, 2.0}};
  CHECK(opts.get<WrapGh>() ==
        GeneralizedHarmonic::Solutions::WrapGh<gr::Solutions::KerrSchild>(
            mass, spin, center));
}

template <typename SolutionType>
void test_copy_and_move(const SolutionType& solution) noexcept {
  test_copy_semantics(solution);
  auto solution_copy = solution;
  auto solution_copy2 = solution;
  // clang-tidy: std::move of trivially copyable type
  test_move_semantics(std::move(solution_copy2), solution_copy);  // NOLINT
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Gr.WrapGh",
                  "[PointwiseFunctions][Unit]") {
  gr::Solutions::Minkowski<1> minkowski1{};
  gr::Solutions::Minkowski<2> minkowski2{};
  gr::Solutions::Minkowski<3> minkowski3{};

  const double mass = 0.5;
  const std::array<double, 3> spin{{0.1, 0.2, 0.3}};
  const std::array<double, 3> center{{1.0, 3.0, 2.0}};
  gr::Solutions::KerrSchild kerr_schild(mass, spin, center);

  GeneralizedHarmonic::Solutions::WrapGh<gr::Solutions::Minkowski<1>>
      wrapped_minkowski1{};
  GeneralizedHarmonic::Solutions::WrapGh<gr::Solutions::Minkowski<2>>
      wrapped_minkowski2{};
  GeneralizedHarmonic::Solutions::WrapGh<gr::Solutions::Minkowski<3>>
      wrapped_minkowski3{};
  GeneralizedHarmonic::Solutions::WrapGh<gr::Solutions::KerrSchild>
      wrapped_kerr_schild(mass, spin, center);

  test_generalized_harmonic_solution<gr::Solutions::Minkowski<1>>(
      minkowski1, wrapped_minkowski1);
  test_generalized_harmonic_solution<gr::Solutions::Minkowski<2>>(
      minkowski2, wrapped_minkowski2);
  test_generalized_harmonic_solution<gr::Solutions::Minkowski<3>>(
      minkowski3, wrapped_minkowski3);
  test_generalized_harmonic_solution<gr::Solutions::KerrSchild>(
      kerr_schild, wrapped_kerr_schild);

  test_serialization(wrapped_minkowski1);
  test_serialization(wrapped_minkowski2);
  test_serialization(wrapped_minkowski3);
  test_serialization(wrapped_kerr_schild);

  test_copy_and_move(wrapped_minkowski1);
  test_copy_and_move(wrapped_minkowski2);
  test_copy_and_move(wrapped_minkowski3);
  test_copy_and_move(wrapped_kerr_schild);

  test_construct_from_options();
}
