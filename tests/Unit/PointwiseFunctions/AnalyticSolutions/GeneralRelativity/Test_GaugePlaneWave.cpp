// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/GeneralRelativity/VerifyGrSolution.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugePlaneWave.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativesOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<MathFunction<1, Frame::Inertial>,
                   tmpl::list<MathFunctions::PowX<1, Frame::Inertial>>>>;
  };
};

template <size_t Dim, typename DataType>
tnsr::I<DataType, Dim, Frame::Inertial> spatial_coords(
    const DataType& used_for_size) {
  auto x = make_with_value<tnsr::I<DataType, Dim, Frame::Inertial>>(
      used_for_size, 0.0);
  get<0>(x) = 3.0;
  if (Dim >= 2) {
    x.get(1) = 5.0;
  }
  if (Dim >= 3) {
    x.get(2) = 7.0;
  }
  return x;
}

template <size_t Dim>
std::array<double, Dim> make_wave_vector() {
  if constexpr (Dim == 1) {
    return {{0.5}};
  }
  if constexpr (Dim == 2) {
    return {{0.3, 0.4}};
  }
  if constexpr (Dim == 3) {
    return {{0.3, 0.4, 1.2}};
  }
}

template <size_t Dim, typename DataType>
DataType u(const std::array<double, Dim>& k,
           const tnsr::I<DataType, Dim, Frame::Inertial>& x, const double t) {
  const double omega = magnitude(k);
  auto u = make_with_value<DataType>(x, -omega * t);
  for (size_t i = 0; i < Dim; ++i) {
    u += gsl::at(k, i) * x.get(i);
  }
  return u;
}

template <size_t Dim, typename DataType>
void test_gauge_wave(const gr::Solutions::GaugePlaneWave<Dim>& solution,
                     const DataType& used_for_size) {
  using GaugePlaneWave = gr::Solutions::GaugePlaneWave<Dim>;

  // The spacetime metric for this solution is in Kerr-Schild form
  // g_ab = \eta_ab + H l_a l_b
  //
  // where \eta_ab = diag(-1, 1, 1, 1) is the Minkowski metric
  // and   l_a = (-\omega, k_a) is a null vector
  // with k_a a constant wave vector and \omega = k^a k_a
  // and   H = F[u] is a scalar function of u = k_a x^a - \omega t
  //
  // Thus d_a H = F'[u] l_a
  //
  // Therefore d_c g_ab = F'[u] l_c l_a l_b

  // Parameters for GaugePlaneWave solution
  const auto wave_vector = make_wave_vector<Dim>();
  const auto null_covector = prepend(wave_vector, -magnitude(wave_vector));

  // Verify that the solution passed in was constructed with the correct
  // parameters
  REQUIRE(solution ==
          GaugePlaneWave{
              wave_vector,
              std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(2)});

  const tnsr::I<DataType, Dim, Frame::Inertial> x =
      spatial_coords<Dim>(used_for_size);
  const double t = 11.0;

  const auto expected_u = u(wave_vector, x, t);
  const auto expected_h = square(expected_u);
  const auto expected_d_h = 2.0 * expected_u;

  const tuples::tagged_tuple_from_typelist<
      typename gr::Solutions::GaugePlaneWave<Dim>::template tags<DataType>>
      vars = solution.variables(
          x, t, typename GaugePlaneWave::template tags<DataType>{});
  const auto& lapse = get<gr::Tags::Lapse<DataType>>(vars);
  const auto& dt_lapse = get<Tags::dt<gr::Tags::Lapse<DataType>>>(vars);
  const auto& d_lapse =
      get<typename GaugePlaneWave::template DerivLapse<DataType>>(vars);
  const auto& shift = get<gr::Tags::Shift<DataType, Dim>>(vars);
  const auto& d_shift =
      get<typename GaugePlaneWave::template DerivShift<DataType>>(vars);
  const auto& dt_shift = get<Tags::dt<gr::Tags::Shift<DataType, Dim>>>(vars);
  const auto& gamma = get<gr::Tags::SpatialMetric<DataType, Dim>>(vars);
  const auto& dt_gamma =
      get<Tags::dt<gr::Tags::SpatialMetric<DataType, Dim>>>(vars);
  const auto& d_gamma =
      get<typename GaugePlaneWave::template DerivSpatialMetric<DataType>>(vars);

  // Rather than directly checking the 3+1 quantities, the test computes the
  // spacetime metric and its derivatives from the 3+1 quantities and checks
  // that they are the simple expressions listed above.

  const auto g = gr::spacetime_metric(lapse, shift, gamma);

  for (size_t a = 0; a < Dim + 1; ++a) {
    for (size_t b = a; b < Dim + 1; ++b) {
      DataType expected_g_ab =
          expected_h * gsl::at(null_covector, a) * gsl::at(null_covector, b);
      if (a == b) {
        expected_g_ab += (a == 0 ? -1.0 : 1.0);
      }
      CHECK_ITERABLE_APPROX(expected_g_ab, g.get(a, b));
    }
  }

  const auto d_g = gr::derivatives_of_spacetime_metric(
      lapse, dt_lapse, d_lapse, shift, dt_shift, d_shift, gamma, dt_gamma,
      d_gamma);

  for (size_t c = 0; c < Dim + 1; ++c) {
    CAPTURE(c);
    for (size_t a = 0; a < Dim + 1; ++a) {
      CAPTURE(a);
      for (size_t b = a; b < Dim + 1; ++b) {
        CAPTURE(b);
        DataType expected_d_g_cab = expected_d_h * gsl::at(null_covector, c) *
                                    gsl::at(null_covector, a) *
                                    gsl::at(null_covector, b);
        CHECK_ITERABLE_APPROX(expected_d_g_cab, d_g.get(c, a, b));
      }
    }
  }

  // Check that we can retrieve tags individually from the solution
  tmpl::for_each<typename GaugePlaneWave::template tags<DataType>>(
      [&solution, &vars, &x, &t](auto tag_v) {
        using Tag = tmpl::type_from<decltype(tag_v)>;
        CHECK_ITERABLE_APPROX(get<Tag>(vars), get<Tag>(solution.variables(
                                                  x, t, tmpl::list<Tag>{})));
      });

  TestHelpers::AnalyticSolutions::test_tag_retrieval(
      solution, x, t,
      typename gr::Solutions::GaugePlaneWave<Dim>::template tags<DataType>{});
}

void test_consistency() {
  const gr::Solutions::GaugePlaneWave<3> solution(
      make_wave_vector<3>(),
      std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(2));
  TestHelpers::VerifyGrSolution::verify_consistency(
      solution, 1.234, tnsr::I<double, 3>{{{1.2, 2.3, 3.4}}}, 0.01, 1.0e-8);
}

void test_serialize() {
  register_factory_classes_with_charm<Metavariables>();
  const gr::Solutions::GaugePlaneWave<3> solution(
      make_wave_vector<3>(),
      std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(2));
  test_serialization(solution);
  test_gauge_wave(serialize_and_deserialize(solution), DataVector{5});
}

void test_copy_and_move() {
  gr::Solutions::GaugePlaneWave<3> solution(
      make_wave_vector<3>(),
      std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(2));
  test_copy_semantics(solution);
  auto solution_copy = solution;
  // clang-tidy: std::move of trivially copyable type
  test_move_semantics(std::move(solution), solution_copy);  // NOLINT
}

void test_construct_from_options() {
  const gr::Solutions::GaugePlaneWave<3> solution(
      make_wave_vector<3>(),
      std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(2));
  const auto created =
      TestHelpers::test_creation<gr::Solutions::GaugePlaneWave<3>,
                                 Metavariables>(
          "WaveVector: [0.3, 0.4, 1.2]\n"
          "Profile:\n"
          "    PowX:\n"
          "      Power: 2");
  CHECK(created == solution);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Gr.GaugePlaneWave",
                  "[PointwiseFunctions][Unit]") {
  {
    const gr::Solutions::GaugePlaneWave<1> solution(
        make_wave_vector<1>(),
        std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(2));
    test_gauge_wave(
        solution, DataVector(5, std::numeric_limits<double>::signaling_NaN()));
    test_gauge_wave(solution, std::numeric_limits<double>::signaling_NaN());
  }
  {
    const gr::Solutions::GaugePlaneWave<2> solution(
        make_wave_vector<2>(),
        std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(2));
    test_gauge_wave(
        solution, DataVector(5, std::numeric_limits<double>::signaling_NaN()));
    test_gauge_wave(solution, std::numeric_limits<double>::signaling_NaN());
  }
  {
    const gr::Solutions::GaugePlaneWave<3> solution(
        make_wave_vector<3>(),
        std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(2));
    test_gauge_wave(
        solution, DataVector(5, std::numeric_limits<double>::signaling_NaN()));
    test_gauge_wave(solution, std::numeric_limits<double>::signaling_NaN());
  }
  test_consistency();
  test_copy_and_move();
  test_serialize();
  test_construct_from_options();
}
