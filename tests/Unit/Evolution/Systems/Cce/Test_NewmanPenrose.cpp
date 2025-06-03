// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/Cce/NewmanPenrose.hpp"
#include "Evolution/Systems/Cce/PreSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/PrecomputeCceDependencies.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/VolumeTestHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
namespace {
void pypp_test_volume_np_spin_coefficients() {
  const pypp::SetupLocalPythonEnvironment
      local_python_env{"Evolution/Systems/Cce/"};

  const size_t num_pts = 5;

  pypp::check_with_random_values<1>(&newman_penrose_alpha, "NewmanPenrose",
                                    {"newman_penrose_alpha"}, {{{1.0, 5.0}}},
                                    DataVector{num_pts});
  pypp::check_with_random_values<1>(&newman_penrose_beta, "NewmanPenrose",
                                    {"newman_penrose_beta"}, {{{1.0, 5.0}}},
                                    DataVector{num_pts});
  pypp::check_with_random_values<1>(&newman_penrose_gamma, "NewmanPenrose",
                                    {"newman_penrose_gamma"}, {{{1.0, 5.0}}},
                                    DataVector{num_pts});
  pypp::check_with_random_values<1>(&newman_penrose_epsilon, "NewmanPenrose",
                                    {"newman_penrose_epsilon"}, {{{1.0, 5.0}}},
                                    DataVector{num_pts});
  // In our choice of tetrad, \kappa=0
  pypp::check_with_random_values<1>(&newman_penrose_tau, "NewmanPenrose",
                                    {"newman_penrose_tau"}, {{{1.0, 5.0}}},
                                    DataVector{num_pts});
  pypp::check_with_random_values<1>(&newman_penrose_sigma, "NewmanPenrose",
                                    {"newman_penrose_sigma"}, {{{1.0, 5.0}}},
                                    DataVector{num_pts});
  pypp::check_with_random_values<1>(&newman_penrose_rho, "NewmanPenrose",
                                    {"newman_penrose_rho"}, {{{1.0, 5.0}}},
                                    DataVector{num_pts});
  pypp::check_with_random_values<1>(&newman_penrose_pi, "NewmanPenrose",
                                    {"newman_penrose_pi"}, {{{1.0, 5.0}}},
                                    DataVector{num_pts});
  pypp::check_with_random_values<1>(&newman_penrose_nu, "NewmanPenrose",
                                    {"newman_penrose_nu"}, {{{1.0, 5.0}}},
                                    DataVector{num_pts});
  pypp::check_with_random_values<1>(&newman_penrose_mu, "NewmanPenrose",
                                    {"newman_penrose_mu"}, {{{1.0, 5.0}}},
                                    DataVector{num_pts});
  pypp::check_with_random_values<1>(&newman_penrose_lambda, "NewmanPenrose",
                                    {"newman_penrose_lambda"}, {{{1.0, 5.0}}},
                                    DataVector{num_pts});
}

void pypp_test_volume_weyl() {
  const pypp::SetupLocalPythonEnvironment
      local_python_env{"Evolution/Systems/Cce/"};

  const size_t num_pts = 5;

  pypp::check_with_random_values<1>(&(VolumeWeyl<Tags::Psi0>::apply),
                                    "NewmanPenrose", {"psi0"}, {{{1.0, 5.0}}},
                                    DataVector{num_pts});
  pypp::check_with_random_values<1>(&(VolumeWeyl<Tags::Psi1>::apply),
                                    "NewmanPenrose", {"psi1"}, {{{1.0, 5.0}}},
                                    DataVector{num_pts});
}
}  // namespace

namespace {

void test_np_spin_coefficient_compute_tags() {
  using ::TestHelpers::db::test_compute_tag;

  test_compute_tag<Cce::Tags::NewmanPenroseAlphaCompute>(
      "NewmanPenroseAlpha");
  test_compute_tag<Cce::Tags::NewmanPenroseBetaCompute>(
      "NewmanPenroseBeta");
  test_compute_tag<Cce::Tags::NewmanPenroseGammaCompute>(
      "NewmanPenroseGamma");
  test_compute_tag<Cce::Tags::NewmanPenroseEpsilonCompute>(
      "NewmanPenroseEpsilon");
  // In our choice of tetrad, \kappa=0
  test_compute_tag<Cce::Tags::NewmanPenroseTauCompute>(
      "NewmanPenroseTau");
  test_compute_tag<Cce::Tags::NewmanPenroseSigmaCompute>(
      "NewmanPenroseSigma");
  test_compute_tag<Cce::Tags::NewmanPenroseRhoCompute>(
      "NewmanPenroseRho");
  test_compute_tag<Cce::Tags::NewmanPenrosePiCompute>(
      "NewmanPenrosePi");
  test_compute_tag<Cce::Tags::NewmanPenroseNuCompute>(
      "NewmanPenroseNu");
  test_compute_tag<Cce::Tags::NewmanPenroseMuCompute>(
      "NewmanPenroseMu");
  test_compute_tag<Cce::Tags::NewmanPenroseLambdaCompute>(
      "NewmanPenroseLambda");
}

// This unit test is to validate the calculation of the Weyl scalar psi0 on the
// worldtube. The structure is in parallel with Test_GaugeTransformBoundaryData
// (most codes are copied from there). The test constructs a stationary Kerr
// spacetime in nontrivial time-dependent oscillating coordinates on both Cauchy
// and CCE grids. Then we compute psi0 on the worldtube. In principle, it should
// be consistent with 0.
template <typename Generator>
void compute_psi0_of_bh_on_wt(const gsl::not_null<Generator*> gen) {
  const size_t l_max = 12;
  const size_t number_of_radial_grid_points = 10;
  const size_t number_of_angular_grid_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  auto box = Cce::TestHelpers::create_cce_volume_box(
      gen, l_max, number_of_radial_grid_points, true);

  // Now we need to transform the boundary data of BondiJ and BondiR from the
  // Cauchy grid to the partially flat grid.
  const auto perform_gauge_adjustment = [&box](auto tag_v) {
    using tag = typename decltype(tag_v)::type;
    INFO("computing tag : " << db::tag_name<tag>());
    db::mutate_apply<GaugeAdjustedBoundaryValue<tag>>(make_not_null(&box));
  };

  using gauge_adjustments =
      tmpl::list<Tags::BondiR, Tags::BondiJ, Tags::Dr<Tags::BondiJ>>;
  tmpl::for_each<gauge_adjustments>(perform_gauge_adjustment);

  // Now we construct the volume data of BondiJ (on a null slice) based on its
  // boundary data.
  db::mutate_apply<Cce::TestHelpers::InverseCubicEvolutionGauge::mutate_tags,
                   Cce::TestHelpers::InverseCubicEvolutionGauge::argument_tags>(
      Cce::TestHelpers::InverseCubicEvolutionGauge{}, make_not_null(&box));

  // Then we compute psi0 on the worldtube
  db::mutate_apply<TransformBondiJToCauchyCoords>(make_not_null(&box));
  db::mutate_apply<PreSwshDerivatives<Tags::Dy<Tags::BondiJCauchyView>>>(
      make_not_null(&box));
  db::mutate_apply<
      PreSwshDerivatives<Tags::Dy<Tags::Dy<Tags::BondiJCauchyView>>>>(
      make_not_null(&box));
  db::mutate_apply<
      PrecomputeCceDependencies<Tags::BoundaryValue, Tags::OneMinusY>>(
      make_not_null(&box));
  db::mutate_apply<VolumeWeyl<Tags::Psi0Match>>(make_not_null(&box));
  db::mutate_apply<PreSwshDerivatives<Tags::Dy<Tags::Psi0Match>>>(
      make_not_null(&box));
  db::mutate_apply<InnerBoundaryWeyl>(make_not_null(&box));

  // Finally, we expect the results should be consistent with 0.
  const auto& psi0_wt = db::get<Tags::BoundaryValue<Tags::Psi0Match>>(box);
  SpinWeighted<ComplexDataVector, 2> psi0_desired{number_of_angular_grid_points,
                                                  0.0};
  Approx interpolation_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e6)
          .scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(get(psi0_wt).data(), psi0_desired.data(),
                               interpolation_approx);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.NewmanPenrose", "[Unit][Cce]") {
  test_np_spin_coefficient_compute_tags();
  pypp_test_volume_np_spin_coefficients();

  pypp_test_volume_weyl();

  MAKE_GENERATOR(gen);
  compute_psi0_of_bh_on_wt(make_not_null(&gen));
}
}  // namespace Cce
