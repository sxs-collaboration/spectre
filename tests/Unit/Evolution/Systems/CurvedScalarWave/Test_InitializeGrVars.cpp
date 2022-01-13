// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <limits>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Initialize.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace {

template <typename BackgroundSolution>
void test_initialize_gr_vars(const BackgroundSolution& analytic_solution,
                             const gsl::not_null<std::mt19937*> generator) {
  static constexpr size_t Dim = BackgroundSolution::volume_dim;
  using gr_vars_tag =
      typename CurvedScalarWave::System<Dim>::spacetime_variables_tag;
  using GrVars = typename gr_vars_tag::type;
  const size_t num_points = 42;
  std::uniform_real_distribution dist{-10., 10.};
  const auto random_coords = make_with_random_values<tnsr::I<DataVector, Dim>>(
      generator, make_not_null(&dist), DataVector{num_points});
  auto box = db::create<
      db::AddSimpleTags<gr_vars_tag, ::Initialization::Tags::InitialTime,
                        domain::Tags::Coordinates<Dim, Frame::Inertial>,
                        ::Tags::AnalyticSolution<BackgroundSolution>>>(
      GrVars{}, 0., random_coords, analytic_solution);
  db::mutate_apply<CurvedScalarWave::Initialization::InitializeGrVars<Dim>>(
      make_not_null(&box));
  const auto sol = analytic_solution.variables(random_coords, 0.,
                                               typename GrVars::tags_list{});
  tmpl::for_each<typename GrVars::tags_list>([&box, &sol](auto tag_v) {
    using tag = typename decltype(tag_v)::type;
    CHECK(get<tag>(box) == get<tag>(sol));
  });
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.CurvedScalarWave.InitializeGrVars",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);
  test_initialize_gr_vars(gr::Solutions::Minkowski<1>(),
                          make_not_null(&generator));
  test_initialize_gr_vars(gr::Solutions::Minkowski<2>(),
                          make_not_null(&generator));
  test_initialize_gr_vars(gr::Solutions::Minkowski<3>(),
                          make_not_null(&generator));
  test_initialize_gr_vars(
      gr::Solutions::KerrSchild(1., {0.5, 0., 0.1}, {0.2, 0.5, -0.7}),
      make_not_null(&generator));
}
