// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/DgSubcell/CellCenteredFlux.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/CellCenteredFlux.hpp"
#include "Evolution/DgSubcell/Tags/DidRollback.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct Var2 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct Var3 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct TestConservativeSystem {
  using variables_tag = ::Tags::Variables<tmpl::list<Var1, Var2>>;
  using flux_variables = tmpl::list<Var1, Var2>;
};

struct TestMixedSystem {
  using variables_tag = ::Tags::Variables<tmpl::list<Var1, Var2, Var3>>;
  using flux_variables = tmpl::list<Var1, Var2>;
};

template <size_t Dim>
struct Fluxes {
  using return_tags =
      tmpl::list<::Tags::Flux<Var1, tmpl::size_t<Dim>, Frame::Inertial>,
                 ::Tags::Flux<Var2, tmpl::size_t<Dim>, Frame::Inertial>>;
  using argument_tags = tmpl::list<Var1, Var2>;

  static void apply(const gsl::not_null<tnsr::I<DataVector, Dim>*> flux_var1,
                    const gsl::not_null<tnsr::I<DataVector, Dim>*> flux_var2,
                    const Scalar<DataVector>& var1,
                    const Scalar<DataVector>& var2) {
    for (size_t i = 0; i < Dim; ++i) {
      flux_var1->get(i) = (1.0 + static_cast<double>(i)) * get(var1);
      flux_var2->get(i) = 5.0 * (1.0 + static_cast<double>(i)) * get(var2);
    }
  }
};

template <typename TestSystem, size_t Dim, bool ComputeOnlyOnRollback>
void test(const fd::DerivativeOrder derivative_order, const bool did_rollback) {
  CAPTURE(Dim);
  CAPTURE(ComputeOnlyOnRollback);
  CAPTURE(derivative_order);
  CAPTURE(did_rollback);
  using variables = typename TestSystem::variables_tag::tags_list;
  using flux_variables = typename TestSystem::flux_variables;
  using CellCenteredFluxTag =
      evolution::dg::subcell::Tags::CellCenteredFlux<flux_variables, Dim>;
  const Mesh<Dim> dg_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh{9, Spectral::Basis::FiniteDifference,
                               Spectral::Quadrature::CellCentered};
  const std::optional<tnsr::I<DataVector, Dim>> dg_mesh_velocity{};
  Variables<variables> vars{subcell_mesh.number_of_grid_points()};
  get(get<Var1>(vars)) = 1.0;
  get(get<Var2>(vars)) = 2.0;

  auto box =
      db::create<tmpl::list<evolution::dg::subcell::Tags::DidRollback,
                            evolution::dg::subcell::Tags::SubcellOptions<Dim>,
                            evolution::dg::subcell::Tags::Mesh<Dim>,
                            CellCenteredFluxTag, domain::Tags::Mesh<Dim>,
                            domain::Tags::MeshVelocity<Dim, Frame::Inertial>,
                            ::Tags::Variables<variables>>>(
          did_rollback,
          evolution::dg::subcell::SubcellOptions{
              4.0, 1_st, 1.0e-3, 1.0e-4, false, false,
              evolution::dg::subcell::fd::ReconstructionMethod::DimByDim, false,
              std::nullopt, derivative_order, 1, 1, 1},
          subcell_mesh, typename CellCenteredFluxTag::type{}, dg_mesh,
          dg_mesh_velocity,
          Variables<variables>{subcell_mesh.number_of_grid_points(), 1.0});

  db::mutate_apply<evolution::dg::subcell::fd::CellCenteredFlux<
      TestSystem, Fluxes<Dim>, Dim, ComputeOnlyOnRollback>>(
      make_not_null(&box));
  if (derivative_order != fd::DerivativeOrder::Two and
      (not ComputeOnlyOnRollback or did_rollback)) {
    REQUIRE(get<evolution::dg::subcell::Tags::CellCenteredFlux<flux_variables,
                                                               Dim>>(box)
                .has_value());
    const auto& [flux1, flux2] = get<CellCenteredFluxTag>(box).value();
    const auto& box_vars = get<::Tags::Variables<variables>>(box);
    for (size_t i = 0; i < Dim; ++i) {
      CHECK(flux1.get(i) == DataVector((1.0 + static_cast<double>(i)) *
                                       get(get<Var1>(box_vars))));
      CHECK(flux2.get(i) == DataVector(5.0 * (1.0 + static_cast<double>(i)) *
                                       get(get<Var2>(box_vars))));
    }
  } else {
    CHECK(not get<evolution::dg::subcell::Tags::CellCenteredFlux<flux_variables,
                                                                 Dim>>(box)
                  .has_value());
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.CellCenteredFlux",
                  "[Evolution][Unit]") {
  using DO = fd::DerivativeOrder;
  for (const DO derivative_order :
       {DO::Two, DO::Four, DO::Six, DO::Eight, DO::Ten, DO::OneHigherThanRecons,
        DO::OneHigherThanReconsButFiveToFour}) {
    for (const bool did_rollback : {true, false}) {
      test<TestConservativeSystem, 1, false>(derivative_order, did_rollback);
      test<TestConservativeSystem, 2, false>(derivative_order, did_rollback);
      test<TestConservativeSystem, 3, false>(derivative_order, did_rollback);

      test<TestConservativeSystem, 1, true>(derivative_order, did_rollback);
      test<TestConservativeSystem, 2, true>(derivative_order, did_rollback);
      test<TestConservativeSystem, 3, true>(derivative_order, did_rollback);

      test<TestMixedSystem, 1, false>(derivative_order, did_rollback);
      test<TestMixedSystem, 2, false>(derivative_order, did_rollback);
      test<TestMixedSystem, 3, false>(derivative_order, did_rollback);

      test<TestMixedSystem, 1, true>(derivative_order, did_rollback);
      test<TestMixedSystem, 2, true>(derivative_order, did_rollback);
      test<TestMixedSystem, 3, true>(derivative_order, did_rollback);
    }
  }
}
}  // namespace
