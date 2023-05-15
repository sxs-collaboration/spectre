// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <catch.hpp>
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
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct Var2 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct TestSystem {
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

template <size_t Dim, bool ComputeOnlyOnRollback>
void test(const fd::DerivativeOrder derivative_order, const bool did_rollback) {
  CAPTURE(Dim);
  CAPTURE(ComputeOnlyOnRollback);
  CAPTURE(derivative_order);
  CAPTURE(did_rollback);
  using flux_variables = typename TestSystem::flux_variables;
  using CellCenteredFluxTag =
      evolution::dg::subcell::Tags::CellCenteredFlux<flux_variables, Dim>;
  const Mesh<Dim> dg_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh{9, Spectral::Basis::FiniteDifference,
                               Spectral::Quadrature::CellCentered};
  const std::optional<tnsr::I<DataVector, Dim>> dg_mesh_velocity{};

  auto box =
      db::create<tmpl::list<evolution::dg::subcell::Tags::DidRollback,
                            evolution::dg::subcell::Tags::SubcellOptions<Dim>,
                            evolution::dg::subcell::Tags::Mesh<Dim>,
                            CellCenteredFluxTag, domain::Tags::Mesh<Dim>,
                            domain::Tags::MeshVelocity<Dim, Frame::Inertial>,
                            ::Tags::Variables<flux_variables>>>(
          did_rollback,
          evolution::dg::subcell::SubcellOptions{
              1.0e-3, 1.0e-4, 2.0e-3, 2.0e-4, 4.0, 4.1, false,
              evolution::dg::subcell::fd::ReconstructionMethod::DimByDim, false,
              std::nullopt, derivative_order},
          subcell_mesh, typename CellCenteredFluxTag::type{}, dg_mesh,
          dg_mesh_velocity,
          Variables<flux_variables>{subcell_mesh.number_of_grid_points(), 1.0});

  db::mutate_apply<evolution::dg::subcell::fd::CellCenteredFlux<
      TestSystem, Fluxes<Dim>, Dim, ComputeOnlyOnRollback>>(
      make_not_null(&box));
  if (derivative_order != fd::DerivativeOrder::Two and
      (not ComputeOnlyOnRollback or did_rollback)) {
    REQUIRE(get<evolution::dg::subcell::Tags::CellCenteredFlux<flux_variables,
                                                               Dim>>(box)
                .has_value());
    const auto& [flux1, flux2] = get<CellCenteredFluxTag>(box).value();
    const auto& [var1, var2] = get<::Tags::Variables<flux_variables>>(box);
    for (size_t i = 0; i < Dim; ++i) {
      CHECK(flux1.get(i) ==
            DataVector((1.0 + static_cast<double>(i)) * get(var1)));
      CHECK(flux2.get(i) ==
            DataVector(5.0 * (1.0 + static_cast<double>(i)) * get(var2)));
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
      test<1, false>(derivative_order, did_rollback);
      test<2, false>(derivative_order, did_rollback);
      test<3, false>(derivative_order, did_rollback);

      test<1, true>(derivative_order, did_rollback);
      test<2, true>(derivative_order, did_rollback);
      test<3, true>(derivative_order, did_rollback);
    }
  }
}
}  // namespace
