// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/DgSubcell/ReconstructionOrder.hpp"
#include "Evolution/DgSubcell/Tags/ReconstructionOrder.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace {
template <size_t Dim>
void test() {
  namespace subcell = evolution::dg::subcell;
  const Mesh<Dim> subcell_mesh{5, Spectral::Basis::FiniteDifference,
                               Spectral::Quadrature::CellCentered};

  // Test first with no data
  auto box =
      db::create<db::AddSimpleTags<subcell::Tags::Mesh<Dim>,
                                   subcell::Tags::ReconstructionOrder<Dim>>>(
          subcell_mesh,
          typename subcell::Tags::ReconstructionOrder<Dim>::type{});

  subcell::store_reconstruction_order_in_databox(
      make_not_null(&box),
      std::optional<std::array<gsl::span<std::uint8_t>, Dim>>{});

  CHECK(not db::get<subcell::Tags::ReconstructionOrder<Dim>>(box).has_value());

  std::optional<std::array<std::vector<std::uint8_t>, Dim>> recons_data{
      make_array<Dim>(std::vector<std::uint8_t>(
          (subcell_mesh.extents(0) + 2) *
          subcell_mesh.extents().slice_away(0).product()))};

  std::optional<std::array<gsl::span<std::uint8_t>, Dim>> recons_order{
      std::array<gsl::span<std::uint8_t>, Dim>{}};
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < gsl::at(recons_data.value(), i).size(); ++j) {
      gsl::at(recons_data.value(), i)[j] = static_cast<double>(i) + 5.0;
    }
    gsl::at(recons_order.value(), i) =
        gsl::make_span(gsl::at(recons_data.value(), i));
  }

  subcell::store_reconstruction_order_in_databox(make_not_null(&box),
                                                 recons_order);

  REQUIRE(db::get<subcell::Tags::ReconstructionOrder<Dim>>(box).has_value());
  for (size_t d = 0; d < Dim; ++d) {
    CHECK(db::get<subcell::Tags::ReconstructionOrder<Dim>>(box).value().get(
              d) == static_cast<double>(d) + 5.0);
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.ReconstructionOrder",
                  "[Evolution][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
