// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include "DataStructures/Index.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Evolution/DgSubcell/StartsOnSubcellGrid.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.StartsOnSubcellGrid",
                  "[Evolution][Unit]") {
  using evolution::dg::subcell::starts_on_subcell_grid;
  using evolution::dg::subcell::subcell_allowed_in_element;
  // Three blocks in a row, each with 4 elements.
  const auto domain = rectilinear_domain<1>(
      Index<1>{3}, std::array<std::vector<double>, 1>{{{0.0, 1.0, 2.0, 3.0}}});
  const std::vector<std::array<size_t, 1>> refinement(3, {{2}});
  const auto element = [&domain, &refinement](const size_t block_id,
                                              const size_t index) {
    return domain::create_initial_element(
        ElementId<1>{block_id, {{SegmentId{2, index}}}}, domain.blocks(),
        refinement);
  };
  const Mesh<1> dg_mesh{5, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const std::vector<size_t> only_dg_block_ids{2};

  // Interior element of an FD-capable block.
  CHECK(subcell_allowed_in_element(element(1, 1), dg_mesh, only_dg_block_ids));
  // Element bordering a DG-only block.
  CHECK_FALSE(
      subcell_allowed_in_element(element(1, 3), dg_mesh, only_dg_block_ids));
  // Element in a DG-only block.
  CHECK_FALSE(
      subcell_allowed_in_element(element(2, 1), dg_mesh, only_dg_block_ids));
  // Mesh that doesn't support subcell.
  CHECK_FALSE(subcell_allowed_in_element(
      element(1, 1),
      Mesh<1>{5, Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular},
      only_dg_block_ids));

  // Element on the external boundary.
  CHECK(subcell_allowed_in_element(element(0, 0), dg_mesh, only_dg_block_ids));
  CHECK(
      starts_on_subcell_grid(element(0, 0), dg_mesh, only_dg_block_ids, true));
  CHECK_FALSE(
      starts_on_subcell_grid(element(0, 0), dg_mesh, only_dg_block_ids, false));
  // Interior elements don't depend on the external boundary setting.
  CHECK(
      starts_on_subcell_grid(element(1, 1), dg_mesh, only_dg_block_ids, false));
  CHECK_FALSE(
      starts_on_subcell_grid(element(1, 3), dg_mesh, only_dg_block_ids, true));
}
