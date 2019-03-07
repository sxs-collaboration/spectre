// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <memory>

#include "Domain/Creators/AlignedLattice.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "tests/Unit/Domain/DomainTestHelpers.hpp"
#include "tests/Unit/TestCreation.hpp"

// IWYU pragma: no_include <vector>

namespace Frame {
struct Inertial;
}  // namespace Frame

namespace domain {
namespace {
template <size_t VolumeDim>
void test_aligned_blocks(
    const creators::AlignedLattice<VolumeDim, Frame::Inertial>&
        aligned_blocks) noexcept {
  const auto domain = aligned_blocks.create_domain();
  test_initial_domain(domain, aligned_blocks.initial_refinement_levels());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.AlignedLattice", "[Domain][Unit]") {
  const auto domain_creator_1d =
      test_factory_creation<DomainCreator<1, Frame::Inertial>>(
          "  AlignedLattice:\n"
          "    BlockBounds: [[0.1, 2.6, 5.1, 5.2, 7.2]]\n"
          "    IsPeriodicIn: [false]\n"
          "    InitialGridPoints: [3]\n"
          "    InitialRefinement: [2]\n");
  const auto* aligned_blocks_creator_1d =
      dynamic_cast<const creators::AlignedLattice<1, Frame::Inertial>*>(
          domain_creator_1d.get());
  test_aligned_blocks(*aligned_blocks_creator_1d);

  const auto domain_creator_2d =
      test_factory_creation<DomainCreator<2, Frame::Inertial>>(
          "  AlignedLattice:\n"
          "    BlockBounds: [[0.1, 2.6, 5.1], [-0.4, 3.2, 6.2, 8.9]]\n"
          "    IsPeriodicIn: [false, true]\n"
          "    InitialGridPoints: [3, 4]\n"
          "    InitialRefinement: [2, 1]\n");
  const auto* aligned_blocks_creator_2d =
      dynamic_cast<const creators::AlignedLattice<2, Frame::Inertial>*>(
          domain_creator_2d.get());
  test_aligned_blocks(*aligned_blocks_creator_2d);

  const auto domain_creator_3d =
      test_factory_creation<DomainCreator<3, Frame::Inertial>>(
          "  AlignedLattice:\n"
          "    BlockBounds: [[0.1, 2.6, 5.1], [-0.4, 3.2, 6.2], [-0.2, 3.2]]\n"
          "    IsPeriodicIn: [false, true, false]\n"
          "    InitialGridPoints: [3, 4, 5]\n"
          "    InitialRefinement: [2, 1, 0]\n");
  const auto* aligned_blocks_creator_3d =
      dynamic_cast<const creators::AlignedLattice<3, Frame::Inertial>*>(
          domain_creator_3d.get());
  test_aligned_blocks(*aligned_blocks_creator_3d);

  const auto cubical_shell_domain =
      test_factory_creation<DomainCreator<3, Frame::Inertial>>(
          "  AlignedLattice:\n"
          "    BlockBounds: [[0.1, 2.6, 5.1, 6.0], [-0.4, 3.2, 6.2, 7.0], "
          "[-0.2, 3.2, 4.0, 5.2]]\n"
          "    IsPeriodicIn: [false, false, false]\n"
          "    InitialGridPoints: [3, 4, 5]\n"
          "    InitialRefinement: [2, 1, 0]\n"
          "    BlocksToExclude: [[1, 1, 1]]");
  const auto* cubical_shell_creator_3d =
      dynamic_cast<const creators::AlignedLattice<3, Frame::Inertial>*>(
          cubical_shell_domain.get());
  test_aligned_blocks(*cubical_shell_creator_3d);

  const auto unit_cubical_shell_domain =
      test_factory_creation<DomainCreator<3, Frame::Inertial>>(
          "  AlignedLattice:\n"
          "    BlockBounds: [[-1.5, -0.5, 0.5, 1.5], [-1.5, -0.5, 0.5, 1.5], "
          "[-1.5, -0.5, 0.5, 1.5]]\n"
          "    IsPeriodicIn: [false, false, false]\n"
          "    InitialGridPoints: [5, 5, 5]\n"
          "    InitialRefinement: [1, 1, 1]\n"
          "    BlocksToExclude: [[1, 1, 1]]");
  const auto* unit_cubical_shell_creator_3d =
      dynamic_cast<const creators::AlignedLattice<3, Frame::Inertial>*>(
          unit_cubical_shell_domain.get());
  test_aligned_blocks(*unit_cubical_shell_creator_3d);
}

// [[OutputRegex, Cannot exclude blocks as well as have periodic boundary
// conditions!]]
SPECTRE_TEST_CASE("Unit.Domain.Creators.AlignedLattice.Error",
                  "[Unit][ErrorHandling]") {
  ERROR_TEST();
  const auto failed_cubical_shell_domain =
      test_factory_creation<DomainCreator<3, Frame::Inertial>>(
          "  AlignedLattice:\n"
          "    BlockBounds: [[-1.5, -0.5, 0.5, 1.5], [-1.5, -0.5, 0.5, 1.5], "
          "[-1.5, -0.5, 0.5, 1.5]]\n"
          "    IsPeriodicIn: [true, false, false]\n"
          "    InitialGridPoints: [5, 5, 5]\n"
          "    InitialRefinement: [1, 1, 1]\n"
          "    BlocksToExclude: [[1, 1, 1]]");
}
}  // namespace domain
