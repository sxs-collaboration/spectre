// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/DiscontinuousGalerkin/UsingSubcell.hpp"

namespace {
template <bool SubcellEnabled>
struct MetavarsSubcell {
  struct SubcellOptions {
    static constexpr bool subcell_enabled = SubcellEnabled;
  };
};

struct MetavarsNoSubcell {};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.UsingSubcell", "[Unit][Evolution]") {
  static_assert(not evolution::dg::using_subcell_v<MetavarsNoSubcell>);
  static_assert(not evolution::dg::using_subcell_v<MetavarsSubcell<false>>);
  static_assert(evolution::dg::using_subcell_v<MetavarsSubcell<true>>);

  CHECK(not evolution::dg::using_subcell_v<MetavarsNoSubcell>);
  CHECK(not evolution::dg::using_subcell_v<MetavarsSubcell<false>>);
  CHECK(evolution::dg::using_subcell_v<MetavarsSubcell<true>>);
}
