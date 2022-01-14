// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"

namespace {
template <size_t Dim>
void test(const Mesh<Dim>& mesh) {
  const auto box =
      db::create<db::AddSimpleTags<domain::Tags::Mesh<Dim>>,
                 db::AddComputeTags<Events::Tags::ObserverMeshCompute<Dim>>>(
          mesh);
  CHECK(db::get<Events::Tags::ObserverMesh<Dim>>(box) == mesh);
  TestHelpers::db::test_simple_tag<Events::Tags::ObserverMesh<Dim>>(
      "ObserverMesh");
  TestHelpers::db::test_compute_tag<Events::Tags::ObserverMeshCompute<Dim>>(
      "ObserverMesh");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.Events.Tags", "[Unit]") {
  test(Mesh<1>{3, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto});
  test(Mesh<2>{3, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto});
  test(Mesh<3>{3, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto});
}
