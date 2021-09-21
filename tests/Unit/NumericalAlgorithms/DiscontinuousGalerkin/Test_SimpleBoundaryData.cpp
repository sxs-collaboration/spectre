// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/OrientationMapHelpers.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace {

struct SomeField : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct ExtraData {
  using type = int;
};

}  // namespace

SPECTRE_TEST_CASE("Unit.DG.SimpleBoundaryData", "[Unit][NumericalAlgorithms]") {
  const size_t num_points = 5;
  dg::SimpleBoundaryData<tmpl::list<SomeField>, tmpl::list<ExtraData>> data{
      num_points};
  const Scalar<DataVector> field{num_points, 1.};
  get<SomeField>(data.field_data) = field;
  get<ExtraData>(data.extra_data) = 2;

  // Test serialization
  data = serialize_and_deserialize(data);
  CHECK(get<SomeField>(data.field_data) == field);
  CHECK(get<ExtraData>(data.extra_data) == 2);

  // Test projections
  const Mesh<1> face_mesh{num_points, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<1> mortar_mesh{num_points + 1, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto};
  const std::array<Spectral::MortarSize, 1> mortar_size{
      {Spectral::MortarSize::UpperHalf}};
  const auto projected_data =
      data.project_to_mortar(face_mesh, mortar_mesh, mortar_size);
  CHECK(projected_data.field_data ==
        dg::project_to_mortar(data.field_data, face_mesh, mortar_mesh,
                              mortar_size));
  CHECK(projected_data.extra_data == data.extra_data);

  // Test orientation
  const size_t sliced_dim = 1;
  const auto slice_extents = face_mesh.extents();
  const OrientationMap<2> orientation_of_neighbor{
      {{Direction<2>::lower_xi(), Direction<2>::lower_eta()}},
      {{Direction<2>::upper_eta(), Direction<2>::lower_xi()}}};
  auto oriented_data = data;
  oriented_data.orient_on_slice(slice_extents, sliced_dim,
                                orientation_of_neighbor);
  CHECK(oriented_data.field_data ==
        orient_variables_on_slice(data.field_data, slice_extents, sliced_dim,
                                  orientation_of_neighbor));
  CHECK(oriented_data.extra_data == data.extra_data);
}
