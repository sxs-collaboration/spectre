// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/JacobianDiagnostic.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"

// IWYU pragma: no_include "Utilities/Array.hpp"

namespace {
template <size_t Dim>
ElementMap<Dim, Frame::Grid> jac_diag_map_that_fits();

template <>
ElementMap<1, Frame::Grid> jac_diag_map_that_fits() {
  constexpr size_t dim = 1;
  const auto segment_ids = std::array<SegmentId, dim>({{SegmentId(0, 0)}});
  const ElementId<dim> element_id(0, segment_ids);
  const domain::CoordinateMaps::Affine map{-1.0, 1.0, 0.4, 5.5};
  return ElementMap<dim, Frame::Grid>{
      element_id,
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(map)};
}

template <>
ElementMap<2, Frame::Grid> jac_diag_map_that_fits() {
  constexpr size_t dim = 2;
  const auto segment_ids = std::array<SegmentId, dim>({{SegmentId(0, 0)}});
  const ElementId<dim> element_id(0, segment_ids);
  const domain::CoordinateMaps::Rotation<2> map{M_PI_4};
  return ElementMap<dim, Frame::Grid>{
      element_id,
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(map)};
}

template <>
ElementMap<3, Frame::Grid> jac_diag_map_that_fits() {
  constexpr size_t dim = 3;
  const auto segment_ids = std::array<SegmentId, dim>({{SegmentId(0, 0)}});
  const ElementId<dim> element_id(0, segment_ids);
  const domain::CoordinateMaps::Rotation<3> map{M_PI_4, M_PI_2, M_PI_2};
  return ElementMap<dim, Frame::Grid>{
      element_id,
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(map)};
}

template <size_t Dim>
ElementMap<Dim, Frame::Grid> jac_diag_generic_map();

template <>
ElementMap<1, Frame::Grid> jac_diag_generic_map() {
  constexpr size_t dim = 1;
  const auto segment_ids = std::array<SegmentId, dim>({{SegmentId(0, 0)}});
  const ElementId<dim> element_id(0, segment_ids);
  const domain::CoordinateMaps::Equiangular map{-1.0, 1.0, 0.4, 5.5};
  return ElementMap<dim, Frame::Grid>{
      element_id,
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(map)};
}

template <>
ElementMap<2, Frame::Grid> jac_diag_generic_map() {
  constexpr size_t dim = 2;
  const auto segment_ids = std::array<SegmentId, dim>({{SegmentId(0, 0)}});
  const ElementId<dim> element_id(0, segment_ids);
  using Equiangular = domain::CoordinateMaps::Equiangular;
  using Equiangular2D =
      domain::CoordinateMaps::ProductOf2Maps<Equiangular, Equiangular>;
  const Equiangular2D map{{-1.0, 1.0, 0.4, 5.5}, {-1.0, 1.0, -4.4, -0.2}};
  return ElementMap<dim, Frame::Grid>{
      element_id,
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(map)};
}

template <>
ElementMap<3, Frame::Grid> jac_diag_generic_map() {
  constexpr size_t dim = 3;
  const auto segment_ids = std::array<SegmentId, dim>({{SegmentId(0, 0)}});
  const ElementId<dim> element_id(0, segment_ids);
  using Equiangular = domain::CoordinateMaps::Equiangular;
  using Equiangular3D =
      domain::CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular,
                                             Equiangular>;
  const Equiangular3D map{
      {-1.0, 1.0, 0.4, 5.5}, {-1.0, 1.0, -4.4, -0.2}, {-1.0, 1.0, -5.0, 3.0}};
  return ElementMap<dim, Frame::Grid>{
      element_id,
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(map)};
}

template <size_t Dim, bool UseGenericMap>
void test_jacobian_diagnostic_databox() {
  if constexpr (UseGenericMap) {
    TestHelpers::db::test_compute_tag<
        domain::Tags::JacobianDiagnosticCompute<Dim, Frame::Grid>>(
        "JacobianDiagnostic");
  }

  auto setup_databox = [](const size_t points_per_dimension) {
    std::array<size_t, Dim> extents{};
    for (size_t i = 0; i < Dim; ++i) {
      extents[i] = points_per_dimension;
    }
    Mesh<Dim> mesh{extents, Spectral::Basis::Legendre,
                   Spectral::Quadrature::GaussLobatto};
    auto logical_coords = logical_coordinates(mesh);
    auto map = UseGenericMap ? jac_diag_generic_map<Dim>()
                             : jac_diag_map_that_fits<Dim>();

    return db::create<
        tmpl::list<domain::Tags::ElementMap<Dim, Frame::Grid>,
                   domain::Tags::Coordinates<Dim, Frame::ElementLogical>,
                   domain::Tags::Mesh<Dim>>,
        db::AddComputeTags<
            domain::Tags::MappedCoordinates<
                domain::Tags::ElementMap<Dim, Frame::Grid>,
                domain::Tags::Coordinates<Dim, Frame::ElementLogical>>,
            domain::Tags::InverseJacobianCompute<
                domain::Tags::ElementMap<Dim, Frame::Grid>,
                domain::Tags::Coordinates<Dim, Frame::ElementLogical>>,
            domain::Tags::JacobianCompute<Dim, Frame::ElementLogical,
                                          Frame::Grid>,
            domain::Tags::JacobianDiagnosticCompute<Dim, Frame::Grid>>>(
        std::move(map), std::move(logical_coords), std::move(mesh));
  };

  const auto box = setup_databox(5);
  const tnsr::i<DataVector, Dim, typename Frame::ElementLogical> jac_diag =
      db::get<domain::Tags::JacobianDiagnostic<Dim>>(box);

  if constexpr (UseGenericMap == false) {
    const tnsr::i<DataVector, Dim, typename Frame::ElementLogical>
        expected_jac_diag{jac_diag.begin()->size(), 0.0};
    // Map fits in the resolution provided, so the diagnostic should be
    // roundoff
    for (size_t i_hat = 0; i_hat < Dim; ++i_hat) {
      CHECK_ITERABLE_APPROX(jac_diag.get(i_hat), expected_jac_diag.get(i_hat));
    }
    return;
  }

  const auto box_high = setup_databox(7);
  const tnsr::i<DataVector, Dim, typename Frame::ElementLogical> jac_diag_high =
      db::get<domain::Tags::JacobianDiagnostic<Dim>>(box_high);
  for (size_t i_hat = 0; i_hat < Dim; ++i_hat) {
    // Check that higher resolution leads to significantly smaller diagnostic
    CHECK(max(abs(jac_diag_high.get(i_hat))) <
          0.125 * max(abs(jac_diag.get(i_hat))));
  }

  const auto jac =
      db::get<domain::Tags::Jacobian<Dim, Frame::ElementLogical, Frame::Grid>>(
          box);
  const auto mapped_coords = db::get<domain::Tags::MappedCoordinates<
      domain::Tags::ElementMap<Dim, Frame::Grid>,
      domain::Tags::Coordinates<Dim, Frame::ElementLogical>>>(box);
  const auto mesh = db::get<domain::Tags::Mesh<Dim>>(box);
  const auto jac_diag_by_value =
      domain::jacobian_diagnostic(jac, mapped_coords, mesh);
  CHECK_ITERABLE_APPROX(jac_diag, jac_diag_by_value);
}

template <size_t Dim, typename Fr>
void test_jacobian_diagnostic_random() {
  pypp::check_with_random_values<1>(
      static_cast<void (*)(
          const gsl::not_null<
              tnsr::i<DataVector, Dim, typename Frame::ElementLogical>*>,
          const Jacobian<DataVector, Dim, typename Frame::ElementLogical, Fr>&,
          const TensorMetafunctions::prepend_spatial_index<
              tnsr::I<DataVector, Dim, Fr>, Dim, UpLo::Lo,
              typename Frame::ElementLogical>&)>(
          &domain::jacobian_diagnostic<Dim, Fr>),
      "JacobianDiagnostic", {"jacobian_diagnostic"}, {{{-1.0, 1.0}}},
      DataVector(5));
}
}  // namespace
SPECTRE_TEST_CASE("Unit.Domain.JacobianDiagnostic", "[Domain][Unit]") {
  TestHelpers::db::test_simple_tag<domain::Tags::JacobianDiagnostic<1>>(
      "JacobianDiagnostic");
  TestHelpers::db::test_simple_tag<domain::Tags::JacobianDiagnostic<2>>(
      "JacobianDiagnostic");
  TestHelpers::db::test_simple_tag<domain::Tags::JacobianDiagnostic<3>>(
      "JacobianDiagnostic");

  pypp::SetupLocalPythonEnvironment local_python_env{"Domain/"};

  test_jacobian_diagnostic_databox<1, false>();
  test_jacobian_diagnostic_databox<2, false>();
  test_jacobian_diagnostic_databox<3, false>();
  test_jacobian_diagnostic_databox<1, true>();
  test_jacobian_diagnostic_databox<2, true>();
  test_jacobian_diagnostic_databox<3, true>();

  test_jacobian_diagnostic_random<1, Frame::Grid>();
  test_jacobian_diagnostic_random<2, Frame::Grid>();
  test_jacobian_diagnostic_random<3, Frame::Grid>();
  test_jacobian_diagnostic_random<1, Frame::Inertial>();
  test_jacobian_diagnostic_random<2, Frame::Inertial>();
  test_jacobian_diagnostic_random<3, Frame::Inertial>();
}
