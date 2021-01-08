// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Wedge2D.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/DiscontinuousGalerkin/LiftFromBoundary.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MetricIdentityJacobian.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/WeakDivergence.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct Var2 : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
};

template <typename TagsList, size_t Dim>
auto polynomial_volume_fluxes(
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
    const Index<Dim>& powers) noexcept {
  Variables<db::wrap_tags_in<Tags::Flux, TagsList, tmpl::size_t<Dim>,
                             Frame::Inertial>>
      result(get<0>(coords).size(), 1.0);

  if constexpr (tmpl::list_contains_v<TagsList, Var1>) {
    for (size_t i = 1; i < Dim; ++i) {
      get<Tags::Flux<Var1, tmpl::size_t<Dim>, Frame::Inertial>>(result).get(i) =
          0.0;
    }
  }

  for (size_t power_index = 0; power_index < Dim; ++power_index) {
    if constexpr (tmpl::list_contains_v<TagsList, Var1>) {
      get<0>(get<Tags::Flux<Var1, tmpl::size_t<Dim>, Frame::Inertial>>(
          result)) *= (1.0) * pow(coords.get(power_index), powers[power_index]);
    }
    if constexpr (tmpl::list_contains_v<TagsList, Var2<Dim>>) {
      for (size_t i = 0; i < Dim; ++i) {
        for (size_t j = 0; j < Dim; ++j) {
          get<Tags::Flux<Var2<Dim>, tmpl::size_t<Dim>, Frame::Inertial>>(result)
              .get(i, j) *= (i + 2.0) * (j + 3.0) *
                            pow(coords.get(power_index), powers[power_index]);
        }
      }
    }
  }
  return result;
}

using Affine = domain::CoordinateMaps::Affine;
using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

template <size_t VolumeDim>
auto make_map() noexcept;

template <>
auto make_map<1>() noexcept {
  return domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
      Affine{-1.0, 1.0, -0.3, 0.7});
}

template <>
auto make_map<2>() noexcept {
  return domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
      Affine2D{{-1.0, 1.0, -1.0, -0.9}, {-1.0, 1.0, -1.0, -0.9}},
      domain::CoordinateMaps::Wedge2D{1.0, 2.0, 0.0, 1.0, {}, false});
}

template <>
auto make_map<3>() noexcept {
  return domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
      Affine3D{{-1.0, 1.0, -1.0, -0.9},
               {-1.0, 1.0, -1.0, -0.9},
               {-1.0, 1.0, -1.0, 1.0}},
      domain::CoordinateMaps::ProductOf2Maps<domain::CoordinateMaps::Wedge2D,
                                             Affine>{
          {1.0, 2.0, 0.0, 1.0, {}, false}, {0.0, 1.0, 0.0, 1.0}});
}

template <size_t Dim>
void test(const double eps) {
  // Allow specifying either Var1 or Var2 so that debugging is easier. This test
  // will be useful when trying to better understand preserving the metric
  // identities numerically.
  //
  // Currently the weak form with the lifting test as coded here does not
  // satisfy the metric identities on curved meshes. This is likely because the
  // boundary term does not have the metric identity satisfying normal and
  // Jacobian terms.
  using tags = tmpl::list<Var1, Var2<Dim>>;
  Mesh<Dim> volume_mesh{8, Spectral::Basis::Legendre,
                        Spectral::Quadrature::Gauss};
  CAPTURE(Dim);
  CAPTURE(volume_mesh);

  Index<Dim> powers{};
  for (size_t i = 0; i < Dim; ++i) {
    powers[i] = volume_mesh.extents(i) - 4 - i;
  }

  const auto map = make_map<Dim>();
  const auto logical_coords = logical_coordinates(volume_mesh);
  const auto inertial_coords = map(logical_coords);
  const auto volume_inv_jacobian = map.inv_jacobian(logical_coords);
  const auto [volume_det_inv_jacobian, volume_jacobian] =
      determinant_and_inverse(volume_inv_jacobian);

  auto volume_fluxes = polynomial_volume_fluxes<tags>(inertial_coords, powers);

  InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>
      det_jac_times_inverse_jacobian{volume_mesh.number_of_grid_points()};
  dg::metric_identity_det_jac_times_inv_jac(
      make_not_null(&det_jac_times_inverse_jacobian), volume_mesh,
      inertial_coords, volume_jacobian);

  Variables<db::wrap_tags_in<
      Tags::div, typename std::decay_t<decltype(volume_fluxes)>::tags_list>>
      weak_div_fluxes{volume_mesh.number_of_grid_points()};
  weak_divergence(make_not_null(&weak_div_fluxes), volume_fluxes, volume_mesh,
                  det_jac_times_inverse_jacobian);

  Variables<db::wrap_tags_in<Tags::dt, tags>> dt_vars_lifted_one_at_a_time{
      volume_mesh.number_of_grid_points(), 0.0};
  Variables<db::wrap_tags_in<Tags::dt, tags>> dt_vars_lifted_two_at_a_time{
      volume_mesh.number_of_grid_points()};
  std::copy(weak_div_fluxes.data(),
            weak_div_fluxes.data() + weak_div_fluxes.size(),
            dt_vars_lifted_one_at_a_time.data());
  std::copy(weak_div_fluxes.data(),
            weak_div_fluxes.data() + weak_div_fluxes.size(),
            dt_vars_lifted_two_at_a_time.data());

  dt_vars_lifted_one_at_a_time *= get(volume_det_inv_jacobian);
  dt_vars_lifted_two_at_a_time *= get(volume_det_inv_jacobian);

  for (const auto& direction : Direction<Dim>::all_directions()) {
    // compute geometric terms
    const auto face_mesh = volume_mesh.slice_away(direction.dimension());
    const auto face_logical_coords =
        interface_logical_coordinates(face_mesh, direction);
    const auto face_inertial_coords = map(face_logical_coords);
    const auto face_det_jacobian =
        determinant(map.jacobian(face_logical_coords));
    const auto face_normal =
        unnormalized_face_normal(face_mesh, map, direction);
    const auto face_normal_magnitude = magnitude(face_normal);
    auto unit_face_normal = face_normal;
    for (auto& t : unit_face_normal) {
      t /= get(face_normal_magnitude);
    }

    const auto boundary_corrections = normal_dot_flux<tags>(
        unit_face_normal,
        polynomial_volume_fluxes<tags>(face_inertial_coords, powers));

    // Now perform lifting
    evolution::dg::lift_boundary_terms_gauss_points(
        make_not_null(&dt_vars_lifted_one_at_a_time), volume_det_inv_jacobian,
        volume_mesh, direction, boundary_corrections, face_normal_magnitude,
        face_det_jacobian);

    if (direction.side() == Side::Upper) {
      // Test lifting both upper and lower correction at the same time
      const auto lower_direction = direction.opposite();
      const auto lower_face_mesh =
          volume_mesh.slice_away(lower_direction.dimension());
      const auto lower_face_logical_coords =
          interface_logical_coordinates(lower_face_mesh, lower_direction);
      const auto lower_face_inertial_coords = map(lower_face_logical_coords);
      const auto lower_face_det_jacobian =
          determinant(map.jacobian(lower_face_logical_coords));
      const auto lower_face_normal =
          unnormalized_face_normal(lower_face_mesh, map, lower_direction);
      const auto lower_face_normal_magnitude = magnitude(lower_face_normal);
      auto lower_unit_face_normal = lower_face_normal;
      for (auto& t : lower_unit_face_normal) {
        t /= get(lower_face_normal_magnitude);
      }

      const auto lower_boundary_corrections = normal_dot_flux<tags>(
          lower_unit_face_normal,
          polynomial_volume_fluxes<tags>(lower_face_inertial_coords, powers));

      // Now perform lifting
      evolution::dg::lift_boundary_terms_gauss_points(
          make_not_null(&dt_vars_lifted_two_at_a_time), volume_det_inv_jacobian,
          volume_mesh, direction.dimension(), boundary_corrections,
          face_normal_magnitude, face_det_jacobian, lower_boundary_corrections,
          lower_face_normal_magnitude, lower_face_det_jacobian);
    }
  }

  Variables<db::wrap_tags_in<
      Tags::div, typename std::decay_t<decltype(volume_fluxes)>::tags_list>>
      div_fluxes{volume_mesh.number_of_grid_points()};
  divergence(make_not_null(&div_fluxes), volume_fluxes, volume_mesh,
             volume_inv_jacobian);
  Variables<db::wrap_tags_in<Tags::dt, tags>> expected_dt_vars{
      volume_mesh.number_of_grid_points()};
  std::copy(div_fluxes.data(), div_fluxes.data() + div_fluxes.size(),
            expected_dt_vars.data());
  expected_dt_vars *= -1.0;

  Approx local_approx = Approx::custom().epsilon(eps).scale(1.0);
  tmpl::for_each<typename decltype(expected_dt_vars)::tags_list>(
      [&dt_vars_lifted_one_at_a_time, &dt_vars_lifted_two_at_a_time,
       &expected_dt_vars, &local_approx](auto tag_v) noexcept {
        using tag = tmpl::type_from<decltype(tag_v)>;
        CHECK_ITERABLE_CUSTOM_APPROX(get<tag>(dt_vars_lifted_one_at_a_time),
                                     get<tag>(expected_dt_vars), local_approx);
        CHECK_ITERABLE_CUSTOM_APPROX(get<tag>(dt_vars_lifted_two_at_a_time),
                                     get<tag>(expected_dt_vars), local_approx);
      });
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.LiftFromBoundary", "[Unit][Evolution]") {
  // We test the lifting procedure (and at the same time the weak divergence
  // more thoroughly) by checking the
  //    weak divergence + lifting boundary terms == strong divergence
  //
  // Currently the weak form with the lifting test as coded here does not
  // satisfy the metric identities on curved meshes. This is likely because the
  // boundary term does not have the metric identity satisfying normal and
  // Jacobian terms.
  test<1>(1.0e-12);
  test<2>(1.0e-8);
  test<3>(1.0e-8);
}
