// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <unordered_set>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/BulgedCube.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Frustum.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/CartesianFluxDivergence.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/GhostZoneInverseJacobian.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/Tags/GhostZoneInverseJacobian.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"
#include "NumericalAlgorithms/FiniteDifference/HighOrderFluxCorrection.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace {
struct Scalar0 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct Vector0 : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
};

class DummyReconstructor {
 public:
  explicit DummyReconstructor(const size_t ghost_zone_size)
      : ghost_zone_size_(ghost_zone_size){};
  size_t ghost_zone_size_;
  size_t ghost_zone_size() const { return ghost_zone_size_; }
};

namespace Tags {
struct Reconstructor : db::SimpleTag {
  using type = std::unique_ptr<DummyReconstructor>;
};
}  // namespace Tags

template <size_t Dim, bool AlignedCoordinates>
using CoordinateMap = tmpl::conditional_t<
    Dim == 1 or AlignedCoordinates, domain::CoordinateMaps::Identity<Dim>,
    tmpl::conditional_t<Dim == 2 and not AlignedCoordinates,
                        domain::CoordinateMaps::Wedge<2>,
                        domain::CoordinateMaps::BulgedCube>>;

template <size_t Dim, bool AlignedCoordinates>
double test(const fd::DerivativeOrder correction_order) {
  CAPTURE(AlignedCoordinates);
  CAPTURE(Dim);
  if constexpr (not AlignedCoordinates) {
    ASSERT(Dim == 2 or Dim == 3,
           "A test for a non-aligned grid is not provided for 1-D.");
  };
  const size_t max_degree =
      correction_order == fd::DerivativeOrder::OneHigherThanRecons
          ? 6
          : (correction_order ==
                     fd::DerivativeOrder::OneHigherThanReconsButFiveToFour
                 ? 4
                 : static_cast<size_t>(correction_order));
  const size_t points_per_dimension = static_cast<size_t>(max_degree) + 2;
  const size_t stencil_width = max_degree + 1;
  const size_t number_of_ghost_points = (stencil_width - 1) / 2 + 1;
  CAPTURE(points_per_dimension);

  using FluxTags = tmpl::list<Scalar0, Vector0<Dim>>;
  using Scalar0Flux = ::Tags::Flux<Scalar0, tmpl::size_t<Dim>, Frame::Inertial>;
  using Vector0Flux =
      ::Tags::Flux<Vector0<Dim>, tmpl::size_t<Dim>, Frame::Inertial>;
  using FluxVars =
      Variables<db::wrap_tags_in<::Tags::Flux, FluxTags, tmpl::size_t<Dim>,
                                 Frame::Inertial>>;
  using CorrectionVars = Variables<FluxTags>;

  const Mesh<Dim> dg_mesh{points_per_dimension, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh{points_per_dimension,
                               Spectral::Basis::FiniteDifference,
                               Spectral::Quadrature::CellCentered};
  const auto logical_coords = logical_coordinates(subcell_mesh);
  const auto dg_logical_coords = logical_coordinates(dg_mesh);

  CoordinateMap<Dim, AlignedCoordinates> coordinate_map;
  if constexpr (Dim == 1 or AlignedCoordinates) {
    coordinate_map = domain::CoordinateMaps::Identity<Dim>();
  } else if constexpr (Dim == 2 and not AlignedCoordinates) {
    coordinate_map = domain::CoordinateMaps::Wedge<2>(
        1., 4., 1., 1., OrientationMap<2>::create_aligned(), true,
        domain::CoordinateMaps::Wedge<2>::WedgeHalves::Both,
        domain::CoordinateMaps::Distribution::Linear,
        std::array<double, 1>{M_PI_2});
  } else if constexpr (Dim == 3 and not AlignedCoordinates) {
    coordinate_map = domain::CoordinateMaps::BulgedCube(1., 0.1, false);
  }

  ElementId<Dim> element_id{};
  if constexpr (AlignedCoordinates) {
    element_id = ElementId<Dim>{0, 0};
  } else if constexpr (Dim == 2 and not AlignedCoordinates) {
    element_id = ElementId<2>{0, {SegmentId{3, 4}, SegmentId{3, 4}}};
  } else if constexpr (Dim == 3 and not AlignedCoordinates) {
    element_id =
        ElementId<3>{0, {SegmentId{3, 4}, SegmentId{3, 4}, SegmentId{3, 4}}};
  }

  const auto element_map = ElementMap<Dim, Frame::Grid>(
      element_id,
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
          coordinate_map));

  const auto grid_coords = (element_map)(logical_coords);
  const auto jacobian = element_map.jacobian(logical_coords);
  const auto inv_jacobian = element_map.inv_jacobian(logical_coords);
  const auto det_inv_jacobian = determinant(inv_jacobian);

  typename evolution::dg::subcell::Tags::GhostZoneInverseJacobian<Dim>::type
      ghost_zone_inv_jac{};

  const DummyReconstructor dummy_reconstructor(number_of_ghost_points);
  evolution::dg::subcell::GhostZoneInverseJacobian<
      Dim, DummyReconstructor>::apply(make_not_null(&ghost_zone_inv_jac),
                                      subcell_mesh, element_map,
                                      dummy_reconstructor);

  // Compute polynomial on cell centers in FD cluster of points
  const auto set_polynomial = Overloader{
      [max_degree](const gsl::not_null<FluxVars*> vars_ptr,
                   const auto& local_coords) {
        (void)max_degree;
        for (size_t storage_index = 0;
             storage_index < get<Scalar0Flux>(*vars_ptr).size();
             ++storage_index) {
          get<Scalar0Flux>(*vars_ptr)[storage_index] = 0.0;
          for (size_t degree = 1; degree <= max_degree; ++degree) {
            for (size_t i = 0; i < Dim; ++i) {
              get<Scalar0Flux>(*vars_ptr)[storage_index] +=
                  pow(local_coords.get(i), degree);
            }
          }
        }
        for (size_t storage_index = 0;
             storage_index < get<Vector0Flux>(*vars_ptr).size();
             ++storage_index) {
          get<Vector0Flux>(*vars_ptr)[storage_index] = 0.0;
          for (size_t degree = 1; degree <= max_degree; ++degree) {
            for (size_t i = 0; i < Dim; ++i) {
              get<Vector0Flux>(*vars_ptr)[storage_index] +=
                  pow(local_coords.get(i), degree);
            }
          }
        }
      },
      [max_degree](const gsl::not_null<CorrectionVars*> vars_ptr,
                   const auto& local_coords) {
        (void)max_degree;
        for (size_t storage_index = 0;
             storage_index < get<Scalar0>(*vars_ptr).size(); ++storage_index) {
          get<Scalar0>(*vars_ptr)[storage_index] = 0.0;
          for (size_t degree = 1; degree <= max_degree; ++degree) {
            for (size_t i = 0; i < Dim; ++i) {
              get<Scalar0>(*vars_ptr)[storage_index] +=
                  pow(local_coords.get(i), degree);
            }
          }
        }
        for (size_t storage_index = 0;
             storage_index < get<Vector0<Dim>>(*vars_ptr).size();
             ++storage_index) {
          get<Vector0<Dim>>(*vars_ptr)[storage_index] = 0.0;
          for (size_t degree = 1; degree <= max_degree; ++degree) {
            for (size_t i = 0; i < Dim; ++i) {
              get<Vector0<Dim>>(*vars_ptr)[storage_index] +=
                  pow(local_coords.get(i), degree);
            }
          }
        }
      }};
  const auto set_polynomial_divergence =
      [max_degree](const gsl::not_null<CorrectionVars*> d_vars_ptr,
                   const auto& local_coords) {
        (void)max_degree;
        get(get<Scalar0>(*d_vars_ptr)) = 0.0;
        for (size_t i = 0; i < Dim; ++i) {
          // constant deriv is zero
          get<Vector0<Dim>>(*d_vars_ptr).get(i) = 0.0;
        }
        // Compute divergence
        for (size_t deriv_dim = 0; deriv_dim < Dim; ++deriv_dim) {
          for (size_t degree = 1; degree <= max_degree; ++degree) {
            get(get<Scalar0>(*d_vars_ptr)) +=
                degree * pow(local_coords.get(deriv_dim), degree - 1);
            for (size_t i = 0; i < Dim; ++i) {
              get<Vector0<Dim>>(*d_vars_ptr).get(i) +=
                  degree * pow(local_coords.get(deriv_dim), degree - 1);
            }
          }
        }
      };
  std::optional<FluxVars> volume_vars(subcell_mesh.number_of_grid_points());
  set_polynomial(&(volume_vars.value()), grid_coords);

  CorrectionVars expected_divergence(subcell_mesh.number_of_grid_points());
  set_polynomial_divergence(&expected_divergence, grid_coords);

  // Compute the polynomial at the cell center for the neighbor data that we
  // "received".
  //
  // We do this by computing the solution in our entire neighbor, then using
  // slice_data to get the subset of points that are needed.
  DirectionMap<Dim, FluxVars> neighbor_data{};
  DirectionalIdMap<Dim, evolution::dg::subcell::GhostData>
      reconstruction_ghost_data{};

  for (const auto& direction : Direction<Dim>::all_directions()) {
    auto neighbor_logical_coords = logical_coords;
    neighbor_logical_coords.get(direction.dimension()) +=
        direction.sign() * 2.0;
    const auto neighbor_grid_coords = (element_map)(neighbor_logical_coords);
    FluxVars neighbor_vars(subcell_mesh.number_of_grid_points(), 0.0);
    set_polynomial(&neighbor_vars, neighbor_grid_coords);

    const auto sliced_data = evolution::dg::subcell::slice_data(
        neighbor_vars, subcell_mesh.extents(), number_of_ghost_points,
        std::unordered_set{direction.opposite()}, 0, {});
    CAPTURE(number_of_ghost_points);
    REQUIRE(sliced_data.size() == 1);
    REQUIRE(sliced_data.contains(direction.opposite()));
    REQUIRE(sliced_data.at(direction.opposite()).size() %
                FluxVars::number_of_independent_components ==
            0);
    neighbor_data[direction].initialize(
        sliced_data.at(direction.opposite()).size() /
        FluxVars::number_of_independent_components);
    std::copy(sliced_data.at(direction.opposite()).begin(),
              sliced_data.at(direction.opposite()).end(),
              neighbor_data[direction].data());

    const DirectionalId<Dim> mortar_id{direction, ElementId<Dim>{0}};
    reconstruction_ghost_data[mortar_id] = evolution::dg::subcell::GhostData{1};
    reconstruction_ghost_data[mortar_id]
        .neighbor_ghost_data_for_reconstruction() =
        DataVector{sliced_data.at(direction.opposite()).size()};
    std::copy(sliced_data.at(direction.opposite()).begin(),
              sliced_data.at(direction.opposite()).end(),
              reconstruction_ghost_data[mortar_id]
                  .neighbor_ghost_data_for_reconstruction()
                  .data());
  }

  std::array<CorrectionVars, Dim> second_order_corrections{};
  std::array<
      InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Grid>, Dim>
      correction_inv_jacobians{};
  std::array<Scalar<DataVector>, Dim> correction_det_inv_jacobians{};

  for (size_t i = 0; i < Dim; ++i) {
    // Compare to analytic solution on the faces.
    const auto basis = make_array<Dim>(Spectral::Basis::FiniteDifference);
    auto quadrature = make_array<Dim>(Spectral::Quadrature::CellCentered);
    auto extents = make_array<Dim>(points_per_dimension);
    gsl::at(extents, i) = points_per_dimension + 1;
    gsl::at(quadrature, i) = Spectral::Quadrature::FaceCentered;
    const Mesh<Dim> face_centered_mesh{extents, basis, quadrature};
    auto face_logical_coords = logical_coordinates(face_centered_mesh);
    const auto face_grid_coords = (element_map)(face_logical_coords);

    gsl::at(second_order_corrections, i)
        .initialize(face_centered_mesh.number_of_grid_points());
    set_polynomial(make_not_null(&gsl::at(second_order_corrections, i)),
                   face_grid_coords);

    gsl::at(correction_inv_jacobians, i) =
        element_map.inv_jacobian(face_logical_coords);
    gsl::at(correction_det_inv_jacobians, i) =
        determinant(gsl::at(correction_inv_jacobians, i));

    // We use n_i F^i in the code, so need to negate to get sign to agree.
    gsl::at(second_order_corrections, i) *= -1.0;
  }

  // If not using an aligned coordinate map, we must convert the boundary term
  // to a local Cartesian flux on the logical grid. Assuming a flat spacetime,
  // we do this by computing G^{(\hat{i})} = J \pdv{\xi^\hat{i}}{x^j} G^{(i)}.
  // In this case, we are `cheating' a bit because the polynomial is defined
  // such that the flux in each direction of the grid frame is the same.
  // This is exploited in the coordinate transformation.
  if (not AlignedCoordinates) {
    const auto grid_second_order_corrections = second_order_corrections;
    for (size_t storage_index = 0;
         storage_index <
         get(get<Scalar0>(gsl::at(second_order_corrections, 0))).size();
         ++storage_index) {
      for (size_t i = 0; i < Dim; ++i) {
        get(get<Scalar0>(gsl::at(second_order_corrections, i)))[storage_index] =
            0.;
        for (size_t j = 0; j < Dim; ++j) {
          get(get<Scalar0>(
              gsl::at(second_order_corrections, i)))[storage_index] +=
              gsl::at(correction_inv_jacobians, i).get(i, j)[storage_index] *
              get(get<Scalar0>(
                  gsl::at(grid_second_order_corrections, i)))[storage_index] /
              get(gsl::at(correction_det_inv_jacobians, i))[storage_index];
          get<Vector0<Dim>>(gsl::at(second_order_corrections, i))
              .get(j)[storage_index] = 0.;
          for (size_t k = 0; k < Dim; ++k) {
            get<Vector0<Dim>>(gsl::at(second_order_corrections, i))
                .get(j)[storage_index] +=
                gsl::at(correction_inv_jacobians, i).get(i, k)[storage_index] *
                get<Vector0<Dim>>(gsl::at(grid_second_order_corrections, i))
                    .get(k)[storage_index] /
                get(gsl::at(correction_det_inv_jacobians, i))[storage_index];
          }
        }
      }
    }
  }

  std::array<std::vector<std::uint8_t>, Dim> reconstruction_order_storage{};
  std::array<gsl::span<std::uint8_t>, Dim> reconstruction_order{};
  if (correction_order == fd::DerivativeOrder::OneHigherThanRecons or
      correction_order ==
          fd::DerivativeOrder::OneHigherThanReconsButFiveToFour) {
    Index<Dim> recons_extents = subcell_mesh.extents();
    recons_extents[0] += 2;
    for (size_t i = 0; i < Dim; ++i) {
      gsl::at(reconstruction_order_storage, i) =
          std::vector<std::uint8_t>(recons_extents.product(), 5);
      gsl::at(reconstruction_order, i) =
          gsl::span(gsl::at(reconstruction_order_storage, i).data(),
                    gsl::at(reconstruction_order_storage, i).size());
    }
  }

  // The unnormalized normal vector is n_j = d \xi^{\hat i}/dx^j with "i"
  // the current face. When normalizing, we adopt a sign convention which
  // is compatible with DG.
  std::array<tnsr::i<DataVector, Dim, Frame::Inertial>, Dim> conormal;
  std::array<DirectionMap<Dim, tnsr::i<DataVector, Dim, Frame::Inertial>>, Dim>
      ghost_conormal;
  if (not AlignedCoordinates) {
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = 0; j < Dim; j++) {
        gsl::at(conormal, i).get(j) =
            inv_jacobian.get(i, j) / get(det_inv_jacobian);
      }

      for (const auto& direction : Direction<Dim>::all_directions()) {
        const auto& ghost_cells_grid_coords =
            get<evolution::dg::subcell::Tags::Coordinates<Dim, Frame::Grid>>(
                ghost_zone_inv_jac.at(direction));
        const auto& ghost_cells_inv_jacobian =
            get<evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToGrid<
                Dim>>(ghost_zone_inv_jac.at(direction));
        const auto ghost_cells_det_inv_jacobian =
            determinant(ghost_cells_inv_jacobian);
        tnsr::i<DataVector, Dim, Frame::Inertial> ghost_conormal_in_dir{
            ghost_cells_grid_coords.size()};
        for (size_t j = 0; j < Dim; j++) {
          ghost_conormal_in_dir.get(j) = ghost_cells_inv_jacobian.get(i, j) /
                                         get(ghost_cells_det_inv_jacobian);
        }
        gsl::at(ghost_conormal, i)
            .insert_or_assign(direction, ghost_conormal_in_dir);
      }
    }
  }

  // Now compute the Cartesian derivative of the high_order_corrections to
  // verify that it is computed sufficiently accurately.
  std::optional<std::array<CorrectionVars, Dim>> high_order_corrections{};
  ::fd::cartesian_high_order_flux_corrections(
      make_not_null(&high_order_corrections), volume_vars,
      second_order_corrections, correction_order, reconstruction_ghost_data,
      subcell_mesh, number_of_ghost_points, reconstruction_order,
      AlignedCoordinates, conormal, ghost_conormal);

  CorrectionVars flux_divergence{subcell_mesh.number_of_grid_points(), 0.0};
  for (size_t d = 0; d < Dim; ++d) {
    const auto& corrections_in_dim =
        high_order_corrections.has_value()
            ? gsl::at(high_order_corrections.value(), d)
            : gsl::at(second_order_corrections, d);
    // Note: assumes isotropic mesh
    const double one_over_delta_xi =
        -1.0 / (logical_coords.get(0)[1] - logical_coords.get(0)[0]);
    evolution::dg::subcell::add_cartesian_flux_divergence(
        make_not_null(&get(get<Scalar0>(flux_divergence))), one_over_delta_xi,
        get(det_inv_jacobian), get(get<Scalar0>(corrections_in_dim)),
        subcell_mesh.extents(), d);
    for (size_t i = 0; i < Dim; ++i) {
      evolution::dg::subcell::add_cartesian_flux_divergence(
          make_not_null(&get<Vector0<Dim>>(flux_divergence).get(i)),
          one_over_delta_xi, get(det_inv_jacobian),
          get<Vector0<Dim>>(corrections_in_dim).get(i), subcell_mesh.extents(),
          d);
    }
  }

  // With high-order corrections roundoff can accumulate for aligned coordinates
  // case.
  // In the case of non aligned coordinates, we adopt a higher error since the
  // Jacobian is more difficult to resolve. Note that this is mostly relevant
  // for the lowest derivative orders, which use very few grid points,
  // and the error converges rapidly with derivative order, emphasized by our
  // use of exponential scaling with `max_degree`. We chose a case with
  // relatively high error for 2nd order (error is ~2e-3) so that we can see the
  // improvement with each derivative order. Starting with too low an error in
  // 2nd order case would mean that you would approach error floors too quickly
  // with e.g. 10th order case, and the improvement would not be apparent.
  const Approx custom_approx =
      AlignedCoordinates ? Approx::custom().epsilon(2.e-10)
                         : Approx::custom().epsilon(pow(
                               10., -0.5 - static_cast<double>(max_degree)));
  CHECK_ITERABLE_CUSTOM_APPROX(get<Scalar0>(flux_divergence),
                               get<Scalar0>(expected_divergence),
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(get<Vector0<Dim>>(flux_divergence),
                               get<Vector0<Dim>>(expected_divergence),
                               custom_approx);

  CorrectionVars divergence_error(subcell_mesh.number_of_grid_points());
  get(get<Scalar0>(divergence_error)) =
      abs(get(get<Scalar0>(flux_divergence)) -
          get(get<Scalar0>(expected_divergence)));
  for (size_t i = 0; i < Dim; ++i) {
    get<Vector0<Dim>>(divergence_error).get(i) =
        abs(get<Vector0<Dim>>(flux_divergence).get(i) -
            get<Vector0<Dim>>(expected_divergence).get(i));
  }

  // Test assertions
#ifdef SPECTRE_DEBUG
  if (correction_order != fd::DerivativeOrder::Two) {
    std::optional<std::array<CorrectionVars, Dim>>
        high_order_corrections_assert = make_array<Dim>(CorrectionVars{
            second_order_corrections[0].number_of_grid_points()});
    high_order_corrections_assert.value()[0].initialize(
        second_order_corrections[0].number_of_grid_points() * 2);
    CHECK_THROWS_WITH(
        ::fd::cartesian_high_order_flux_corrections(
            make_not_null(&high_order_corrections_assert), volume_vars,
            second_order_corrections, correction_order,
            reconstruction_ghost_data, subcell_mesh, number_of_ghost_points),
        Catch::Matchers::ContainsSubstring(
            "The high_order_corrections must all have size"));
  }
  if constexpr (Dim > 1) {
    auto second_order_corrections_copy = second_order_corrections;
    second_order_corrections_copy[0].initialize(
        second_order_corrections_copy[0].number_of_grid_points() * 2);
    CHECK_THROWS_WITH(
        ::fd::cartesian_high_order_flux_corrections(
            make_not_null(&high_order_corrections), volume_vars,
            second_order_corrections_copy, correction_order,
            reconstruction_ghost_data, subcell_mesh, number_of_ghost_points),
        Catch::Matchers::ContainsSubstring(
            "All second-order boundary corrections must be of the same size"));
  }
#endif  // SPECTRE_DEBUG

  return std::max({max(get(get<Scalar0>(divergence_error))),
                   max(get(magnitude(get<Vector0<Dim>>(divergence_error))))});
}

// [[TimeOut, 10]]
SPECTRE_TEST_CASE("Unit.FiniteDifference.CartesianHighOrderFluxCorrection",
                  "[Unit][NumericalAlgorithms]") {
  using DO = fd::DerivativeOrder;
  for (const fd::DerivativeOrder correction_order :
       {DO::Two, DO::Four, DO::Six, DO::Eight, DO::Ten, DO::OneHigherThanRecons,
        DO::OneHigherThanReconsButFiveToFour}) {
    CAPTURE(correction_order);
    test<1, true>(correction_order);
    test<2, true>(correction_order);
    test<3, true>(correction_order);
    test<2, false>(correction_order);
    test<3, false>(correction_order);
  }
  // Test that error improves as we increase the correction order.
  std::optional<double> previous_error{};
  for (const fd::DerivativeOrder correction_order :
       {DO::Two, DO::Four, DO::Six, DO::Eight, DO::Ten}) {
    CAPTURE(correction_order);
    const auto current_error = test<3, false>(correction_order);
    if (previous_error.has_value()) {
      CHECK(current_error < previous_error);
    }
    previous_error = current_error;
  }
}
}  // namespace
