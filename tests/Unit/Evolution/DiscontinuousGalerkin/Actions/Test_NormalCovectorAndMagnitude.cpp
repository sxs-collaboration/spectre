// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceComputeTags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

template <size_t VolumeDim>
auto make_affine_map() noexcept;

template <>
auto make_affine_map<1>() noexcept {
  return domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
      Affine{-1.0, 1.0, -0.3, 0.7});
}

template <>
auto make_affine_map<2>() noexcept {
  return domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
      Affine2D{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55}});
}

template <>
auto make_affine_map<3>() noexcept {
  return domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
      Affine3D{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55},
               Affine{-1.0, 1.0, 2.3, 2.8}});
}

struct ScaleZeroIndex : db::SimpleTag {
  using type = double;
};

template <size_t Dim>
struct SimpleUnnormalizedFaceNormal
    : db::ComputeTag,
      domain::Tags::UnnormalizedFaceNormal<Dim> {
  using base = domain::Tags::UnnormalizedFaceNormal<Dim>;
  using return_type = typename base::type;
  static void function(const gsl::not_null<return_type*> result,
                       const double scale_zero_index,
                       const Mesh<Dim - 1>& face_mesh,
                       const Direction<Dim>& direction) noexcept {
    for (size_t i = 0; i < Dim; ++i) {
      result->get(i) =
          DataVector{face_mesh.number_of_grid_points(),
                     i == direction.dimension() ? 1.0 * i + 0.25 : 0.0};
    }
    result->get(0) *= scale_zero_index;
  }
  using argument_tags = tmpl::list<ScaleZeroIndex, domain::Tags::Mesh<Dim - 1>,
                                   domain::Tags::Direction<Dim>>;
  using volume_tags = tmpl::list<ScaleZeroIndex>;
};

struct FlatSpaceSystem {};

template <size_t Dim>
struct InverseSpatialMetric : db::SimpleTag {
  using type = tnsr::II<DataVector, Dim, Frame::Inertial>;
};

template <size_t Dim>
struct SystemWithInverseMetric {
  using inverse_spatial_metric_tag = InverseSpatialMetric<Dim>;
};

template <size_t Dim>
auto create_box(const size_t number_of_grid_points_per_dim,
                const bool use_moving_mesh) {
  using internal_directions = domain::Tags::InternalDirections<Dim>;
  using simple_tags = tmpl::list<
      ScaleZeroIndex, domain::Tags::Mesh<Dim>,
      domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                  Frame::Inertial>,
      domain::Tags::Element<Dim>,
      evolution::dg::Tags::InternalFace::NormalCovectorAndMagnitude<Dim>>;
  using compute_tags =
      tmpl::list<domain::Tags::InternalDirectionsCompute<Dim>,
                 domain::Tags::InterfaceCompute<internal_directions,
                                                domain::Tags::Direction<Dim>>,
                 domain::Tags::InterfaceCompute<
                     internal_directions, domain::Tags::InterfaceMesh<Dim>>,
                 domain::Tags::InterfaceCompute<
                     internal_directions, SimpleUnnormalizedFaceNormal<Dim>>>;

  auto grid_to_inertial_map =
      use_moving_mesh
          ? make_affine_map<Dim>()
          : domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
                domain::CoordinateMaps::Identity<Dim>{});

  DirectionMap<Dim, Neighbors<Dim>> neighbors{};
  ElementId<Dim> self_id{};
  ElementId<Dim> east_id{};
  ElementId<Dim> south_id{};  // not used in 1d
  ElementId<Dim> up_id{};  // not used in 1d or 2d

  if constexpr (Dim == 1) {
    self_id = ElementId<Dim>{0, {{{1, 0}}}};
    east_id = ElementId<Dim>{0, {{{1, 1}}}};
    neighbors[Direction<Dim>::upper_xi()] = Neighbors<Dim>{{east_id}, {}};
  } else if constexpr (Dim == 2) {
    self_id = ElementId<Dim>{0, {{{1, 0}, {0, 0}}}};
    east_id = ElementId<Dim>{0, {{{1, 1}, {0, 0}}}};
    south_id = ElementId<Dim>{1, {{{1, 0}, {0, 0}}}};
    neighbors[Direction<Dim>::upper_xi()] = Neighbors<Dim>{{east_id}, {}};
    neighbors[Direction<Dim>::lower_eta()] = Neighbors<Dim>{{south_id}, {}};
  } else {
    static_assert(Dim == 3, "Only implemented tests in 1, 2, and 3d");
    self_id = ElementId<Dim>{0, {{{1, 0}, {0, 0}, {2, 1}}}};
    east_id = ElementId<Dim>{0, {{{1, 1}, {0, 0}, {2, 1}}}};
    south_id = ElementId<Dim>{1, {{{1, 0}, {0, 0}, {2, 1}}}};
    up_id = ElementId<Dim>{0, {{{1, 0}, {0, 0}, {2, 2}}}};
    neighbors[Direction<Dim>::upper_xi()] = Neighbors<Dim>{{east_id}, {}};
    neighbors[Direction<Dim>::lower_eta()] = Neighbors<Dim>{{south_id}, {}};
    neighbors[Direction<Dim>::upper_zeta()] = Neighbors<Dim>{{up_id}, {}};
  }
  const Element<Dim> element{self_id, neighbors};

  DirectionMap<Dim,
               std::optional<Variables<tmpl::list<
                   evolution::dg::Tags::InternalFace::MagnitudeOfNormal,
                   evolution::dg::Tags::InternalFace::NormalCovector<Dim>>>>>
      normal_covector_quantities{};
  for (const auto& [direction, local_neighbors] : element.neighbors()) {
    (void)local_neighbors;
    normal_covector_quantities[direction] = std::nullopt;
  }

  return db::create<simple_tags, compute_tags>(
      1.0,
      Mesh<Dim>{number_of_grid_points_per_dim, Spectral::Basis::Legendre,
                Spectral::Quadrature::Gauss},
      std::move(grid_to_inertial_map), std::move(element),
      std::move(normal_covector_quantities));
}

template <bool UseFlatSpace, size_t Dim, typename DbTagsList>
void check_normal_covector_quantities(
    const gsl::not_null<db::DataBox<DbTagsList>*> box) noexcept {
  using field_face_tags = tmpl::conditional_t<
      UseFlatSpace,
      tmpl::list<evolution::dg::Actions::detail::OneOverNormalVectorMagnitude>,
      tmpl::list<evolution::dg::Actions::detail::OneOverNormalVectorMagnitude,
                 evolution::dg::Actions::detail::NormalVector<Dim>,
                 InverseSpatialMetric<Dim>>>;
  using internal_directions = domain::Tags::InternalDirections<Dim>;
  const auto& unnormalized_normal_covectors = db::get<domain::Tags::Interface<
      internal_directions, domain::Tags::UnnormalizedFaceNormal<Dim>>>(*box);
  for (const auto& [direction, neighbors] :
       db::get<domain::Tags::Element<Dim>>(*box).neighbors()) {
    (void)neighbors;
    Variables<field_face_tags> fields_on_face{
        db::get<domain::Tags::Interface<internal_directions,
                                        domain::Tags::Mesh<Dim - 1>>>(*box)
            .at(direction)
            .number_of_grid_points()};
    if constexpr (not UseFlatSpace) {
      double temp = 0.1;
      for (auto& component : get<InverseSpatialMetric<Dim>>(fields_on_face)) {
        for (double& t : component) {
          t = temp;
          temp *= 1.1;
        }
      }
    }

    db::mutate<
        evolution::dg::Tags::InternalFace::NormalCovectorAndMagnitude<Dim>>(
        box,
        &evolution::dg::Actions::detail::
            unit_normal_vector_and_covector_and_magnitude<
                tmpl::conditional_t<UseFlatSpace, FlatSpaceSystem,
                                    SystemWithInverseMetric<Dim>>,
                Dim, field_face_tags>,
        make_not_null(&fields_on_face), direction,
        unnormalized_normal_covectors,
        db::get<domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                            Frame::Inertial>>(
            *box));

    if constexpr (UseFlatSpace) {
      const auto& normal_covector =
          get<evolution::dg::Tags::InternalFace::NormalCovector<Dim>>(
              *get<evolution::dg::Tags::InternalFace::
                       NormalCovectorAndMagnitude<Dim>>(*box)
                   .at(direction));
      const DataVector expected_normal_magnitude{
          fields_on_face.number_of_grid_points(), 1.0};
      CHECK_ITERABLE_APPROX(expected_normal_magnitude,
                            get(magnitude(normal_covector)));
    } else {
      const auto& normal_covector =
          get<evolution::dg::Tags::InternalFace::NormalCovector<Dim>>(
              *get<evolution::dg::Tags::InternalFace::
                       NormalCovectorAndMagnitude<Dim>>(*box)
                   .at(direction));
      CHECK_ITERABLE_APPROX(
          (DataVector{fields_on_face.number_of_grid_points(), 1.0}),
          sqrt(get(
              dot_product(normal_covector, normal_covector,
                          get<InverseSpatialMetric<Dim>>(fields_on_face)))));
    }

    const auto& magnitude_of_normal =
        get(get<evolution::dg::Tags::InternalFace::MagnitudeOfNormal>(
            *get<evolution::dg::Tags::InternalFace::NormalCovectorAndMagnitude<
                 Dim>>(*box)
                 .at(direction)));
    CHECK(min(magnitude_of_normal) > 0.0);

    for (size_t i = 0; i < Dim; ++i) {
      const auto& normal_covector_and_magnitude =
          *get<evolution::dg::Tags::InternalFace::NormalCovectorAndMagnitude<
              Dim>>(*box)
               .at(direction);
      const auto& unnormalized_covector_in_dir =
          unnormalized_normal_covectors.at(direction).get(i);
      const DataVector expected_unnormalized_covector_in_dir{
          get<evolution::dg::Tags::InternalFace::NormalCovector<Dim>>(
              normal_covector_and_magnitude)
              .get(i) *
          get(get<evolution::dg::Tags::InternalFace::MagnitudeOfNormal>(
              normal_covector_and_magnitude))};
      CHECK_ITERABLE_APPROX(unnormalized_covector_in_dir,
                            expected_unnormalized_covector_in_dir);
    }
  }
}

template <size_t Dim, bool IsFlatSpace>
void test(const bool use_moving_mesh) {
  CAPTURE(Dim);
  CAPTURE(use_moving_mesh);
  CAPTURE(IsFlatSpace);
  const size_t number_of_grid_points_per_dim = 3;
  auto box = create_box<Dim>(number_of_grid_points_per_dim, use_moving_mesh);
  check_normal_covector_quantities<IsFlatSpace, Dim>(make_not_null(&box));

  if (use_moving_mesh) {
    // Mutate the x component of the unnormalized normal vector to simulate
    // moving mesh
    db::mutate<ScaleZeroIndex>(
        make_not_null(&box),
        [](const gsl::not_null<double*> scale_zero_index) noexcept {
          *scale_zero_index = 2.3;
        });

    check_normal_covector_quantities<IsFlatSpace, Dim>(make_not_null(&box));
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.DG.NormalCovectorAndMagnitude",
                  "[Unit][Evolution][Actions]") {
  for (const auto use_moving_mesh : {true, false}) {
    test<1, true>(use_moving_mesh);
    test<2, true>(use_moving_mesh);
    test<3, true>(use_moving_mesh);

    test<1, false>(use_moving_mesh);
    test<2, false>(use_moving_mesh);
    test<3, false>(use_moving_mesh);
  }
}
}  // namespace
