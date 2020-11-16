// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/FirstOrderSchemeLts.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Protocols.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct SomeField : db::SimpleTag {
  using type = Scalar<DataVector>;
};

using variables_tag = ::Tags::Variables<tmpl::list<SomeField>>;

struct ExtraDataTag : db::SimpleTag {
  using type = int;
};

struct NumericalFlux : tt::ConformsTo<dg::protocols::NumericalFlux> {
  using variables_tags = tmpl::list<SomeField>;
  using argument_tags = tmpl::list<SomeField, ExtraDataTag>;
  using volume_tags = tmpl::list<ExtraDataTag>;
  using package_field_tags = tmpl::list<SomeField>;
  using package_extra_tags = tmpl::list<ExtraDataTag>;
  static void package_data(
      const gsl::not_null<Scalar<DataVector>*> packaged_field,
      const gsl::not_null<int*> packaged_extra_data,
      const Scalar<DataVector>& field, const int& extra_data) noexcept {
    *packaged_field = field;
    *packaged_extra_data = extra_data;
  }
  void operator()(const gsl::not_null<Scalar<DataVector>*> numerical_flux,
                  const Scalar<DataVector>& field_int,
                  const int& extra_data_int,
                  const Scalar<DataVector>& field_ext,
                  const int& extra_data_ext) const noexcept {
    CHECK(extra_data_int == extra_data_ext);
    get(*numerical_flux) = 0.5 * (get(field_int) + get(field_ext));
  }
};

template <typename NumericalFluxType>
struct NumericalFluxTag : db::SimpleTag {
  using type = NumericalFluxType;
};

template <typename Tag>
struct BoundaryContribution : db::SimpleTag, db::PrefixTag {
  using tag = Tag;
  using type = typename Tag::type;
};

struct TemporalIdTag : db::SimpleTag {
  using type = int;
};

// Helper function to combine local and remote boundary data to mortar data
template <typename BoundaryScheme,
          typename BoundaryData = typename BoundaryScheme::BoundaryData>
auto make_mortar_data(const dg::MortarId<BoundaryScheme::volume_dim>& mortar_id,
                      const TimeStepId& time, BoundaryData&& interior_data,
                      BoundaryData&& exterior_data) noexcept {
  typename ::Tags::Mortars<typename BoundaryScheme::mortar_data_tag,
                           BoundaryScheme::volume_dim>::type all_mortar_data;
  all_mortar_data[mortar_id].local_insert(
      time, std::forward<BoundaryData>(interior_data));
  all_mortar_data[mortar_id].remote_insert(
      time, std::forward<BoundaryData>(exterior_data));
  return all_mortar_data;
}

template <size_t Dim>
void test_first_order_scheme_lts() {
  CAPTURE(Dim);
  using time_stepper_tag = Tags::TimeStepper<TimeSteppers::AdamsBashforthN>;
  using boundary_scheme = dg::FirstOrderScheme::FirstOrderSchemeLts<
      Dim, variables_tag,
      db::add_tag_prefix<BoundaryContribution, variables_tag>,
      NumericalFluxTag<NumericalFlux>, TemporalIdTag, time_stepper_tag>;

  using BoundaryData = typename boundary_scheme::BoundaryData;
  using mortar_data_tag = typename boundary_scheme::mortar_data_tag;
  using all_normal_dot_fluxes_tag = domain::Tags::Interface<
      domain::Tags::InternalDirections<Dim>,
      db::add_tag_prefix<::Tags::NormalDotFlux, variables_tag>>;
  using magnitude_of_face_normal_tag =
      ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>;
  using all_magnitude_of_face_normals_tag =
      domain::Tags::Interface<domain::Tags::InternalDirections<Dim>,
                              magnitude_of_face_normal_tag>;
  {
    INFO("Collect boundary data from a DataBox");
    // Create a DataBox that holds the arguments for the numerical flux plus
    // those for the strong first-order boundary scheme
    const Mesh<Dim> mesh{3, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};
    const size_t num_points = mesh.number_of_grid_points();
    const ElementId<Dim> element_id{0};
    const auto face_direction = Direction<Dim>::upper_xi();
    const ElementId<Dim> neighbor_id{1};
    const Element<Dim> element{element_id,
                               {{face_direction, {{neighbor_id}, {}}}}};
    const size_t num_points_on_face =
        mesh.slice_away(face_direction.dimension()).number_of_grid_points();
    typename db::add_tag_prefix<::Tags::NormalDotFlux, variables_tag>::type
        normal_dot_fluxes{num_points_on_face};
    get<::Tags::NormalDotFlux<SomeField>>(normal_dot_fluxes) =
        Scalar<DataVector>{num_points_on_face, 3.};
    Scalar<DataVector> magnitude_of_face_normal{num_points_on_face, 1.5};
    const int extra_data = 2;
    const auto box = db::create<
        db::AddSimpleTags<NumericalFluxTag<NumericalFlux>, SomeField,
                          ExtraDataTag, domain::Tags::Mesh<Dim>,
                          domain::Tags::Element<Dim>, all_normal_dot_fluxes_tag,
                          all_magnitude_of_face_normals_tag>,
        db::AddComputeTags<
            domain::Tags::InternalDirectionsCompute<Dim>,
            domain::Tags::InterfaceCompute<
                domain::Tags::InternalDirections<Dim>,
                domain::Tags::Direction<Dim>>,
            domain::Tags::InterfaceCompute<
                domain::Tags::InternalDirections<Dim>,
                domain::Tags::InterfaceMesh<Dim>>,
            domain::Tags::Slice<domain::Tags::InternalDirections<Dim>,
                                SomeField>>>(
        NumericalFlux{}, Scalar<DataVector>{num_points, 2.}, extra_data, mesh,
        element,
        typename all_normal_dot_fluxes_tag::type{
            {face_direction, std::move(normal_dot_fluxes)}},
        typename all_magnitude_of_face_normals_tag::type{
            {face_direction, magnitude_of_face_normal}});
    // Collect the boundary data needed by the boundary scheme
    const auto all_boundary_data =
        interface_apply<domain::Tags::InternalDirections<Dim>,
                        typename boundary_scheme::boundary_data_computer>(box);
    // Make sure the collected boundary data is what we expect
    const auto& boundary_data = all_boundary_data.at(face_direction);
    CHECK(get<SomeField>(boundary_data.field_data) ==
          Scalar<DataVector>{num_points_on_face, 2.});
    CHECK(get<::Tags::NormalDotFlux<SomeField>>(boundary_data.field_data) ==
          Scalar<DataVector>{num_points_on_face, 3.});
    CHECK(get<ExtraDataTag>(boundary_data.extra_data) == extra_data);
    CHECK(get<magnitude_of_face_normal_tag>(boundary_data.extra_data) ==
          magnitude_of_face_normal);
  }
  {
    // This part only tests that the boundary scheme can be applied to mutate a
    // DataBox. It can be replaced by a generic test that checks the struct
    // conforms to the interface that `mutate_apply` expects (once we have such
    // a test).
    INFO("Apply to DataBox");
    MAKE_GENERATOR(generator);
    std::uniform_real_distribution<> dist(-1., 1.);
    std::uniform_real_distribution<> dist_positive(0.5, 1.);
    const auto nn_generator = make_not_null(&generator);
    const auto nn_dist = make_not_null(&dist);
    const auto nn_dist_positive = make_not_null(&dist_positive);
    // Setup a volume mesh
    const Mesh<Dim> mesh{3, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};
    const size_t num_points = mesh.number_of_grid_points();
    const int extra_data = 2;

    const Slab slab{0., 1.};
    const auto time_step = slab.duration() / 2;
    const auto now = slab.start() + time_step;

    // Setup a mortar
    const auto mortar_mesh = mesh.slice_away(0);
    const size_t mortar_num_points = mortar_mesh.number_of_grid_points();
    const dg::MortarId<Dim> mortar_id{Direction<Dim>::upper_xi(),
                                      ElementId<Dim>{1}};
    dg::MortarSize<Dim - 1> mortar_size{};
    mortar_size.fill(Spectral::MortarSize::Full);
    const DataVector used_for_size_on_mortar{mortar_num_points};
    // Fake some boundary data
    const auto make_boundary_data = [&used_for_size_on_mortar, &nn_generator,
                                     &nn_dist, &nn_dist_positive]() noexcept {
      BoundaryData boundary_data{used_for_size_on_mortar.size()};
      get<SomeField>(boundary_data.field_data) =
          make_with_random_values<Scalar<DataVector>>(nn_generator, nn_dist,
                                                      used_for_size_on_mortar);
      get<::Tags::NormalDotFlux<SomeField>>(boundary_data.field_data) =
          make_with_random_values<Scalar<DataVector>>(nn_generator, nn_dist,
                                                      used_for_size_on_mortar);
      get<ExtraDataTag>(boundary_data.extra_data) = extra_data;
      get<magnitude_of_face_normal_tag>(boundary_data.extra_data) =
          make_with_random_values<Scalar<DataVector>>(
              nn_generator, nn_dist_positive, used_for_size_on_mortar);
      return boundary_data;
    };
    const auto local_boundary_data = make_boundary_data();
    const auto remote_boundary_data = make_boundary_data();
    auto all_mortar_data = make_mortar_data<boundary_scheme>(
        mortar_id, {true, 0, now}, local_boundary_data, remote_boundary_data);
    auto all_mortar_data_copy = make_mortar_data<boundary_scheme>(
        mortar_id, {true, 0, now}, local_boundary_data, remote_boundary_data);
    // Assemble a DataBox and test
    typename variables_tag::type boundary_contributions{num_points, 0.};
    auto box = db::create<db::AddSimpleTags<
        NumericalFluxTag<NumericalFlux>, domain::Tags::Mesh<Dim>,
        ::Tags::Mortars<mortar_data_tag, Dim>,
        ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
        ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>, variables_tag,
        time_stepper_tag, Tags::TimeStep>>(
        NumericalFlux{}, mesh,
        std::move(all_mortar_data),
        dg::MortarMap<Dim, Mesh<Dim - 1>>{{mortar_id, mortar_mesh}},
        dg::MortarMap<Dim, dg::MortarSize<Dim - 1>>{{mortar_id, mortar_size}},
        std::move(boundary_contributions),
        std::make_unique<TimeSteppers::AdamsBashforthN>(1), time_step);
    db::mutate_apply<boundary_scheme>(make_not_null(&box));
    typename variables_tag::type expected_boundary_contributions{num_points,
                                                                 0.};
    boundary_scheme::apply(
        make_not_null(&expected_boundary_contributions),
        make_not_null(&all_mortar_data_copy), mesh,
        get<::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>>(box),
        get<::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>(box),
        get<NumericalFluxTag<NumericalFlux>>(box), get<time_stepper_tag>(box),
        get<Tags::TimeStep>(box));
    const auto& mutated_boundary_contributions = get<variables_tag>(box);
    CHECK_VARIABLES_APPROX(mutated_boundary_contributions,
                           expected_boundary_contributions);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.DG.FirstOrderScheme.Lts",
                  "[Unit][NumericalAlgorithms]") {
  test_first_order_scheme_lts<1>();
  test_first_order_scheme_lts<2>();
  test_first_order_scheme_lts<3>();

  {
    // This test is carried over from:
    // tests/Unit/NumericalAlgorithms/DiscontinuousGalerkin/Actions/
    // Test_ApplyBoundaryFluxesLocalTimeStepping.cpp
    INFO("Slow and fast mortar");

    struct TestNumericalFlux : tt::ConformsTo<dg::protocols::NumericalFlux> {
      using variables_tags = tmpl::list<SomeField>;
      using argument_tags = tmpl::list<Tags::NormalDotFlux<SomeField>>;
      using package_field_tags = argument_tags;
      using package_extra_tags = tmpl::list<>;
      static void package_data(
          const gsl::not_null<Scalar<DataVector>*> packaged_n_dot_field,
          const Scalar<DataVector>& n_dot_field) noexcept {
        *packaged_n_dot_field = n_dot_field;
      }
      void operator()(const gsl::not_null<Scalar<DataVector>*> numerical_flux,
                      const Scalar<DataVector>& ndotf_internal,
                      const Scalar<DataVector>& ndotf_external) const {
        get(*numerical_flux) =
            10. * get(ndotf_internal) + 12. * get(ndotf_external);
      }
    };

    using time_stepper_tag = Tags::TimeStepper<TimeSteppers::AdamsBashforthN>;
    using boundary_scheme = dg::FirstOrderScheme::FirstOrderSchemeLts<
        2, variables_tag,
        db::add_tag_prefix<BoundaryContribution, variables_tag>,
        NumericalFluxTag<TestNumericalFlux>, TemporalIdTag, time_stepper_tag>;

    using BoundaryData = typename boundary_scheme::BoundaryData;
    using mortar_data_tag = typename boundary_scheme::mortar_data_tag;
    using magnitude_of_face_normal_tag =
        ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<2>>;

    const Slab slab(0., 1.);
    const auto time_step = slab.duration() / 2;
    const auto now = slab.start() + time_step;

    const ElementId<2> id(0);
    const auto face_direction = Direction<2>::upper_xi();
    const auto face_dimension = face_direction.dimension();
    const Mesh<2> mesh(3, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto);
    const auto slow_mortar = std::make_pair(
        Direction<2>::upper_xi(), ElementId<2>(1, {{{0, 0}, {1, 0}}}));
    const auto fast_mortar = std::make_pair(
        Direction<2>::upper_xi(), ElementId<2>(1, {{{0, 0}, {1, 1}}}));

    typename Tags::Mortars<domain::Tags::Mesh<1>, 2>::type mortar_meshes{
        {slow_mortar, mesh.slice_away(face_dimension)},
        {fast_mortar, mesh.slice_away(face_dimension)}};
    typename Tags::Mortars<Tags::MortarSize<1>, 2>::type mortar_sizes{
        {slow_mortar,
         dg::mortar_size(id, slow_mortar.second, face_dimension, {})},
        {fast_mortar,
         dg::mortar_size(id, fast_mortar.second, face_dimension, {})}};

    using Vars = Variables<tmpl::list<SomeField>>;
    Vars variables(mesh.number_of_grid_points(), 2.);

    auto local_data = make_array<2, BoundaryData>(
        mesh.slice_away(face_dimension).number_of_grid_points());
    auto remote_data = make_array<3, BoundaryData>(
        mesh.slice_away(face_dimension).number_of_grid_points());
    get(get<Tags::NormalDotFlux<SomeField>>(
        gsl::at(local_data, 0).field_data)) = DataVector{2., 3., 5.};
    get(get<magnitude_of_face_normal_tag>(gsl::at(local_data, 0).extra_data)) =
        DataVector{7., 11., 13.};
    get(get<Tags::NormalDotFlux<SomeField>>(
        gsl::at(local_data, 1).field_data)) = DataVector{17., 19., 23.};
    get(get<magnitude_of_face_normal_tag>(gsl::at(local_data, 1).extra_data)) =
        DataVector{29., 31., 37.};
    get(get<Tags::NormalDotFlux<SomeField>>(
        gsl::at(remote_data, 0).field_data)) = DataVector{41., 43., 47.};
    get(get<Tags::NormalDotFlux<SomeField>>(
        gsl::at(remote_data, 1).field_data)) = DataVector{53., 59., 61.};
    get(get<Tags::NormalDotFlux<SomeField>>(
        gsl::at(remote_data, 2).field_data)) = DataVector{67., 71., 73.};

    typename ::Tags::Mortars<mortar_data_tag, 2>::type mortar_data;
    mortar_data[slow_mortar].local_insert(TimeStepId(true, 0, now),
                                          gsl::at(local_data, 0));
    mortar_data[fast_mortar].local_insert(TimeStepId(true, 0, now),
                                          gsl::at(local_data, 1));
    mortar_data[slow_mortar].remote_insert(
        TimeStepId(true, 0, now - time_step / 2), gsl::at(remote_data, 0));
    mortar_data[fast_mortar].remote_insert(TimeStepId(true, 0, now),
                                           gsl::at(remote_data, 1));
    mortar_data[fast_mortar].remote_insert(
        TimeStepId(true, 0, now + time_step / 3), gsl::at(remote_data, 2));

    auto box = db::create<db::AddSimpleTags<
        NumericalFluxTag<TestNumericalFlux>, domain::Tags::Mesh<2>,
        ::Tags::Mortars<mortar_data_tag, 2>,
        ::Tags::Mortars<domain::Tags::Mesh<1>, 2>,
        ::Tags::Mortars<::Tags::MortarSize<1>, 2>, variables_tag,
        time_stepper_tag, Tags::TimeStep>>(
        TestNumericalFlux{}, mesh, std::move(mortar_data), mortar_meshes,
        mortar_sizes, variables,
        std::make_unique<TimeSteppers::AdamsBashforthN>(1), time_step);
    db::mutate_apply<boundary_scheme>(make_not_null(&box));

    add_slice_to_data(
        make_not_null(&variables),
        Vars(time_step.value() *
             dg::FirstOrderScheme::boundary_flux(
                 gsl::at(local_data, 0), gsl::at(remote_data, 0),
                 TestNumericalFlux{},
                 get<magnitude_of_face_normal_tag>(
                     gsl::at(local_data, 0).extra_data),
                 mesh.extents(face_dimension), mesh.slice_away(face_dimension),
                 mortar_meshes.at(slow_mortar), mortar_sizes.at(slow_mortar))),
        mesh.extents(), face_dimension,
        index_to_slice_at(mesh.extents(), face_direction));

    add_slice_to_data(
        make_not_null(&variables),
        Vars((time_step / 3).value() *
             dg::FirstOrderScheme::boundary_flux(
                 gsl::at(local_data, 1), gsl::at(remote_data, 1),
                 TestNumericalFlux{},
                 get<magnitude_of_face_normal_tag>(
                     gsl::at(local_data, 1).extra_data),
                 mesh.extents(face_dimension), mesh.slice_away(face_dimension),
                 mortar_meshes.at(fast_mortar), mortar_sizes.at(fast_mortar))),
        mesh.extents(), face_dimension,
        index_to_slice_at(mesh.extents(), face_direction));

    add_slice_to_data(
        make_not_null(&variables),
        Vars((time_step * 2 / 3).value() *
             dg::FirstOrderScheme::boundary_flux(
                 gsl::at(local_data, 1), gsl::at(remote_data, 2),
                 TestNumericalFlux{},
                 get<magnitude_of_face_normal_tag>(
                     gsl::at(local_data, 1).extra_data),
                 mesh.extents(face_dimension), mesh.slice_away(face_dimension),
                 mortar_meshes.at(fast_mortar), mortar_sizes.at(fast_mortar))),
        mesh.extents(), face_dimension,
        index_to_slice_at(mesh.extents(), face_direction));

    CHECK_ITERABLE_APPROX(get<SomeField>(variables), get<SomeField>(box));
  }
}
