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
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/FirstOrderScheme.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Protocols.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
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

struct NumericalFluxTag : db::SimpleTag {
  using type = NumericalFlux;
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
                      const int time, BoundaryData&& interior_data,
                      BoundaryData&& exterior_data) noexcept {
  typename BoundaryScheme::mortar_data_tag::type mortar_data{};
  mortar_data.local_insert(time, std::forward<BoundaryData>(interior_data));
  mortar_data.remote_insert(time, std::forward<BoundaryData>(exterior_data));
  return typename ::Tags::Mortars<typename BoundaryScheme::mortar_data_tag,
                                  BoundaryScheme::volume_dim>::type{
      {mortar_id, std::move(mortar_data)}};
}

template <size_t Dim>
void test_first_order_scheme() {
  CAPTURE(Dim);
  using boundary_scheme = dg::FirstOrderScheme::FirstOrderScheme<
      Dim, variables_tag,
      db::add_tag_prefix<BoundaryContribution, variables_tag>, NumericalFluxTag,
      TemporalIdTag>;

  using BoundaryData = typename boundary_scheme::BoundaryData;
  using mortar_data_tag = typename boundary_scheme::mortar_data_tag;
  using all_normal_dot_fluxes_tag = domain::Tags::Interface<
      domain::Tags::InternalDirections<Dim>,
      db::add_tag_prefix<::Tags::NormalDotFlux, variables_tag>>;
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
    const int extra_data = 2;
    const auto box = db::create<
        db::AddSimpleTags<NumericalFluxTag, SomeField, ExtraDataTag,
                          domain::Tags::Mesh<Dim>, domain::Tags::Element<Dim>,
                          all_normal_dot_fluxes_tag>,
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
            {face_direction, std::move(normal_dot_fluxes)}});
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
  }
  {
    // This part only tests that the boundary scheme can be applied to mutate a
    // DataBox. It can be replaced by a generic test that checks the struct
    // conforms to the interface that `mutate_apply` expects (once we have such
    // a test).
    INFO("Apply to DataBox");
    MAKE_GENERATOR(generator);
    std::uniform_real_distribution<> dist(-1., 1.);
    const auto nn_generator = make_not_null(&generator);
    const auto nn_dist = make_not_null(&dist);
    // Setup a volume mesh
    const Mesh<Dim> mesh{3, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};
    const size_t num_points = mesh.number_of_grid_points();
    const int extra_data = 2;
    const int dummy_time = 1;
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
                                     &nn_dist]() noexcept {
      BoundaryData boundary_data{used_for_size_on_mortar.size()};
      get<SomeField>(boundary_data.field_data) =
          make_with_random_values<Scalar<DataVector>>(nn_generator, nn_dist,
                                                      used_for_size_on_mortar);
      get<::Tags::NormalDotFlux<SomeField>>(boundary_data.field_data) =
          make_with_random_values<Scalar<DataVector>>(nn_generator, nn_dist,
                                                      used_for_size_on_mortar);
      get<ExtraDataTag>(boundary_data.extra_data) = extra_data;
      return boundary_data;
    };
    auto all_mortar_data = make_mortar_data<boundary_scheme>(
        mortar_id, dummy_time, make_boundary_data(), make_boundary_data());
    // Assemble a DataBox and test
    typename db::add_tag_prefix<BoundaryContribution, variables_tag>::type
        boundary_contributions{num_points, 0.};
    auto box = db::create<db::AddSimpleTags<
        NumericalFluxTag, domain::Tags::Mesh<Dim>,
        domain::Tags::Interface<
            domain::Tags::InternalDirections<Dim>,
            ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
        domain::Tags::Interface<
            domain::Tags::BoundaryDirectionsInterior<Dim>,
            ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
        ::Tags::Mortars<mortar_data_tag, Dim>,
        ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
        ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>,
        db::add_tag_prefix<BoundaryContribution, variables_tag>>>(
        NumericalFlux{}, mesh,
        std::unordered_map<Direction<Dim>, Scalar<DataVector>>{
            {mortar_id.first,
             make_with_random_values<Scalar<DataVector>>(
                 nn_generator, nn_dist, used_for_size_on_mortar)}},
        std::unordered_map<Direction<Dim>, Scalar<DataVector>>{},
        all_mortar_data,
        dg::MortarMap<Dim, Mesh<Dim - 1>>{{mortar_id, mortar_mesh}},
        dg::MortarMap<Dim, dg::MortarSize<Dim - 1>>{{mortar_id, mortar_size}},
        std::move(boundary_contributions));
    db::mutate_apply<boundary_scheme>(make_not_null(&box));
    typename db::add_tag_prefix<BoundaryContribution, variables_tag>::type
        expected_boundary_contributions{num_points, 0.};
    boundary_scheme::apply(
        make_not_null(&expected_boundary_contributions),
        make_not_null(&all_mortar_data), mesh,
        get<::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>>(box),
        get<::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>(box),
        get<NumericalFluxTag>(box),
        get<domain::Tags::Interface<
            domain::Tags::InternalDirections<Dim>,
            ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>>(box),
        get<domain::Tags::Interface<
            domain::Tags::BoundaryDirectionsInterior<Dim>,
            ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>>(
            box));
    const auto& mutated_boundary_contributions =
        get<db::add_tag_prefix<BoundaryContribution, variables_tag>>(box);
    CHECK_VARIABLES_APPROX(mutated_boundary_contributions,
                           expected_boundary_contributions);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.DG.FirstOrderScheme", "[Unit][NumericalAlgorithms]") {
  test_first_order_scheme<1>();
  test_first_order_scheme<2>();
  test_first_order_scheme<3>();
}
