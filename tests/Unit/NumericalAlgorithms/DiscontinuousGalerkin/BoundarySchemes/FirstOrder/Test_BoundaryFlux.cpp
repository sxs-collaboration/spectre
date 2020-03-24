// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceComputeTags.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Utilities/ProtocolTestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/BoundaryData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/BoundaryFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Protocols.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace {

struct SomeField : db::SimpleTag {
  using type = Scalar<DataVector>;
};

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
    // A simple central flux
    get(*numerical_flux) = 0.5 * (get(field_int) + get(field_ext));
  }
};

static_assert(
    test_protocol_conformance<NumericalFlux, dg::protocols::NumericalFlux>,
    "Failed testing protocol conformance");

// A flux used in earlier versions of this test (see history of
// tests/Unit/NumericalAlgorithms/DiscontinuousGalerkin/Test_MortarHelpers.cpp),
// included here to make sure it didn't break
struct RefinementTestsNumericalFlux
    : tt::ConformsTo<dg::protocols::NumericalFlux> {
  DataVector answer;
  explicit RefinementTestsNumericalFlux(DataVector local_answer) noexcept
      : answer{std::move(local_answer)} {};

  using variables_tags = tmpl::list<SomeField>;
  using argument_tags = tmpl::list<SomeField>;
  using package_field_tags = tmpl::list<SomeField>;
  using package_extra_tags = tmpl::list<>;
  void package_data(const gsl::not_null<Scalar<DataVector>*> packaged_field,
                    const Scalar<DataVector>& field) const noexcept {
    *packaged_field = field;
  }
  void operator()(const gsl::not_null<Scalar<DataVector>*> numerical_flux,
                  const Scalar<DataVector>& local_var,
                  const Scalar<DataVector>& remote_var) const noexcept {
    CHECK(get(local_var) == DataVector{1., 2., 3.});
    CHECK(get(remote_var) == DataVector{6., 5., 4.});
    get(*numerical_flux) = answer;
  }
};

static_assert(test_protocol_conformance<RefinementTestsNumericalFlux,
                                        dg::protocols::NumericalFlux>,
              "Failed testing protocol conformance");

// Helper function to compare a simple setup to the Python implementation
template <size_t Dim, size_t NumPointsPerDim, typename NumericalFluxType>
Scalar<DataVector> simple_boundary_flux(
    const Scalar<DataVector>& field_int,
    const Scalar<DataVector>& field_ext) noexcept {
  constexpr size_t num_points_per_dim = NumPointsPerDim;
  CAPTURE(num_points_per_dim);
  using BoundaryData = dg::FirstOrderScheme::BoundaryData<NumericalFluxType>;
  // Setup a mortar
  const Mesh<Dim - 1> mortar_mesh{num_points_per_dim, Spectral::Basis::Legendre,
                                  Spectral::Quadrature::GaussLobatto};
  const size_t mortar_num_points = mortar_mesh.number_of_grid_points();
  dg::MortarSize<Dim - 1> mortar_size{};
  mortar_size.fill(Spectral::MortarSize::Full);
  // Make sure the input fields are of the correct size
  ASSERT(get(field_int).size() == mortar_num_points &&
             get(field_ext).size() == mortar_num_points,
         "The input fields have "
             << get(field_int).size() << " points but should have "
             << mortar_num_points
             << "points since they represent data on the boundary.");
  // Construct data on either side of the mortar
  const auto make_boundary_data =
      [&mortar_num_points](const Scalar<DataVector>& field,
                           const Direction<Dim>& direction) noexcept {
        BoundaryData boundary_data{mortar_num_points};
        get<SomeField>(boundary_data.field_data) = field;
        get(get<::Tags::NormalDotFlux<SomeField>>(boundary_data.field_data)) =
            direction.sign() * get(field);
        get<ExtraDataTag>(boundary_data.extra_data) = 1;
        return boundary_data;
      };
  return get<SomeField>(dg::FirstOrderScheme::boundary_flux(
      make_boundary_data(field_int, Direction<Dim>::upper_xi()),
      make_boundary_data(field_ext, Direction<Dim>::lower_xi()),
      NumericalFluxType{}, Scalar<DataVector>{mortar_num_points, 1.},
      num_points_per_dim, mortar_mesh, mortar_mesh, mortar_size));
}

template <size_t Dim>
void test_simple_boundary_flux() noexcept {
  CAPTURE(Dim);
  {
    INFO("Compare to Python implementation");
    constexpr size_t num_points_per_dim = 5;
    const DataVector used_for_size_on_face{pow<Dim - 1>(num_points_per_dim)};
    pypp::check_with_random_values<1>(
        &simple_boundary_flux<Dim, num_points_per_dim, NumericalFlux>,
        "BoundaryFlux", "boundary_flux", {{{-1., 1.}}},
        used_for_size_on_face);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DG.FirstOrderScheme.BoundaryFlux",
                  "[Unit][NumericalAlgorithms]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/"
      "FirstOrder/");
  test_simple_boundary_flux<1>();
  test_simple_boundary_flux<2>();
  test_simple_boundary_flux<3>();

  {
    // This test was carried over from Test_MortarHelpers.cpp
    INFO("p-refinement");
    static constexpr size_t Dim = 2;
    const RefinementTestsNumericalFlux numerical_flux{{0., 3., 0.}};
    using BoundaryData =
        dg::FirstOrderScheme::BoundaryData<RefinementTestsNumericalFlux>;
    const Mesh<Dim> mesh{{{4, 2}},
                         Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};
    // Setup a mortar
    const dg::MortarId<Dim> mortar_id{Direction<Dim>::upper_xi(),
                                      ElementId<Dim>{1}};
    const size_t slice_dim = mortar_id.first.dimension();
    const auto face_mesh = mesh.slice_away(slice_dim);
    const size_t face_num_points = face_mesh.number_of_grid_points();
    // The face has 2 grid points, but we make a mortar mesh with 3 grid points,
    // so this test includes a projection from a p-refined mortar mesh.
    const Mesh<Dim - 1> mortar_mesh{3, Spectral::Basis::Legendre,
                                    Spectral::Quadrature::GaussLobatto};
    const size_t mortar_num_points = mortar_mesh.number_of_grid_points();
    const size_t perpendicular_extent = mesh.extents(slice_dim);
    const dg::MortarSize<Dim - 1> mortar_size{{Spectral::MortarSize::Full}};
    // Construct boundary data
    const Scalar<DataVector> magnitude_of_face_normal{2., 5.};
    BoundaryData interior_data{mortar_num_points};
    get(get<SomeField>(interior_data.field_data)) = DataVector{1., 2., 3.};
    get(get<Tags::NormalDotFlux<SomeField>>(interior_data.field_data)) =
        DataVector{-3., 0., 3.};
    BoundaryData exterior_data{mortar_num_points};
    get(get<SomeField>(exterior_data.field_data)) = DataVector{6., 5., 4.};
    // Apply boundary scheme
    const auto boundary_flux = dg::FirstOrderScheme::boundary_flux(
        interior_data, exterior_data, numerical_flux, magnitude_of_face_normal,
        perpendicular_extent, face_mesh, mortar_mesh, mortar_size);
    // Projected F* - F = {5., -1.}
    Variables<tmpl::list<::Tags::NormalDotFlux<SomeField>>> fstar_minus_f{
        face_num_points};
    get(get<::Tags::NormalDotFlux<SomeField>>(fstar_minus_f)) =
        DataVector{5., -1.};
    const auto expected = dg::lift_flux(fstar_minus_f, perpendicular_extent,
                                        magnitude_of_face_normal);
    CHECK_VARIABLES_APPROX(boundary_flux, expected);
  }
  {
    // This test was carried over from Test_MortarHelpers.cpp
    INFO("h-refinement");
    constexpr size_t Dim = 2;
    using BoundaryData =
        dg::FirstOrderScheme::BoundaryData<RefinementTestsNumericalFlux>;
    const Mesh<Dim> mesh{{{4, 3}},
                         Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};
    const auto compute_contribution = [&mesh](
                                          const dg::MortarId<Dim>& mortar_id,
                                          const dg::MortarSize<Dim - 1>&
                                              mortar_size,
                                          const DataVector&
                                              numerical_flux) noexcept {
      const auto mortar_mesh = mesh.slice_away(mortar_id.first.dimension());
      const size_t mortar_num_points = mortar_mesh.number_of_grid_points();

      // These are all arbitrary
      const DataVector local_flux{-1., 5., 7.};
      const Scalar<DataVector> magnitude_of_face_normal{{{{2., 5., 7.}}}};

      BoundaryData interior_data{mortar_num_points};
      get(get<Tags::NormalDotFlux<SomeField>>(interior_data.field_data)) =
          local_flux;
      get<SomeField>(interior_data.field_data) =
          Scalar<DataVector>{mortar_num_points, 0.};
      interior_data = interior_data.project_to_mortar(mortar_mesh, mortar_mesh,
                                                      mortar_size);
      get(get<SomeField>(interior_data.field_data)) = DataVector{1., 2., 3.};

      BoundaryData exterior_data{mortar_num_points};
      get(get<SomeField>(exterior_data.field_data)) = DataVector{6., 5., 4.};

      return dg::FirstOrderScheme::boundary_flux(
          interior_data, exterior_data,
          RefinementTestsNumericalFlux{numerical_flux},
          magnitude_of_face_normal, mesh.extents(mortar_id.first.dimension()),
          mortar_mesh, mortar_mesh, mortar_size);
    };
    const auto unrefined_result =
        compute_contribution({Direction<Dim>::upper_xi(), ElementId<Dim>{0}},
                             {{Spectral::MortarSize::Full}}, {1., 4., 9.});
    const decltype(unrefined_result) refined_result =
        compute_contribution({Direction<Dim>::upper_xi(), ElementId<Dim>{0}},
                             {{Spectral::MortarSize::LowerHalf}},
                             {1., 9. / 4., 4.}) +
        compute_contribution({Direction<Dim>::upper_xi(), ElementId<Dim>{1}},
                             {{Spectral::MortarSize::UpperHalf}},
                             {4., 25. / 4., 9.});
    CHECK_VARIABLES_APPROX(unrefined_result, refined_result);
  }
}
