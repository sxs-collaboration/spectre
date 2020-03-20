// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/BoundaryData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Protocols.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

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

}  // namespace

SPECTRE_TEST_CASE("Unit.DG.FirstOrderScheme.BoundaryData",
                  "[Unit][NumericalAlgorithms]") {
  const Scalar<DataVector> field{{{{1., 2., 3.}}}};
  const int extra_data = 1;
  Variables<tmpl::list<::Tags::NormalDotFlux<SomeField>>> normal_dot_fluxes{3};
  get<::Tags::NormalDotFlux<SomeField>>(normal_dot_fluxes) =
      Scalar<DataVector>{{{{4., 5., 6.}}}};
  const auto packaged_data = dg::FirstOrderScheme::package_boundary_data(
      NumericalFlux{},
      Mesh<1>{3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto},
      normal_dot_fluxes, field, extra_data);
  CHECK(get<SomeField>(packaged_data.field_data) == field);
  CHECK(get<ExtraDataTag>(packaged_data.extra_data) == extra_data);
  CHECK(get<::Tags::NormalDotFlux<SomeField>>(packaged_data.field_data) ==
        get<::Tags::NormalDotFlux<SomeField>>(normal_dot_fluxes));
}
