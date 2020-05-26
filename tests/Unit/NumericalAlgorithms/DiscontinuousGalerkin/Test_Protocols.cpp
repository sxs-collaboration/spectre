// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Protocols.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct FieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

// [numerical_flux_example]
struct CentralFlux : tt::ConformsTo<dg::protocols::NumericalFlux> {
  using variables_tags = tmpl::list<FieldTag>;
  using argument_tags = tmpl::list<::Tags::NormalDotFlux<FieldTag>>;
  using package_field_tags = tmpl::list<::Tags::NormalDotFlux<FieldTag>>;
  using package_extra_tags = tmpl::list<>;
  static void package_data(
      const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux,
      const Scalar<DataVector>& normal_dot_flux) noexcept {
    *packaged_normal_dot_flux = normal_dot_flux;
  }
  void operator()(const gsl::not_null<Scalar<DataVector>*> numerical_flux,
                  const Scalar<DataVector>& normal_dot_flux_interior,
                  const Scalar<DataVector>& normal_dot_flux_exterior) const
      noexcept {
    // The minus sign appears because the `normal_dot_flux_exterior` was
    // computed with the interface normal from the neighboring element
    get(*numerical_flux) =
        0.5 * (get(normal_dot_flux_interior) - get(normal_dot_flux_exterior));
  }
};
// [numerical_flux_example]

static_assert(
    tt::assert_conforms_to<CentralFlux, dg::protocols::NumericalFlux>);

}  // namespace
