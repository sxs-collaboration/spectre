// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Protocols.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ProtocolTestHelpers.hpp"

namespace {

struct FieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct ExtraTag : db::SimpleTag {
  using type = int;
};

struct ValidNumericalFlux : tt::ConformsTo<dg::protocols::NumericalFlux> {
  using variables_tags = tmpl::list<FieldTag>;
  using argument_tags = tmpl::list<FieldTag>;
  using package_field_tags = tmpl::list<FieldTag>;
  using package_extra_tags = tmpl::list<ExtraTag>;
  void package_data(gsl::not_null<Scalar<DataVector>*> packaged_field,
                    gsl::not_null<int*> packaged_int,
                    const Scalar<DataVector>& field) const noexcept;
  void operator()(gsl::not_null<Scalar<DataVector>*> numerical_flux_for_field,
                  const Scalar<DataVector>& field_interior,
                  const int& int_interior,
                  const Scalar<DataVector>& field_exterior,
                  const int& int_exterior) const noexcept;
};

struct NumericalFluxMissingVarsTags
    : tt::ConformsTo<dg::protocols::NumericalFlux> {
  using argument_tags = tmpl::list<>;
  using package_field_tags = tmpl::list<>;
  using package_extra_tags = tmpl::list<>;
};
struct NumericalFluxMissingArgumentTags
    : tt::ConformsTo<dg::protocols::NumericalFlux> {
  using variables_tags = tmpl::list<>;
  using package_field_tags = tmpl::list<>;
  using package_extra_tags = tmpl::list<>;
};
struct NumericalFluxMissingPackageFieldTags
    : tt::ConformsTo<dg::protocols::NumericalFlux> {
  using variables_tags = tmpl::list<>;
  using argument_tags = tmpl::list<>;
  using package_extra_tags = tmpl::list<>;
};
struct NumericalFluxMissingPackageExtraTags
    : tt::ConformsTo<dg::protocols::NumericalFlux> {
  using variables_tags = tmpl::list<>;
  using argument_tags = tmpl::list<>;
  using package_field_tags = tmpl::list<>;
};
struct NumericalFluxMissingPackageData
    : tt::ConformsTo<dg::protocols::NumericalFlux> {
  using variables_tags = tmpl::list<>;
  using argument_tags = tmpl::list<>;
  using package_field_tags = tmpl::list<>;
  using package_extra_tags = tmpl::list<>;
  void operator()(gsl::not_null<Scalar<DataVector>*> numerical_flux_for_field,
                  const Scalar<DataVector>& field_interior,
                  const int& int_interior,
                  const Scalar<DataVector>& field_exterior,
                  const int& int_exterior) const noexcept;
};
struct NumericalFluxInvalidPackageData
    : tt::ConformsTo<dg::protocols::NumericalFlux> {
  using variables_tags = tmpl::list<FieldTag>;
  using argument_tags = tmpl::list<FieldTag>;
  using package_field_tags = tmpl::list<FieldTag>;
  using package_extra_tags = tmpl::list<ExtraTag>;
  void package_data(gsl::not_null<Scalar<DataVector>*> packaged_field,
                    const Scalar<DataVector>& field) const noexcept;
  void operator()(gsl::not_null<Scalar<DataVector>*> numerical_flux_for_field,
                  const Scalar<DataVector>& field_interior,
                  const int& int_interior,
                  const Scalar<DataVector>& field_exterior,
                  const int& int_exterior) const noexcept;
};
struct NumericalFluxMissingCallOperator
    : tt::ConformsTo<dg::protocols::NumericalFlux> {
  using variables_tags = tmpl::list<FieldTag>;
  using argument_tags = tmpl::list<FieldTag>;
  using package_field_tags = tmpl::list<FieldTag>;
  using package_extra_tags = tmpl::list<ExtraTag>;
  void package_data(gsl::not_null<Scalar<DataVector>*> packaged_field,
                    gsl::not_null<int*> packaged_int,
                    const Scalar<DataVector>& field) const noexcept;
};
struct NumericalFluxInvalidCallOperator
    : tt::ConformsTo<dg::protocols::NumericalFlux> {
  using variables_tags = tmpl::list<FieldTag>;
  using argument_tags = tmpl::list<FieldTag>;
  using package_field_tags = tmpl::list<FieldTag>;
  using package_extra_tags = tmpl::list<ExtraTag>;
  void package_data(gsl::not_null<Scalar<DataVector>*> packaged_field,
                    gsl::not_null<int*> packaged_int,
                    const Scalar<DataVector>& field) const noexcept;
  void operator()(const Scalar<DataVector>& field_interior,
                  const Scalar<DataVector>& field_exterior) const noexcept;
};

static_assert(dg::protocols::NumericalFlux<ValidNumericalFlux>::value,
              "Failed testing protocol");
static_assert(
    not dg::protocols::NumericalFlux<NumericalFluxMissingVarsTags>::value,
    "Failed testing protocol");
static_assert(
    not dg::protocols::NumericalFlux<NumericalFluxMissingArgumentTags>::value,
    "Failed testing protocol");
static_assert(not dg::protocols::NumericalFlux<
                  NumericalFluxMissingPackageFieldTags>::value,
              "Failed testing protocol");
static_assert(not dg::protocols::NumericalFlux<
                  NumericalFluxMissingPackageExtraTags>::value,
              "Failed testing protocol");
static_assert(
    not dg::protocols::NumericalFlux<NumericalFluxMissingPackageData>::value,
    "Failed testing protocol");
static_assert(
    not dg::protocols::NumericalFlux<NumericalFluxInvalidPackageData>::value,
    "Failed testing protocol");
static_assert(
    not dg::protocols::NumericalFlux<NumericalFluxMissingCallOperator>::value,
    "Failed testing protocol");
static_assert(
    not dg::protocols::NumericalFlux<NumericalFluxInvalidCallOperator>::value,
    "Failed testing protocol");

static_assert(
    test_protocol_conformance<ValidNumericalFlux, dg::protocols::NumericalFlux>,
    "Failed testing protocol conformance");

// [numerical_flux_example]
struct CentralFlux : tt::ConformsTo<dg::protocols::NumericalFlux> {
  using variables_tags = tmpl::list<FieldTag>;
  using argument_tags = tmpl::list<::Tags::NormalDotFlux<FieldTag>>;
  using package_field_tags = tmpl::list<::Tags::NormalDotFlux<FieldTag>>;
  using package_extra_tags = tmpl::list<>;
  void package_data(
      const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux,
      const Scalar<DataVector>& normal_dot_flux) const noexcept {
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
    test_protocol_conformance<CentralFlux, dg::protocols::NumericalFlux>,
    "Failed testing protocol conformance");

}  // namespace
