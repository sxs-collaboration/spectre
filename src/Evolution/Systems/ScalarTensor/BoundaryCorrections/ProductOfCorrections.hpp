// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>

#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/ScalarTensor/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace ScalarTensor::BoundaryCorrections {

/*!
 * \brief Apply a boundary condition to the combined Generalized Harmonic (::gh)
 * and scalar field (::CurvedScalarWave) system using boundary corrections
 * defined separately.
 * \see gh::BoundaryCorrections and CurvedScalarWave::BoundaryCorrections.
 */
/// @{
template <
    typename DerivedGhCorrection, typename DerivedScalarCorrection,
    typename = typename DerivedGhCorrection::dg_package_field_tags,
    typename = typename DerivedScalarCorrection::dg_package_field_tags,
    typename = typename DerivedGhCorrection::dg_package_data_temporary_tags,
    typename = typename DerivedScalarCorrection::dg_package_data_temporary_tags,
    typename = typename DerivedGhCorrection::dg_package_data_volume_tags,
    typename = typename DerivedScalarCorrection::dg_package_data_volume_tags,
    typename = typename DerivedGhCorrection::dg_boundary_terms_volume_tags,
    typename = typename DerivedScalarCorrection::dg_boundary_terms_volume_tags>
class ProductOfCorrections;

template <typename DerivedGhCorrection, typename DerivedScalarCorrection,
          typename... GhDgPackagedFieldTags,
          typename... ScalarDgPackagedFieldTags,
          typename... GhDgPackageDataTemporaryTags,
          typename... ScalarDgPackageDataTemporaryTags,
          typename... GhDgPackageDataVolumeTags,
          typename... ScalarDgPackageDataVolumeTags,
          typename... GhDgBoundaryTermsVolumeTags,
          typename... ScalarDgBoundaryTermsVolumeTags>
class ProductOfCorrections<DerivedGhCorrection, DerivedScalarCorrection,
                           tmpl::list<GhDgPackagedFieldTags...>,
                           tmpl::list<ScalarDgPackagedFieldTags...>,
                           tmpl::list<GhDgPackageDataTemporaryTags...>,
                           tmpl::list<ScalarDgPackageDataTemporaryTags...>,
                           tmpl::list<GhDgPackageDataVolumeTags...>,
                           tmpl::list<ScalarDgPackageDataVolumeTags...>,
                           tmpl::list<GhDgBoundaryTermsVolumeTags...>,
                           tmpl::list<ScalarDgBoundaryTermsVolumeTags...>>
    final : public evolution::BoundaryCorrection {
 public:
  static constexpr size_t dim = 3;
  using dg_package_field_tags =
      tmpl::list<GhDgPackagedFieldTags..., ScalarDgPackagedFieldTags...>;

  using dg_package_data_temporary_tags =
      tmpl::list<GhDgPackageDataTemporaryTags...,
                 ScalarDgPackageDataTemporaryTags...>;

  using dg_package_data_primitive_tags = tmpl::list<>;

  using dg_package_data_volume_tags =
      tmpl::list<GhDgPackageDataVolumeTags...,
                 ScalarDgPackageDataVolumeTags...>;

  using dg_boundary_terms_volume_tags =
      tmpl::list<GhDgBoundaryTermsVolumeTags...,
                 ScalarDgBoundaryTermsVolumeTags...>;

  static std::string name() {
    return "Product" + pretty_type::name<DerivedGhCorrection>() + "GH" + "And" +
           pretty_type::name<DerivedScalarCorrection>() + "Scalar";
  }

  struct GhCorrection {
    using type = DerivedGhCorrection;
    static std::string name() {
      // We change the default name of the boundary correction to avoid errors
      // during option parsing
      return pretty_type::name<DerivedGhCorrection>() + "GH";
    }
    static constexpr Options::String help{
        "The Generalized Harmonic part of the product boundary condition"};
  };
  struct ScalarCorrection {
    using type = DerivedScalarCorrection;
    static std::string name() {
      // We change the default name of the boundary correction to avoid errors
      // during option parsing
      return pretty_type::name<DerivedScalarCorrection>() + "Scalar";
    }
    static constexpr Options::String help{
        "The scalar part of the product boundary condition"};
  };

  using options = tmpl::list<GhCorrection, ScalarCorrection>;

  static constexpr Options::String help = {
      "Direct product of a GH and CurvedScalarWave boundary correction. "
      "See the documentation for the two individual boundary corrections for "
      "further details."};

  ProductOfCorrections() = default;
  ProductOfCorrections(DerivedGhCorrection gh_correction,
                       DerivedScalarCorrection scalar_correction)
      : derived_gh_correction_{gh_correction},
        derived_scalar_correction_{scalar_correction} {}
  ProductOfCorrections(const ProductOfCorrections&) = default;
  ProductOfCorrections& operator=(const ProductOfCorrections&) = default;
  ProductOfCorrections(ProductOfCorrections&&) = default;
  ProductOfCorrections& operator=(ProductOfCorrections&&) = default;
  ~ProductOfCorrections() override = default;

  /// \cond
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ProductOfCorrections);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override {
    BoundaryCorrection::pup(p);
    p | derived_gh_correction_;
    p | derived_scalar_correction_;
  }

  std::unique_ptr<BoundaryCorrection> get_clone() const override {
    return std::make_unique<ProductOfCorrections>(*this);
  }

  double dg_package_data(
      const gsl::not_null<
          typename GhDgPackagedFieldTags::type*>... gh_packaged_fields,
      const gsl::not_null<
          typename ScalarDgPackagedFieldTags::type*>... scalar_packaged_fields,
      // GH variables
      const tnsr::aa<DataVector, dim, Frame::Inertial>& spacetime_metric,
      const tnsr::aa<DataVector, dim, Frame::Inertial>& pi,
      const tnsr::iaa<DataVector, dim, Frame::Inertial>& phi,
      // Scalar variables
      const Scalar<DataVector>& psi_scalar, const Scalar<DataVector>& pi_scalar,
      const tnsr::i<DataVector, dim, Frame::Inertial>& phi_scalar,
      // Temporaries
      const typename GhDgPackageDataTemporaryTags::type&... gh_temporaries,
      const typename ScalarDgPackageDataTemporaryTags::
          type&... scalar_temporaries,
      // Mesh variables
      const tnsr::i<DataVector, dim, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, dim, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
      // Volume quantities
      const typename GhDgPackageDataVolumeTags::type&... gh_volume_quantities,
      const typename ScalarDgPackageDataVolumeTags::
          type&... scalar_volume_quantities) const {
    const double gh_correction_result = derived_gh_correction_.dg_package_data(
        gh_packaged_fields..., spacetime_metric, pi, phi, gh_temporaries...,
        normal_covector, normal_vector, mesh_velocity, normal_dot_mesh_velocity,
        gh_volume_quantities...);

    const double scalar_correction_result =
        derived_scalar_correction_.dg_package_data(
            scalar_packaged_fields..., psi_scalar, pi_scalar, phi_scalar,
            scalar_temporaries..., normal_covector, normal_vector,
            mesh_velocity, normal_dot_mesh_velocity,
            scalar_volume_quantities...);
    return std::max(gh_correction_result, scalar_correction_result);
  }

  void dg_boundary_terms(
      // GH boundary corrections
      const gsl::not_null<tnsr::aa<DataVector, dim, Frame::Inertial>*>
          boundary_correction_spacetime_metric,
      const gsl::not_null<tnsr::aa<DataVector, dim, Frame::Inertial>*>
          boundary_correction_pi,
      const gsl::not_null<tnsr::iaa<DataVector, dim, Frame::Inertial>*>
          boundary_correction_phi,
      // Scalar boundary corrections
      const gsl::not_null<Scalar<DataVector>*> psi_boundary_correction_scalar,
      const gsl::not_null<Scalar<DataVector>*> pi_boundary_correction_scalar,
      const gsl::not_null<tnsr::i<DataVector, dim, Frame::Inertial>*>
          phi_boundary_correction_scalar,
      // Packaged fields
      const typename GhDgPackagedFieldTags::type&... gh_packaged_fields_int,
      const typename ScalarDgPackagedFieldTags::
          type&... scalar_packaged_fields_int,
      const typename GhDgPackagedFieldTags::type&... gh_packaged_fields_ext,
      const typename ScalarDgPackagedFieldTags::
          type&... scalar_packaged_fields_ext,
      // DG formulation
      const dg::Formulation dg_formulation) const {
    derived_gh_correction_.dg_boundary_terms(
        boundary_correction_spacetime_metric, boundary_correction_pi,
        boundary_correction_phi, gh_packaged_fields_int...,
        gh_packaged_fields_ext..., dg_formulation);

    derived_scalar_correction_.dg_boundary_terms(
        psi_boundary_correction_scalar, pi_boundary_correction_scalar,
        phi_boundary_correction_scalar, scalar_packaged_fields_int...,
        scalar_packaged_fields_ext..., dg_formulation);
  }

  const DerivedGhCorrection& gh_correction() const {
    return derived_gh_correction_;
  }

  const DerivedScalarCorrection& scalar_correction() const {
    return derived_scalar_correction_;
  }

 private:
  DerivedGhCorrection derived_gh_correction_;
  DerivedScalarCorrection derived_scalar_correction_;
};
/// @}

#if defined(SPECTRE_USE_CHARM)
/// \cond
template <typename DerivedGhCorrection, typename DerivedScalarCorrection,
          typename... GhDgPackagedFieldTags,
          typename... ScalarDgPackagedFieldTags,
          typename... GhDgPackageDataTemporaryTags,
          typename... ScalarDgPackageDataTemporaryTags,
          typename... GhDgPackageDataVolumeTags,
          typename... ScalarDgPackageDataVolumeTags,
          typename... GhDgBoundaryTermsVolumeTags,
          typename... ScalarDgBoundaryTermsVolumeTags>
PUP::able::PUP_ID ProductOfCorrections<
    DerivedGhCorrection, DerivedScalarCorrection,
    tmpl::list<GhDgPackagedFieldTags...>,
    tmpl::list<ScalarDgPackagedFieldTags...>,
    tmpl::list<GhDgPackageDataTemporaryTags...>,
    tmpl::list<ScalarDgPackageDataTemporaryTags...>,
    tmpl::list<GhDgPackageDataVolumeTags...>,
    tmpl::list<ScalarDgPackageDataVolumeTags...>,
    tmpl::list<GhDgBoundaryTermsVolumeTags...>,
    tmpl::list<ScalarDgBoundaryTermsVolumeTags...>>::my_PUP_ID = 0;  // NOLINT
/// \endcond
#endif  // SPECTRE_USE_CHARM
}  // namespace ScalarTensor::BoundaryCorrections
