// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <ostream>
#include <pup.h>
#include <string>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/BoundaryConditions/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Options/String.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace elliptic::BoundaryConditions {

/// \cond
template <typename System, size_t Dim = System::volume_dim,
          typename FieldTags = typename System::primal_fields,
          typename FluxTags = typename System::primal_fluxes>
struct AnalyticSolution;
/// \endcond

/*!
 * \brief Impose the analytic solution on the boundary. Works only if an
 * analytic solution exists.
 *
 * The analytic solution is retrieved from `::Tags::AnalyticSolutionsBase`. It
 * must hold solutions for both the `System::primal_fields` and the
 * `System::primal_fluxes`. The user can select to impose the analytic solution
 * as Dirichlet or Neumann boundary conditions for each field separately.
 * Dirichlet boundary conditions are imposed on the fields and Neumann boundary
 * conditions are imposed on the fluxes.
 *
 * See `elliptic::Actions::InitializeAnalyticSolutions` for an action that can
 * add the analytic solutions to the DataBox.
 */
template <typename System, size_t Dim, typename... FieldTags,
          typename... FluxTags>
class AnalyticSolution<System, Dim, tmpl::list<FieldTags...>,
                       tmpl::list<FluxTags...>>
    : public BoundaryCondition<Dim> {
 private:
  using Base = BoundaryCondition<Dim>;

 public:
  struct Solution {
    using type = std::unique_ptr<elliptic::analytic_data::AnalyticSolution>;
    static constexpr Options::String help = {
        "The analytic solution to impose on the boundary"};
  };

  using options =
      tmpl::list<Solution,
                 elliptic::OptionTags::BoundaryConditionType<FieldTags>...>;
  static constexpr Options::String help =
      "Boundary conditions from the analytic solution";

  AnalyticSolution() = default;
  AnalyticSolution(const AnalyticSolution& rhs) : Base(rhs) { *this = rhs; }
  AnalyticSolution& operator=(const AnalyticSolution& rhs) {
    if (rhs.solution_ != nullptr) {
      solution_ = rhs.solution_->get_clone();
    } else {
      solution_ = nullptr;
    }
    boundary_condition_types_ = rhs.boundary_condition_types_;
    return *this;
  }
  AnalyticSolution(AnalyticSolution&&) = default;
  AnalyticSolution& operator=(AnalyticSolution&&) = default;
  ~AnalyticSolution() = default;

  /// \cond
  explicit AnalyticSolution(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(AnalyticSolution);
  /// \endcond

  /// Select which `elliptic::BoundaryConditionType` to apply for each field
  explicit AnalyticSolution(
      std::unique_ptr<elliptic::analytic_data::AnalyticSolution> solution,
      // This pack expansion repeats the type `elliptic::BoundaryConditionType`
      // for each system field
      const typename elliptic::OptionTags::BoundaryConditionType<
          FieldTags>::type... boundary_condition_types)
      : solution_(std::move(solution)),
        boundary_condition_types_{boundary_condition_types...} {}

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override {
    return std::make_unique<AnalyticSolution>(*this);
  }

  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override {
    std::vector<elliptic::BoundaryConditionType> result{};
    const auto collect = [&result](
                             const auto tag_v,
                             const elliptic::BoundaryConditionType bc_type) {
      using tag = std::decay_t<decltype(tag_v)>;
      for (size_t i = 0; i < tag::type::size(); ++i) {
        result.push_back(bc_type);
      }
    };
    EXPAND_PACK_LEFT_TO_RIGHT(collect(
        FieldTags{}, get<elliptic::Tags::BoundaryConditionType<FieldTags>>(
                         boundary_condition_types_)));
    return result;
  }

  using argument_tags =
      tmpl::list<Parallel::Tags::Metavariables,
                 domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<
                     Dim, Frame::Inertial>>>;
  using volume_tags = tmpl::list<Parallel::Tags::Metavariables>;

  template <typename Metavariables>
  void apply(const gsl::not_null<typename FieldTags::type*>... fields,
             const gsl::not_null<typename FieldTags::type*>... n_dot_fluxes,
             const TensorMetafunctions::prepend_spatial_index<
                 typename FieldTags::type, Dim, UpLo::Lo,
                 Frame::Inertial>&... /*deriv_fields*/,
             const Metavariables& /*meta*/,
             const tnsr::I<DataVector, Dim>& face_inertial_coords,
             const tnsr::i<DataVector, Dim>& face_normal) const {
    // Retrieve variables for both Dirichlet and Neumann conditions, then decide
    // which to impose. We could also retrieve either the field for the flux for
    // each field individually based on the selection, but that would incur the
    // overhead of calling into the analytic solution multiple times and
    // possibly computing temporary quantities multiple times. This performance
    // consideration is probably irrelevant because the boundary conditions are
    // only evaluated once at the beginning of the solve.
    using analytic_tags = tmpl::list<FieldTags..., FluxTags...>;
    using factory_classes =
        typename Metavariables::factory_creation::factory_classes;
    const auto solution_vars = call_with_dynamic_type<
        tuples::tagged_tuple_from_typelist<analytic_tags>,
        tmpl::at<factory_classes, elliptic::analytic_data::AnalyticSolution>>(
        solution_.get(), [&face_inertial_coords](const auto* const derived) {
          return derived->variables(face_inertial_coords, analytic_tags{});
        });
    const auto impose_boundary_condition = [this, &solution_vars, &face_normal](
                                               auto field_tag_v,
                                               auto flux_tag_v,
                                               const auto field,
                                               const auto n_dot_flux) {
      using field_tag = decltype(field_tag_v);
      using flux_tag = decltype(flux_tag_v);
      switch (get<elliptic::Tags::BoundaryConditionType<field_tag>>(
          boundary_condition_types_)) {
        case elliptic::BoundaryConditionType::Dirichlet:
          *field = get<field_tag>(solution_vars);
          break;
        case elliptic::BoundaryConditionType::Neumann:
          normal_dot_flux(n_dot_flux, face_normal,
                          get<flux_tag>(solution_vars));
          break;
        default:
          ERROR("Unsupported boundary condition type: "
                << get<elliptic::Tags::BoundaryConditionType<field_tag>>(
                       boundary_condition_types_));
      }
    };
    EXPAND_PACK_LEFT_TO_RIGHT(impose_boundary_condition(FieldTags{}, FluxTags{},
                                                        fields, n_dot_fluxes));
  }

  using argument_tags_linearized = tmpl::list<>;
  using volume_tags_linearized = tmpl::list<>;

  void apply_linearized(
      const gsl::not_null<typename FieldTags::type*>... fields,
      const gsl::not_null<typename FieldTags::type*>... n_dot_fluxes,
      const TensorMetafunctions::prepend_spatial_index<
          typename FieldTags::type, Dim, UpLo::Lo,
          Frame::Inertial>&... /*deriv_fields*/) const {
    const auto impose_boundary_condition = [this](auto field_tag_v,
                                                  const auto field,
                                                  const auto n_dot_flux) {
      using field_tag = decltype(field_tag_v);
      switch (get<elliptic::Tags::BoundaryConditionType<field_tag>>(
          boundary_condition_types_)) {
        case elliptic::BoundaryConditionType::Dirichlet:
          for (auto& field_component : *field) {
            field_component = 0.;
          }
          break;
        case elliptic::BoundaryConditionType::Neumann:
          for (auto& n_dot_flux_component : *n_dot_flux) {
            n_dot_flux_component = 0.;
          }
          break;
        default:
          ERROR("Unsupported boundary condition type: "
                << get<elliptic::Tags::BoundaryConditionType<field_tag>>(
                       boundary_condition_types_));
      }
    };
    EXPAND_PACK_LEFT_TO_RIGHT(
        impose_boundary_condition(FieldTags{}, fields, n_dot_fluxes));
  }

  // NOLINTNEXTLINE
  void pup(PUP::er& p) override;

 private:
  std::unique_ptr<elliptic::analytic_data::AnalyticSolution> solution_{nullptr};
  tuples::TaggedTuple<elliptic::Tags::BoundaryConditionType<FieldTags>...>
      boundary_condition_types_{};
};

template <typename System, size_t Dim, typename... FieldTags,
          typename... FluxTags>
void AnalyticSolution<System, Dim, tmpl::list<FieldTags...>,
                      tmpl::list<FluxTags...>>::pup(PUP::er& p) {
  Base::pup(p);
  p | solution_;
  p | boundary_condition_types_;
}

/// \cond
template <typename System, size_t Dim, typename... FieldTags,
          typename... FluxTags>
PUP::able::PUP_ID AnalyticSolution<System, Dim, tmpl::list<FieldTags...>,
                                   tmpl::list<FluxTags...>>::my_PUP_ID =
    0;  // NOLINT
/// \endcond

}  // namespace elliptic::BoundaryConditions
