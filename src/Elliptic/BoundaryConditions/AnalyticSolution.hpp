// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <ostream>
#include <pup.h>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/BoundaryConditions/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic::BoundaryConditions {
namespace detail {

template <typename Solution, size_t Dim, typename Tag, typename = std::void_t<>>
struct has_boundary_variables : std::false_type {};

template <typename Solution, size_t Dim, typename Tag>
struct has_boundary_variables<
    Solution, Dim, Tag,
    std::void_t<decltype(std::declval<const Solution&>().variables(
        std::declval<const tnsr::I<DataVector, Dim, Frame::Inertial>&>(),
        tmpl::list<Tag>{}))>> : std::true_type {};

template <typename Solution, size_t Dim, typename Tag>
constexpr bool has_boundary_variables_v =
    has_boundary_variables<Solution, Dim, Tag>::value;

}  // namespace detail

/// \cond
template <typename System, size_t Dim = System::volume_dim,
          typename FieldTags = typename System::primal_fields,
          typename FluxTags = typename System::primal_fluxes>
struct AnalyticSolution;
/// \endcond

/*!
 * \brief Impose the analytic solution on the boundary.
 *
 * The user can select to impose the analytic solution as Dirichlet or
 * Neumann boundary conditions for each field separately.  Dirichlet
 * boundary conditions are imposed on the fields and Neumann boundary
 * conditions are imposed on the fluxes.
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
    using type = std::unique_ptr<elliptic::analytic_data::InitialGuess>;
    static constexpr Options::String help = {
        "The analytic data to impose on the boundary"};
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
      solution_ = serialize_and_deserialize<
          std::unique_ptr<elliptic::analytic_data::InitialGuess>>(
          rhs.solution_);
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
      std::unique_ptr<elliptic::analytic_data::InitialGuess> solution,
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
    using factory_classes =
        typename Metavariables::factory_creation::factory_classes;
    call_with_dynamic_type<
        void, tmpl::at<factory_classes, elliptic::analytic_data::InitialGuess>>(
        solution_.get(), [this, &face_inertial_coords, &face_normal, &fields...,
                          &n_dot_fluxes...](const auto* const derived) {
          const auto impose_boundary_condition = [this, &face_inertial_coords,
                                                  &face_normal, derived](
                                                     auto field_tag_v,
                                                     auto flux_tag_v,
                                                     const auto field,
                                                     const auto n_dot_flux) {
            using field_tag = std::decay_t<decltype(field_tag_v)>;
            using flux_tag = std::decay_t<decltype(flux_tag_v)>;
            using derived_type = std::decay_t<decltype(*derived)>;
            switch (get<elliptic::Tags::BoundaryConditionType<field_tag>>(
                boundary_condition_types_)) {
              case elliptic::BoundaryConditionType::Dirichlet: {
                if constexpr (detail::has_boundary_variables_v<
                                  derived_type, Dim, field_tag>) {
                  const auto solution_vars = derived->variables(
                      face_inertial_coords, tmpl::list<field_tag>{});
                  *field = get<field_tag>(solution_vars);
                } else {
                  ERROR(
                      "The analytic data does not provide the field required "
                      "for this Dirichlet boundary condition.");
                }
                break;
              }
              case elliptic::BoundaryConditionType::Neumann: {
                if constexpr (detail::has_boundary_variables_v<derived_type,
                                                               Dim, flux_tag>) {
                  const auto solution_vars = derived->variables(
                      face_inertial_coords, tmpl::list<flux_tag>{});
                  normal_dot_flux(n_dot_flux, face_normal,
                                  get<flux_tag>(solution_vars));
                } else {
                  ERROR(
                      "The analytic data does not provide the flux required "
                      "for this Neumann boundary condition.");
                }
                break;
              }
              default:
                ERROR("Unsupported boundary condition type: "
                      << get<elliptic::Tags::BoundaryConditionType<field_tag>>(
                             boundary_condition_types_));
            }
          };
          EXPAND_PACK_LEFT_TO_RIGHT(impose_boundary_condition(
              FieldTags{}, FluxTags{}, fields, n_dot_fluxes));
        });
  }

  using argument_tags_linearized = tmpl::list<>;
  using volume_tags_linearized = tmpl::list<>;

  void apply_linearized(
      const gsl::not_null<typename FieldTags::type*>... fields,
      const gsl::not_null<typename FieldTags::type*>... n_dot_fluxes,
      const TensorMetafunctions::prepend_spatial_index<
          typename FieldTags::type, Dim, UpLo::Lo,
          Frame::Inertial>&... /*deriv_fields*/) const {
    const auto impose_boundary_condition =
        [this](auto field_tag_v, const auto field, const auto n_dot_flux) {
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
  std::unique_ptr<elliptic::analytic_data::InitialGuess> solution_{nullptr};
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
