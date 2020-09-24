// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples

namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
/// \endcond

namespace evolution::dg::Actions {
/*!
 * \brief Computes the time derivative for a DG time step.
 *
 * Computes the volume fluxes, the divergence of the fluxes and all additional
 * interior contributions to the time derivatives (both nonconservative products
 * and source terms). The internal mortar data is also computed.
 *
 * The general first-order hyperbolic evolution equation solved for conservative
 * systems is:
 *
 * \f{align*}{
 * \frac{\partial u_\alpha}{\partial \hat{t}}
 *  + \partial_{i}
 *   \left(F^i_\alpha - v^i_g u_\alpha\right)
 *   = S_\alpha-u_\alpha\partial_i v^i_g,
 * \f}
 *
 * where \f$F^i_{\alpha}\f$ are the fluxes when the mesh isn't moving,
 * \f$v^i_g\f$ is the velocity of the mesh, \f$u_{\alpha}\f$ are the evolved
 * variables, \f$S_{\alpha}\f$ are the source terms, and \f$\hat{t}\f$ is the
 * time in the logical frame. For evolution equations that do not have any
 * fluxes and only nonconservative products we evolve:
 *
 * \f{align*}{
 * \frac{\partial u_\alpha}{\partial \hat{t}}
 *   +\left(B^i_{\alpha\beta}-v^i_g \delta_{\alpha\beta}
 *   \right)\partial_{i}u_\beta = S_\alpha.
 * \f}
 *
 * Finally, for equations with both conservative terms and nonconservative
 * products we use:
 *
 * \f{align*}{
 * \frac{\partial u_\alpha}{\partial \hat{t}}
 *   + \partial_{i}
 *   \left(F^i_\alpha - v^i_g u_\alpha\right)
 *   +B^i_{\alpha\beta}\partial_{i}u_\beta
 *   = S_\alpha-u_\alpha\partial_i v^i_g,
 * \f}
 *
 * where \f$B^i_{\alpha\beta}\f$ is the matrix for the nonconservative products.
 *
 * The mesh velocity is added to the flux automatically if the mesh is moving.
 * That is,
 *
 * \f{align*}{
 *  F^i_{\alpha}\to F^i_{\alpha}-v^i_{g} u_{\alpha}
 * \f}
 *
 * The source terms are also altered automatically by adding:
 *
 * \f{align*}{
 *  -u_\alpha \partial_i v^i_g,
 * \f}
 *
 * For systems with equations that only contain nonconservative products, the
 * following mesh velocity is automatically added to the time derivative:
 *
 * \f{align*}{
 *  v^i_g \partial_i u_\alpha,
 * \f}
 *
 * Uses:
 * - System:
 *   - `variables_tag`
 *   - `flux_variables`
 *   - `gradient_variables`
 *   - `compute_volume_time_derivative`
 *
 * - DataBox:
 *   - Items in `system::compute_volume_time_derivative::argument_tags`
 *   - `domain::Tags::MeshVelocity<Metavariables::volume_dim>`
 *   - `Metavariables::system::variables_tag`
 *   - `Metavariables::system::flux_variables`
 *   - `Metavariables::system::gradient_variables`
 *   - `domain::Tags::DivMeshVelocity`
 *   - `DirectionsTag`,
 *   - Required interface items for `Metavariables::system::normal_dot_fluxes`
 *
 * DataBox changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - db::add_tag_prefix<Tags::Flux, variables_tag,
 *                        tmpl::size_t<system::volume_dim>, Frame::Inertial>
 *   - `Tags::dt<system::variable_tags>`
 *   - Tags::Interface<
 *     DirectionsTag, db::add_tag_prefix<Tags::NormalDotFlux, variables_tag>>
 *   - `Tags::Mortars<typename BoundaryScheme::mortar_data_tag, VolumeDim>`
 */
template <typename Metavariables>
struct ComputeTimeDerivative {
 public:
  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* /*meta*/) noexcept;  // NOLINT const

 private:
  /*
   * Computes the volume terms for a discontinuous Galerkin scheme.
   *
   * The function does the following (in order):
   *
   * 1. Compute the partial derivatives of the `System::gradient_variables`.
   *
   *    The partial derivatives are needed in the nonconservative product terms
   *    of the evolution equations. Any variable whose evolution equation does
   *    not contain a flux must contain a nonconservative product and the
   *    variable must be listed in the `System::gradient_variables` type alias.
   *    The partial derivatives are also needed for adding the moving mesh terms
   *    to the equations that do not have a flux term.
   *
   * 2. The volume time derivatives are calculated from
   *    `System::compute_volume_time_derivative`
   *
   *    The source terms and nonconservative products are contributed directly
   *    to the `dt_vars` arguments passed to the time derivative function, while
   *    the volume fluxes are computed into the `volume_fluxes` arguments. The
   *    divergence of the volume fluxes will be computed and added to the time
   *    derivatives later in the function.
   *
   * 3. If the mesh is moving the appropriate mesh velocity terms are added to
   *    the equations.
   *
   *    For equations with fluxes this means that \f$-v^i_g u_\alpha\f$ is
   *    added to the fluxes and \f$-u_\alpha \partial_i v^i_g\f$ is added
   *    to the time derivatives. For equations without fluxes
   *    \f$v^i\partial_i u_\alpha\f$ is added to the time derivatives.
   *
   * 4. Compute flux divergence contribution and add it to the time derivatives.
   *
   *    Either the weak or strong form can be used. Currently only the strong
   *    form is coded, but adding the weak form is quite easy.
   *
   *    Note that the computation of the flux divergence and adding that to the
   *    time derivative must be done *after* the mesh velocity is subtracted
   *    from the fluxes.
   */
  template <size_t Dim, typename ComputeVolumeTimeDerivatives,
            typename DbTagsList, typename... VariablesTags,
            typename... TimeDerivativeArgumentTags,
            typename... PartialDerivTags, typename... FluxVariablesTags,
            typename... TemporaryTags>
  static void volume_terms(
      gsl::not_null<db::DataBox<DbTagsList>*> box,
      gsl::not_null<Variables<tmpl::list<FluxVariablesTags...>>*> volume_fluxes,
      gsl::not_null<Variables<tmpl::list<PartialDerivTags...>>*> partial_derivs,
      gsl::not_null<Variables<tmpl::list<TemporaryTags...>>*> temporaries,
      ::dg::Formulation dg_formulation,
      tmpl::list<VariablesTags...> /*variables_tags*/,
      tmpl::list<TimeDerivativeArgumentTags...> /*meta*/) noexcept;

  template <typename... NormalDotFluxTags, typename... Args>
  static void apply_flux(
      gsl::not_null<Variables<tmpl::list<NormalDotFluxTags...>>*> boundary_flux,
      const Args&... boundary_variables) noexcept {
    Metavariables::system::normal_dot_fluxes::apply(
        make_not_null(&get<NormalDotFluxTags>(*boundary_flux))...,
        boundary_variables...);
  }

  template <typename DbTagsList>
  static void boundary_terms_nonconservative_products(
      gsl::not_null<db::DataBox<DbTagsList>*> box) noexcept;

  template <size_t VolumeDim, typename BoundaryScheme, typename DbTagsList>
  static void fill_mortar_data_for_internal_boundaries(
      gsl::not_null<db::DataBox<DbTagsList>*> box) noexcept;
};

template <typename Metavariables>
template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
          typename ActionList, typename ParallelComponent>
std::tuple<db::DataBox<DbTagsList>&&>
ComputeTimeDerivative<Metavariables>::apply(
    db::DataBox<DbTagsList>& box,
    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
    const Parallel::GlobalCache<Metavariables>& /*cache*/,
    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
    const ParallelComponent* const /*meta*/) noexcept {  // NOLINT const
  static constexpr size_t volume_dim = Metavariables::volume_dim;
  using system = typename Metavariables::system;
  using variables_tags = typename system::variables_tag::tags_list;
  using partial_derivative_tags = typename system::gradient_variables;
  using flux_variables = typename system::flux_variables;
  using compute_volume_time_derivative =
      typename system::compute_volume_time_derivative;

  const Mesh<volume_dim>& mesh = db::get<::domain::Tags::Mesh<volume_dim>>(box);
  // Allocate the Variables classes needed for the time derivative
  // computation.
  //
  // This is factored out so that we will be able to do ADER-DG/CG where a
  // spacetime polynomial is constructed by solving implicit equations in time
  // using a Picard iteration. A high-order initial guess is needed to
  // efficiently construct the ADER spacetime solution. This initial guess is
  // obtained using continuous RK methods, and so we will want to reuse
  // buffers. Thus, the volume_terms function returns by reference rather than
  // by value.
  Variables<typename compute_volume_time_derivative::temporary_tags>
      temporaries{mesh.number_of_grid_points()};
  Variables<db::wrap_tags_in<::Tags::Flux, flux_variables,
                             tmpl::size_t<volume_dim>, Frame::Inertial>>
      volume_fluxes{mesh.number_of_grid_points()};
  Variables<db::wrap_tags_in<::Tags::deriv, partial_derivative_tags,
                             tmpl::size_t<volume_dim>, Frame::Inertial>>
      partial_derivs{mesh.number_of_grid_points()};

  volume_terms<volume_dim, compute_volume_time_derivative>(
      make_not_null(&box), make_not_null(&volume_fluxes),
      make_not_null(&partial_derivs), make_not_null(&temporaries),
      Metavariables::dg_formulation, variables_tags{},
      typename compute_volume_time_derivative::argument_tags{});

  // The below if-else and fill_mortar_data_for_internal_boundaries are for
  // compatibility with the current boundary schemes. Once the current
  // boundary schemes are removed the code will be refactored to reflect that
  // change.
  if constexpr (not std::is_same_v<tmpl::list<>, flux_variables>) {
    using flux_variables_tag = ::Tags::Variables<flux_variables>;
    using fluxes_tag =
        db::add_tag_prefix<::Tags::Flux, flux_variables_tag,
                           tmpl::size_t<Metavariables::volume_dim>,
                           Frame::Inertial>;

    // We currently set the fluxes in the DataBox to interface with the
    // boundary correction communication code.
    db::mutate<fluxes_tag>(make_not_null(&box),
                           [&volume_fluxes](const auto fluxes_ptr) noexcept {
                             *fluxes_ptr = volume_fluxes;
                           });
  } else {
    boundary_terms_nonconservative_products(make_not_null(&box));
  }

  // Compute internal boundary quantities
  fill_mortar_data_for_internal_boundaries<
      volume_dim, typename Metavariables::boundary_scheme>(make_not_null(&box));
  return {std::move(box)};
}

template <typename Metavariables>
template <size_t Dim, typename ComputeVolumeTimeDerivatives,
          typename DbTagsList, typename... VariablesTags,
          typename... TimeDerivativeArgumentTags, typename... PartialDerivTags,
          typename... FluxVariablesTags, typename... TemporaryTags>
void ComputeTimeDerivative<Metavariables>::volume_terms(
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
    const gsl::not_null<Variables<tmpl::list<FluxVariablesTags...>>*>
        volume_fluxes,
    const gsl::not_null<Variables<tmpl::list<PartialDerivTags...>>*>
        partial_derivs,
    const gsl::not_null<Variables<tmpl::list<TemporaryTags...>>*> temporaries,
    const ::dg::Formulation dg_formulation,

    tmpl::list<VariablesTags...> /*variables_tags*/,
    tmpl::list<TimeDerivativeArgumentTags...> /*meta*/) noexcept {
  static constexpr bool has_partial_derivs = sizeof...(PartialDerivTags) != 0;
  static constexpr bool has_fluxes = sizeof...(FluxVariablesTags) != 0;
  static_assert(
      has_fluxes or has_partial_derivs,
      "Must have either fluxes or partial derivatives in a "
      "DG evolution scheme. This means the evolution system struct (usually in "
      "Evolution/Systems/YourSystem/System.hpp) being used does not specify "
      "any flux_variables or gradient_variables. Make sure the type aliases "
      "are defined, and that at least one of them is a non-empty list of "
      "tags.");

  using system = typename Metavariables::system;
  using variables_tag = typename system::variables_tag;
  using variables_tags = typename variables_tag::tags_list;
  using partial_derivative_tags = typename system::gradient_variables;
  using flux_variables = typename system::flux_variables;

  const Mesh<Dim>& mesh = db::get<::domain::Tags::Mesh<Dim>>(*box);
  const auto& logical_to_inertial_inverse_jacobian = db::get<
      ::domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Inertial>>(
      *box);
  const auto& evolved_vars = db::get<variables_tag>(*box);

  // Compute d_i u_\alpha for nonconservative products
  if constexpr (has_partial_derivs) {
    partial_derivatives<partial_derivative_tags>(
        partial_derivs, evolved_vars, mesh,
        logical_to_inertial_inverse_jacobian);
  }

  // Compute volume du/dt and fluxes

  // Compute volume terms that are unrelated to moving meshes
  db::mutate<db::add_tag_prefix<::Tags::dt, variables_tag>>(
      box,
      [&mesh, &partial_derivs, &temporaries, &volume_fluxes](
          const gsl::not_null<Variables<
              db::wrap_tags_in<::Tags::dt, typename variables_tag::tags_list>>*>
              dt_vars_ptr,
          const auto&... args) noexcept {
        // Silence compiler warnings since we genuinely don't always need all
        // the vars but sometimes do. This warning shows up with empty parameter
        // packs, which means the packs aren't "used" below in the
        // ComputeVolumeTimeDerivatives::apply call.
        (void)partial_derivs;
        (void)volume_fluxes;
        (void)temporaries;

        // For now just zero dt_vars. If this is a performance bottle neck we
        // can re-evaluate in the future.
        dt_vars_ptr->initialize(mesh.number_of_grid_points(), 0.0);

        ComputeVolumeTimeDerivatives::apply(
            make_not_null(&get<::Tags::dt<VariablesTags>>(*dt_vars_ptr))...,
            make_not_null(&get<FluxVariablesTags>(*volume_fluxes))...,
            make_not_null(&get<TemporaryTags>(*temporaries))...,
            get<PartialDerivTags>(*partial_derivs)..., args...);
      },
      db::get<TimeDerivativeArgumentTags>(*box)...);

  // Add volume terms for moving meshes
  if (const auto& mesh_velocity =
          db::get<::domain::Tags::MeshVelocity<Dim>>(*box);
      static_cast<bool>(mesh_velocity)) {
    db::mutate<db::add_tag_prefix<::Tags::dt, variables_tag>>(
        box,
        [&evolved_vars, &mesh_velocity, &volume_fluxes](
            const gsl::not_null<Variables<db::wrap_tags_in<
                ::Tags::dt, typename variables_tag::tags_list>>*>
                dt_vars_ptr,
            const boost::optional<Scalar<DataVector>>&
                div_mesh_velocity) noexcept {
          tmpl::for_each<flux_variables>([&div_mesh_velocity, &dt_vars_ptr,
                                          &evolved_vars, &mesh_velocity,
                                          &volume_fluxes](auto tag_v) noexcept {
            // Modify fluxes for moving mesh
            using var_tag = typename decltype(tag_v)::type;
            using flux_var_tag =
                db::add_tag_prefix<::Tags::Flux, var_tag, tmpl::size_t<Dim>,
                                   Frame::Inertial>;
            auto& flux_var = get<flux_var_tag>(*volume_fluxes);
            // Loop over all independent components of flux_var
            for (size_t flux_var_storage_index = 0;
                 flux_var_storage_index < flux_var.size();
                 ++flux_var_storage_index) {
              // Get the flux variable's tensor index, e.g. (i,j) for a F^i of
              // the spatial velocity (or some other spatial tensor).
              const auto flux_var_tensor_index =
                  flux_var.get_tensor_index(flux_var_storage_index);
              // Remove the first index from the flux tensor index, gets back
              // (j)
              const auto var_tensor_index =
                  all_but_specified_element_of(flux_var_tensor_index, 0);
              // Set flux_index to (i)
              const size_t flux_index = gsl::at(flux_var_tensor_index, 0);

              // We now need to index flux(i,j) -= u(j) * v_g(i)
              flux_var[flux_var_storage_index] -=
                  get<var_tag>(evolved_vars).get(var_tensor_index) *
                  mesh_velocity->get(flux_index);
            }

            // Modify time derivative (i.e. source terms) for moving mesh
            auto& dt_var = get<::Tags::dt<var_tag>>(*dt_vars_ptr);
            for (size_t dt_var_storage_index = 0;
                 dt_var_storage_index < dt_var.size(); ++dt_var_storage_index) {
              // This is S -> S - u d_i v^i_g
              dt_var[dt_var_storage_index] -=
                  get<var_tag>(evolved_vars)[dt_var_storage_index] *
                  get(*div_mesh_velocity);
            }
          });
        },
        db::get<::domain::Tags::DivMeshVelocity>(*box));

    // We add the mesh velocity to all equations that don't have flux terms.
    // This doesn't need to be equal to the equations that have partial
    // derivatives. For example, the scalar field evolution equation in
    // first-order form does not have any partial derivatives but still needs
    // the velocity term added. This is because the velocity term arises from
    // transforming the time derivative.
    using non_flux_tags = tmpl::list_difference<variables_tags, flux_variables>;

    db::mutate<db::add_tag_prefix<::Tags::dt, variables_tag>>(
        box, [&mesh_velocity, &partial_derivs](
                 const gsl::not_null<Variables<db::wrap_tags_in<
                     ::Tags::dt, typename variables_tag::tags_list>>*>
                     dt_vars) noexcept {
          tmpl::for_each<non_flux_tags>(
              [&dt_vars, &mesh_velocity,
               &partial_derivs](auto var_tag_v) noexcept {
                using var_tag = typename decltype(var_tag_v)::type;
                using dt_var_tag = ::Tags::dt<var_tag>;
                using deriv_var_tag =
                    ::Tags::deriv<var_tag, tmpl::size_t<Dim>, Frame::Inertial>;

                const auto& deriv_var = get<deriv_var_tag>(*partial_derivs);
                auto& dt_var = get<dt_var_tag>(*dt_vars);

                // Loop over all independent components of the derivative of the
                // variable.
                for (size_t deriv_var_storage_index = 0;
                     deriv_var_storage_index < deriv_var.size();
                     ++deriv_var_storage_index) {
                  // We grab the `deriv_tensor_index`, which would be e.g.
                  // `(i, a, b)`, so `(0, 2, 3)`
                  const auto deriv_var_tensor_index =
                      deriv_var.get_tensor_index(deriv_var_storage_index);
                  // Then we drop the derivative index (the first entry) to get
                  // `(a, b)` (or `(2, 3)`)
                  const auto dt_var_tensor_index =
                      all_but_specified_element_of(deriv_var_tensor_index, 0);
                  // Set `deriv_index` to `i` (or `0` in the example)
                  const size_t deriv_index = gsl::at(deriv_var_tensor_index, 0);
                  dt_var.get(dt_var_tensor_index) +=
                      mesh_velocity->get(deriv_index) *
                      deriv_var[deriv_var_storage_index];
                }
              });
        });
  }

  // Add the flux divergence term to du_\alpha/dt, which must be done
  // after the corrections for the moving mesh are made.
  if constexpr (has_fluxes) {
    db::mutate<db::add_tag_prefix<::Tags::dt, variables_tag>>(
        box,
        [dg_formulation, &logical_to_inertial_inverse_jacobian, &mesh,
         &volume_fluxes](const gsl::not_null<Variables<db::wrap_tags_in<
                             ::Tags::dt, typename variables_tag::tags_list>>*>
                             dt_vars_ptr) noexcept {
          if (dg_formulation == ::dg::Formulation::StrongInertial) {
            const Variables<tmpl::list<::Tags::div<FluxVariablesTags>...>>
                div_fluxes = divergence(*volume_fluxes, mesh,
                                        logical_to_inertial_inverse_jacobian);
            tmpl::for_each<flux_variables>([&dt_vars_ptr, &div_fluxes](
                                               auto var_tag_v) noexcept {
              using var_tag = typename decltype(var_tag_v)::type;
              auto& dt_var = get<::Tags::dt<var_tag>>(*dt_vars_ptr);
              const auto& div_flux_var = get<::Tags::div<
                  ::Tags::Flux<var_tag, tmpl::size_t<Dim>, Frame::Inertial>>>(
                  div_fluxes);
              for (size_t storage_index = 0; storage_index < dt_var.size();
                   ++storage_index) {
                dt_var[storage_index] -= div_flux_var[storage_index];
              }
            });
          } else if (dg_formulation == ::dg::Formulation::WeakInertial) {
            ERROR("Weak inertial DG formulation not yet implemented");
          } else {
            ERROR("Unsupported DG formulation: " << dg_formulation);
          }
        });
  } else {
    (void)dg_formulation;
  }
}

template <typename Metavariables>
template <typename DbTagsList>
void ComputeTimeDerivative<Metavariables>::
    boundary_terms_nonconservative_products(
        const gsl::not_null<db::DataBox<DbTagsList>*> box) noexcept {
  using system = typename Metavariables::system;
  using variables_tag = typename system::variables_tag;
  using DirectionsTag =
      domain::Tags::InternalDirections<Metavariables::volume_dim>;

  using interface_normal_dot_fluxes_tag = domain::Tags::Interface<
      DirectionsTag, db::add_tag_prefix<::Tags::NormalDotFlux, variables_tag>>;

  using interface_compute_item_argument_tags = tmpl::transform<
      typename Metavariables::system::normal_dot_fluxes::argument_tags,
      tmpl::bind<domain::Tags::Interface, DirectionsTag, tmpl::_1>>;

  db::mutate_apply<
      tmpl::list<interface_normal_dot_fluxes_tag>,
      tmpl::push_front<interface_compute_item_argument_tags, DirectionsTag>>(
      [](const auto boundary_fluxes_ptr,
         const std::unordered_set<Direction<Metavariables::volume_dim>>&
             internal_directions,
         const auto&... tensors) noexcept {
        for (const auto& direction : internal_directions) {
          apply_flux(make_not_null(&boundary_fluxes_ptr->at(direction)),
                     tensors.at(direction)...);
        }
      },
      box);
}

template <typename Metavariables>
template <size_t VolumeDim, typename BoundaryScheme, typename DbTagsList>
void ComputeTimeDerivative<Metavariables>::
    fill_mortar_data_for_internal_boundaries(
        const gsl::not_null<db::DataBox<DbTagsList>*> box) noexcept {
  using temporal_id_tag = typename BoundaryScheme::temporal_id_tag;
  using all_mortar_data_tag =
      ::Tags::Mortars<typename BoundaryScheme::mortar_data_tag, VolumeDim>;
  using boundary_data_computer =
      typename BoundaryScheme::boundary_data_computer;

  // Collect data on element interfaces
  auto boundary_data_on_interfaces =
      interface_apply<domain::Tags::InternalDirections<VolumeDim>,
                      boundary_data_computer>(*box);

  // Project collected data to all internal mortars and store in DataBox
  const auto& element = db::get<domain::Tags::Element<VolumeDim>>(*box);
  const auto& face_meshes =
      get<domain::Tags::Interface<domain::Tags::InternalDirections<VolumeDim>,
                                  domain::Tags::Mesh<VolumeDim - 1>>>(*box);
  const auto& mortar_meshes =
      get<::Tags::Mortars<domain::Tags::Mesh<VolumeDim - 1>, VolumeDim>>(*box);
  const auto& mortar_sizes =
      get<::Tags::Mortars<::Tags::MortarSize<VolumeDim - 1>, VolumeDim>>(*box);
  const auto& temporal_id = get<temporal_id_tag>(*box);
  for (const auto& [direction, neighbors] : element.neighbors()) {
    const auto& face_mesh = face_meshes.at(direction);
    for (const auto& neighbor : neighbors) {
      const auto mortar_id = std::make_pair(direction, neighbor);
      const auto& mortar_mesh = mortar_meshes.at(mortar_id);
      const auto& mortar_size = mortar_sizes.at(mortar_id);

      // Project the data from the face to the mortar.
      // Where no projection is necessary we `std::move` the data directly to
      // avoid a copy. We can't move the data or modify it in-place when
      // projecting, because in that case the face may touch two mortars so we
      // need to keep the data around.
      auto boundary_data_on_mortar =
          ::dg::needs_projection(face_mesh, mortar_mesh, mortar_size)
              ? boundary_data_on_interfaces.at(direction).project_to_mortar(
                    face_mesh, mortar_mesh, mortar_size)
              : std::move(boundary_data_on_interfaces.at(direction));

      // Store the boundary data on this side of the mortar
      db::mutate<all_mortar_data_tag>(
          box, [&mortar_id, &temporal_id, &boundary_data_on_mortar](
                   const gsl::not_null<db::item_type<all_mortar_data_tag>*>
                       all_mortar_data) noexcept {
            all_mortar_data->at(mortar_id).local_insert(
                temporal_id, std::move(boundary_data_on_mortar));
          });
    }
  }
}
}  // namespace evolution::dg::Actions
