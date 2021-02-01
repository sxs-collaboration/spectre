// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/range/join.hpp>
#include <cstddef>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Elliptic/DiscontinuousGalerkin/Penalty.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleMortarData.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// #include "Parallel/Printf.hpp"

namespace elliptic::dg {

namespace Tags {
struct PerpendicularNumPoints : db::SimpleTag {
  using type = size_t;
};

struct ElementSize : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <typename Tag>
struct NormalDotFluxForJump : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};
}  // namespace Tags

/*!
 * \brief Data that is projected to mortars and communicated across element
 * boundaries
 */
template <typename PrimalVars, typename AuxiliaryVars>
using BoundaryData = ::dg::SimpleBoundaryData<
    tmpl::append<db::wrap_tags_in<::Tags::NormalDotFlux, PrimalVars>,
                 db::wrap_tags_in<::Tags::NormalDotFlux, AuxiliaryVars>,
                 db::wrap_tags_in<Tags::NormalDotFluxForJump, PrimalVars>,
                 tmpl::list<Tags::ElementSize>>,
    tmpl::list<Tags::PerpendicularNumPoints>>;

template <typename PrimalMortarVars, typename AuxiliaryMortarVars, size_t Dim>
BoundaryData<PrimalMortarVars, AuxiliaryMortarVars>
zero_boundary_data_on_mortar(
    const Direction<Dim>& direction, const Mesh<Dim>& mesh,
    const Scalar<DataVector>& face_normal_magnitude,
    const Mesh<Dim - 1>& mortar_mesh,
    const ::dg::MortarSize<Dim - 1>& mortar_size) noexcept {
  const auto face_mesh = mesh.slice_away(direction.dimension());
  const size_t face_num_points = face_mesh.number_of_grid_points();
  BoundaryData<PrimalMortarVars, AuxiliaryMortarVars> boundary_data{
      face_num_points};
  boundary_data.field_data.initialize(face_num_points, 0.);
  get<Tags::PerpendicularNumPoints>(boundary_data.extra_data) =
      mesh.extents(direction.dimension());
  get(get<Tags::ElementSize>(boundary_data.field_data)) =
      2. / get(face_normal_magnitude);
  return ::dg::needs_projection(face_mesh, mortar_mesh, mortar_size)
             ? boundary_data.project_to_mortar(face_mesh, mortar_mesh,
                                               mortar_size)
             : boundary_data;
}

/*!
 * \brief Boundary data on both sides of a mortar
 */
// This is a struct so it can be used to deduce the template parameters
template <typename PrimalVars, typename AuxiliaryVars>
struct MortarData
    : ::dg::SimpleMortarData<size_t, BoundaryData<PrimalVars, AuxiliaryVars>,
                             BoundaryData<PrimalVars, AuxiliaryVars>> {};

namespace Tags {
template <typename PrimalVars, typename AuxiliaryVars>
struct MortarData : db::SimpleTag {
  using type = elliptic::dg::MortarData<PrimalVars, AuxiliaryVars>;
};
}  // namespace Tags

namespace detail {
template <typename System, bool Linearized,
          typename PrimalFields = typename System::primal_fields,
          typename AuxiliaryFields = typename System::auxiliary_fields,
          typename PrimalFluxes = typename System::primal_fluxes,
          typename AuxiliaryFluxes = typename System::auxiliary_fluxes>
struct DgOperatorImpl;

template <typename System, bool Linearized, typename... PrimalFields,
          typename... AuxiliaryFields, typename... PrimalFluxes,
          typename... AuxiliaryFluxes>
struct DgOperatorImpl<System, Linearized, tmpl::list<PrimalFields...>,
                      tmpl::list<AuxiliaryFields...>,
                      tmpl::list<PrimalFluxes...>,
                      tmpl::list<AuxiliaryFluxes...>> {
  static constexpr size_t Dim = System::volume_dim;
  using FluxesComputer = typename System::fluxes_computer;
  using SourcesComputer = typename System::sources_computer;

  struct AllDirections {
    bool operator()(const Direction<Dim>& /*unused*/) const noexcept {
      return true;
    }
  };

  template <bool DataIsZero, typename... PrimalVars, typename... AuxiliaryVars,
            typename... PrimalMortarVars, typename... AuxiliaryMortarVars,
            typename TemporalId, typename ApplyBoundaryCondition,
            typename... FluxesArgs, typename... SourcesArgs,
            typename DirectionsPredicate = AllDirections>
  static void prepare_mortar_data(
      const gsl::not_null<Variables<tmpl::list<AuxiliaryVars...>>*>
          auxiliary_vars,
      const gsl::not_null<
          ::dg::MortarMap<Dim, MortarData<tmpl::list<PrimalMortarVars...>,
                                          tmpl::list<AuxiliaryMortarVars...>>>*>
          all_mortar_data,
      const Variables<tmpl::list<PrimalVars...>>& primal_vars,
      const Element<Dim>& element, const Mesh<Dim>& mesh,
      const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
          inv_jacobian,
      const std::unordered_map<Direction<Dim>, tnsr::i<DataVector, Dim>>&
          internal_face_normals,
      const std::unordered_map<Direction<Dim>, tnsr::i<DataVector, Dim>>&
          external_face_normals,
      const std::unordered_map<Direction<Dim>, Scalar<DataVector>>&
          internal_face_normal_magnitudes,
      const std::unordered_map<Direction<Dim>, Scalar<DataVector>>&
          external_face_normal_magnitudes,
      const ::dg::MortarMap<Dim, Mesh<Dim - 1>>& all_mortar_meshes,
      const ::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>& all_mortar_sizes,
      const TemporalId& temporal_id,
      const ApplyBoundaryCondition& apply_boundary_condition,
      const std::tuple<FluxesArgs...>& fluxes_args,
      const std::tuple<SourcesArgs...>& sources_args,
      const DirectionMap<Dim, std::tuple<FluxesArgs...>>&
          fluxes_args_on_internal_faces,
      const DirectionMap<Dim, std::tuple<FluxesArgs...>>&
          fluxes_args_on_external_faces,
      const DirectionsPredicate& directions_predicate =
          AllDirections{}) noexcept {
    static_assert(
        sizeof...(PrimalVars) == sizeof...(PrimalFields) and
            sizeof...(AuxiliaryVars) == sizeof...(AuxiliaryFields),
        "The number of variables must match the number of system fields.");
    static_assert(
        (std::is_same_v<typename PrimalVars::type,
                        typename PrimalFields::type> and
         ...) and
            (std::is_same_v<typename AuxiliaryVars::type,
                            typename AuxiliaryFields::type> and
             ...),
        "The variables must have the same tensor types as the system fields.");
#ifdef SPECTRE_DEBUG
    for (size_t d = 0; d < Dim; ++d) {
      ASSERT(mesh.basis(d) == Spectral::Basis::Legendre and
                 mesh.quadrature(d) == Spectral::Quadrature::GaussLobatto,
             "The elliptic DG operator is currently only implemented for "
             "Legendre-Gauss-Lobatto grids, but found basis '"
                 << mesh.basis(d) << "' and quadrature '" << mesh.quadrature(d)
                 << "' in dimension " << d << ".");
    }
#endif  // SPECTRE_DEBUG
    const size_t num_points = mesh.number_of_grid_points();

    // This function and the one below allocate various Variables to compute
    // intermediate quantities. It could be a performance optimization to reduce
    // the number of these allocations and/or move some of the memory buffers
    // into the DataBox to keep them around permanently. This decision should be
    // informed by profiling.

    // Compute the auxiliary variables
    Variables<tmpl::list<AuxiliaryFluxes...>> auxiliary_fluxes{};
    if constexpr (DataIsZero) {
      auxiliary_vars->initialize(num_points, 0.);
    } else {
      auxiliary_fluxes.initialize(num_points);
      std::apply(
          [&auxiliary_fluxes,
           &primal_vars](const auto&... expanded_fluxes_args) noexcept {
            FluxesComputer::apply(
                make_not_null(&get<AuxiliaryFluxes>(auxiliary_fluxes))...,
                expanded_fluxes_args..., get<PrimalVars>(primal_vars)...);
          },
          fluxes_args);
      divergence(auxiliary_vars, auxiliary_fluxes, mesh, inv_jacobian);
      // Parallel::printf("vars:\n%s\n", primal_vars);
      // Parallel::printf("aux vars:\n%s\n\n", *auxiliary_vars);
      *auxiliary_vars *= -1.;
      std::apply(
          [&auxiliary_vars,
           &primal_vars](const auto&... expanded_sources_args) noexcept {
            SourcesComputer::apply(
                make_not_null(&get<AuxiliaryVars>(*auxiliary_vars))...,
                expanded_sources_args..., get<PrimalVars>(primal_vars)...);
          },
          sources_args);
      *auxiliary_vars *= -1.;
    }

    // Populate the mortar data on this element's side of the boundary so it's
    // ready to be sent to neighbors.
    for (const auto& direction : [&element]() noexcept -> decltype(auto) {
           if constexpr (DataIsZero) {
             // Skipping internal boundaries for zero data because they won't
             // contribute boundary corrections anyway.
             return element.external_boundaries();
           } else {
             return boost::join(element.internal_boundaries(),
                                element.external_boundaries());
           };
         }()) {
      if (not directions_predicate(direction)) {
        continue;
      }
      const bool is_internal = element.neighbors().contains(direction);
      const auto face_mesh = mesh.slice_away(direction.dimension());
      const size_t face_num_points = face_mesh.number_of_grid_points();
      const auto& face_normal = is_internal
                                    ? internal_face_normals.at(direction)
                                    : external_face_normals.at(direction);
      const auto& face_normal_magnitude =
          is_internal ? internal_face_normal_magnitudes.at(direction)
                      : external_face_normal_magnitudes.at(direction);
      const auto& fluxes_args_on_face =
          is_internal ? fluxes_args_on_internal_faces.at(direction)
                      : fluxes_args_on_external_faces.at(direction);
      const size_t slice_index = index_to_slice_at(mesh.extents(), direction);
      Variables<tmpl::list<PrimalFluxes..., AuxiliaryFluxes...>> fluxes_on_face{
          face_num_points};
      BoundaryData<tmpl::list<PrimalMortarVars...>,
                   tmpl::list<AuxiliaryMortarVars...>>
          boundary_data{face_num_points};
      if constexpr (DataIsZero) {
        // Just setting all boundary field data to zero. Variable-independent
        // data such as the element size will be set below.
        boundary_data.field_data.initialize(face_num_points, 0.);
      } else {
        // Compute n.F_v
        fluxes_on_face.assign_subset(
            data_on_slice(auxiliary_fluxes, mesh.extents(),
                          direction.dimension(), slice_index));
        EXPAND_PACK_LEFT_TO_RIGHT(normal_dot_flux(
            make_not_null(&get<::Tags::NormalDotFlux<AuxiliaryMortarVars>>(
                boundary_data.field_data)),
            face_normal, get<AuxiliaryFluxes>(fluxes_on_face)));
        // Compute n.F_u
        //
        // For the internal penalty flux we can already slice the n.F_u to the
        // faces at this point, before the boundary corrections have been added
        // to the auxiliary variables. The reason is essentially that the
        // internal penalty flux uses average(grad(u)) in place of average(v),
        // i.e. the raw primal field derivatives without boundary corrections.
        const auto auxiliary_vars_on_face =
            data_on_slice(*auxiliary_vars, mesh.extents(),
                          direction.dimension(), slice_index);
        std::apply(
            [&fluxes_on_face, &auxiliary_vars_on_face](
                const auto&... expanded_fluxes_args_on_face) noexcept {
              FluxesComputer::apply(
                  make_not_null(&get<PrimalFluxes>(fluxes_on_face))...,
                  expanded_fluxes_args_on_face...,
                  get<AuxiliaryVars>(auxiliary_vars_on_face)...);
            },
            fluxes_args_on_face);
        EXPAND_PACK_LEFT_TO_RIGHT(normal_dot_flux(
            make_not_null(&get<::Tags::NormalDotFlux<PrimalMortarVars>>(
                boundary_data.field_data)),
            face_normal, get<PrimalFluxes>(fluxes_on_face)));
        // Compute n.F_u(n.F_v) for jump term, re-using the memory buffer from
        // above
        std::apply(
            [&fluxes_on_face, &boundary_data](
                const auto&... expanded_fluxes_args_on_face) noexcept {
              FluxesComputer::apply(
                  make_not_null(&get<PrimalFluxes>(fluxes_on_face))...,
                  expanded_fluxes_args_on_face...,
                  get<::Tags::NormalDotFlux<AuxiliaryMortarVars>>(
                      boundary_data.field_data)...);
            },
            fluxes_args_on_face);
        EXPAND_PACK_LEFT_TO_RIGHT(normal_dot_flux(
            make_not_null(&get<Tags::NormalDotFluxForJump<PrimalMortarVars>>(
                boundary_data.field_data)),
            face_normal, get<PrimalFluxes>(fluxes_on_face)));
      }

      // Collect the remaining data that's needed on both sides of the boundary
      // These are actually constant throughout the solve, so a performance
      // optimization could be to store them in the DataBox.
      get<Tags::PerpendicularNumPoints>(boundary_data.extra_data) =
          mesh.extents(direction.dimension());
      get(get<Tags::ElementSize>(boundary_data.field_data)) =
          2. / get(face_normal_magnitude);

      if (is_internal) {
        if constexpr (not DataIsZero) {
          // Project boundary data on internal faces to mortars
          for (const auto& neighbor_id : element.neighbors().at(direction)) {
            const ::dg::MortarId<Dim> mortar_id{direction, neighbor_id};
            const auto& mortar_mesh = all_mortar_meshes.at(mortar_id);
            const auto& mortar_size = all_mortar_sizes.at(mortar_id);
            auto projected_boundary_data =
                ::dg::needs_projection(face_mesh, mortar_mesh, mortar_size)
                    ? boundary_data.project_to_mortar(face_mesh, mortar_mesh,
                                                      mortar_size)
                    : std::move(boundary_data);
            (*all_mortar_data)[mortar_id].local_insert(
                temporal_id, std::move(projected_boundary_data));
          }  // loop neighbors in direction
        }
      } else {
        // No need to do projections on external boundaries
        const ::dg::MortarId<Dim> mortar_id{
            direction, ElementId<Dim>::external_boundary_id()};
        (*all_mortar_data)[mortar_id].local_insert(temporal_id, boundary_data);

        // -------------------------
        // Apply boundary conditions
        // -------------------------
        //
        // To apply boundary conditions we fill the boundary data with
        // "exterior" or "ghost" data and set it as remote mortar data, so
        // external boundaries behave just like internal boundaries when
        // applying boundary corrections.
        //
        // We only apply _linearized_ boundary conditions here, since the DG
        // operator must be linear. Note that even for linear elliptic equations
        // we typically apply boundary conditions with a constant, and therefore
        // nonlinear, contribution. Standard examples are inhomogeneous (i.e.
        // non-zero) Dirichlet or Neumann boundary conditions. This nonlinear
        // contribution can be added to the fixed sources using the action
        // `ImposeInhomogeneousBoundaryConditionsOnSource` below, leaving only
        // the linearized boundary conditions here. For standard constant
        // Dirichlet or Neumann boundary conditions the linearization is of
        // course just zero.

        // TODO: Invoke the linearized boundary conditions, passing the
        // Dirichlet fields and the interior n.F_u by reference. Boundary
        // conditions are imposed by modifying the data. Note that all data
        // passed to the boundary conditions is taken from the "interior" side
        // of the boundary, i.e. with a normal vector that points _out_ of the
        // computational domain. Setting the Dirichlet fields to zero for now to
        // impose zero Dirichlet boundary conditions.
        auto dirichlet_vars = data_on_slice(primal_vars, mesh.extents(),
                                            direction.dimension(), slice_index);
        apply_boundary_condition(
            direction, make_not_null(&get<PrimalVars>(dirichlet_vars))...,
            make_not_null(&get<::Tags::NormalDotFlux<PrimalMortarVars>>(
                boundary_data.field_data))...);

        // The n.F_u (Neumann-type conditions) are done, but we have to compute
        // the n.F_v (Dirichlet-type conditions) from the Dirichlet fields. We
        // re-use the memory buffer from above.
        std::apply(
            [&fluxes_on_face, &dirichlet_vars](
                const auto&... expanded_fluxes_args_on_face) noexcept {
              FluxesComputer::apply(
                  make_not_null(&get<AuxiliaryFluxes>(fluxes_on_face))...,
                  expanded_fluxes_args_on_face...,
                  get<PrimalVars>(dirichlet_vars)...);
            },
            fluxes_args_on_face);
        EXPAND_PACK_LEFT_TO_RIGHT(normal_dot_flux(
            make_not_null(&get<::Tags::NormalDotFlux<AuxiliaryMortarVars>>(
                boundary_data.field_data)),
            face_normal, get<AuxiliaryFluxes>(fluxes_on_face)));

        // Invert the sign of the fluxes to account for the inverted normal on
        // exterior faces. Also multiply by 2 and add the interior fluxes to
        // impose the boundary conditions on the _average_ instead of just
        // setting the fields on the exterior.
        const auto invert_sign_and_impose_on_average =
            [](const auto exterior_n_dot_flux,
               const auto& interior_n_dot_flux) {
              for (size_t i = 0; i < interior_n_dot_flux.size(); ++i) {
                (*exterior_n_dot_flux)[i] *= -2.;
                (*exterior_n_dot_flux)[i] += interior_n_dot_flux[i];
              }
            };
        EXPAND_PACK_LEFT_TO_RIGHT(invert_sign_and_impose_on_average(
            make_not_null(&get<::Tags::NormalDotFlux<PrimalMortarVars>>(
                boundary_data.field_data)),
            get<::Tags::NormalDotFlux<PrimalMortarVars>>(
                all_mortar_data->at(mortar_id)
                    .local_data(temporal_id)
                    .field_data)));
        EXPAND_PACK_LEFT_TO_RIGHT(invert_sign_and_impose_on_average(
            make_not_null(&get<::Tags::NormalDotFlux<AuxiliaryMortarVars>>(
                boundary_data.field_data)),
            get<::Tags::NormalDotFlux<AuxiliaryMortarVars>>(
                all_mortar_data->at(mortar_id)
                    .local_data(temporal_id)
                    .field_data)));

        // Compute n.F_u(n.F_v) for jump term
        std::apply(
            [&fluxes_on_face, &boundary_data](
                const auto&... expanded_fluxes_args_on_face) noexcept {
              FluxesComputer::apply(
                  make_not_null(&get<PrimalFluxes>(fluxes_on_face))...,
                  expanded_fluxes_args_on_face...,
                  get<::Tags::NormalDotFlux<AuxiliaryMortarVars>>(
                      boundary_data.field_data)...);
            },
            fluxes_args_on_face);
        EXPAND_PACK_LEFT_TO_RIGHT(normal_dot_flux(
            make_not_null(&get<Tags::NormalDotFluxForJump<PrimalMortarVars>>(
                boundary_data.field_data)),
            face_normal, get<PrimalFluxes>(fluxes_on_face)));
        const auto invert_sign = [](const auto exterior_n_dot_flux) {
          for (size_t i = 0; i < exterior_n_dot_flux->size(); ++i) {
            (*exterior_n_dot_flux)[i] *= -1.;
          }
        };
        EXPAND_PACK_LEFT_TO_RIGHT(invert_sign(
            make_not_null(&get<Tags::NormalDotFluxForJump<PrimalMortarVars>>(
                boundary_data.field_data))));

        // Store the exterior boundary data on the mortar
        all_mortar_data->at(mortar_id).remote_insert(temporal_id,
                                                     std::move(boundary_data));
      }  // if (is_internal)
    }    // loop directions
  }

  // --- This is essentially a break to communicate the mortar data ---

  template <bool DataIsZero, typename... OperatorTags, typename... PrimalVars,
            typename... AuxiliaryVars, typename... PrimalMortarVars,
            typename... AuxiliaryMortarVars, typename TemporalId,
            typename... FluxesArgs, typename... SourcesArgs,
            typename DirectionsPredicate = AllDirections>
  static void apply_operator(
      const gsl::not_null<Variables<tmpl::list<OperatorTags...>>*>
          operator_applied_to_vars,
      const gsl::not_null<Variables<tmpl::list<AuxiliaryVars...>>*>
          auxiliary_vars,
      const gsl::not_null<
          ::dg::MortarMap<Dim, MortarData<tmpl::list<PrimalMortarVars...>,
                                          tmpl::list<AuxiliaryMortarVars...>>>*>
          all_mortar_data,
      const Variables<tmpl::list<PrimalVars...>>& primal_vars,
      const Mesh<Dim>& mesh,
      const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
          inv_jacobian,
      const std::unordered_map<Direction<Dim>, Scalar<DataVector>>&
          internal_face_normal_magnitudes,
      const std::unordered_map<Direction<Dim>, Scalar<DataVector>>&
          external_face_normal_magnitudes,
      const ::dg::MortarMap<Dim, Mesh<Dim - 1>>& all_mortar_meshes,
      const ::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>& all_mortar_sizes,
      const double penalty_parameter, const TemporalId& temporal_id,
      const std::tuple<FluxesArgs...>& fluxes_args,
      const std::tuple<SourcesArgs...>& sources_args,
      const DirectionsPredicate& directions_predicate =
          AllDirections{}) noexcept {
    static_assert(
        sizeof...(PrimalVars) == sizeof...(PrimalFields) and
            sizeof...(AuxiliaryVars) == sizeof...(AuxiliaryFields) and
            sizeof...(PrimalMortarVars) == sizeof...(PrimalFields) and
            sizeof...(AuxiliaryMortarVars) == sizeof...(AuxiliaryFields) and
            sizeof...(OperatorTags) == sizeof...(PrimalFields),
        "The number of variables must match the number of system fields.");
    static_assert(
        (std::is_same_v<typename PrimalVars::type,
                        typename PrimalFields::type> and
         ...) and
            (std::is_same_v<typename AuxiliaryVars::type,
                            typename AuxiliaryFields::type> and
             ...) and
            (std::is_same_v<typename PrimalMortarVars::type,
                            typename PrimalFields::type> and
             ...) and
            (std::is_same_v<typename AuxiliaryMortarVars::type,
                            typename AuxiliaryFields::type> and
             ...) and
            (std::is_same_v<typename OperatorTags::type,
                            typename PrimalFields::type> and
             ...),
        "The variables must have the same tensor types as the system fields.");
#ifdef SPECTRE_DEBUG
    for (size_t d = 0; d < Dim; ++d) {
      ASSERT(mesh.basis(d) == Spectral::Basis::Legendre and
                 mesh.quadrature(d) == Spectral::Quadrature::GaussLobatto,
             "The elliptic DG operator is currently only implemented for "
             "Legendre-Gauss-Lobatto grids, but found basis '"
                 << mesh.basis(d) << "' and quadrature '" << mesh.quadrature(d)
                 << "' in dimension " << d << ".");
    }
#endif  // SPECTRE_DEBUG
    const size_t num_points = mesh.number_of_grid_points();

    // Add boundary corrections to the auxiliary variables _before_ computing
    // primal fluxes. This is called the "flux" formulation. It is equivalent to
    // discretizing the system in first-order form, i.e. treating the primal and
    // auxiliary variables on the same footing, and then taking a Schur
    // complement of the operator. The Schur complement is possible because the
    // auxiliary equations are essentially the definition of the auxiliary
    // variables and can therefore always be solved for them by just inverting
    // the mass matrix. This Schur-complement formulation avoids inflating the
    // DG operator with the DOFs of the auxiliary variables. In this form it is
    // very similar to the "primal" formulation where we get rid of the
    // auxiliary variables through a DG theorem and thus add the auxiliary
    // boundary corrections _after_ computing the primal fluxes. This involves a
    // slightly different lifting operation with differentiation matrices, which
    // we avoid to implement for now by using the flux-formulation.
    for (const auto& [mortar_id, mortar_data] : *all_mortar_data) {
      const auto& [direction, neighbor_id] = mortar_id;
      const bool is_internal =
          (neighbor_id != ElementId<Dim>::external_boundary_id());
      if constexpr (DataIsZero) {
        if (is_internal) {
          continue;
        }
      }
      if (not directions_predicate(direction)) {
        continue;
      }
      const auto face_mesh = mesh.slice_away(direction.dimension());
      const size_t slice_index = index_to_slice_at(mesh.extents(), direction);
      const auto& local_data = mortar_data.local_data(temporal_id);
      const auto& remote_data = mortar_data.remote_data(temporal_id);
      const auto& face_normal_magnitude =
          is_internal ? internal_face_normal_magnitudes.at(direction)
                      : external_face_normal_magnitudes.at(direction);
      const auto& mortar_mesh = all_mortar_meshes.at(mortar_id);
      const auto& mortar_size = all_mortar_sizes.at(mortar_id);
      //   Parallel::printf("mortar:%s\n", mortar_id);

      // This is the _strong_ auxiliary boundary correction avg(n.F_v) - n.F_v
      auto auxiliary_boundary_corrections_on_mortar =
          local_data.field_data.template extract_subset<
              tmpl::list<::Tags::NormalDotFlux<AuxiliaryMortarVars>...>>();
      //   Parallel::printf("aux correction local:\n%s\n",
      //                    auxiliary_boundary_corrections_on_mortar);
      //   Parallel::printf(
      //       "aux correction remote:\n%s\n",
      //       remote_data.field_data.template extract_subset<
      //           tmpl::list<::Tags::NormalDotFlux<AuxiliaryVars>...>>());
      auxiliary_boundary_corrections_on_mortar +=
          remote_data.field_data.template extract_subset<
              tmpl::list<::Tags::NormalDotFlux<AuxiliaryMortarVars>...>>();
      auxiliary_boundary_corrections_on_mortar *= -0.5;
      //   Parallel::printf("aux correction strong:\n%s\n",
      //                    auxiliary_boundary_corrections_on_mortar);

      // Project from the mortar back down to the face if needed
      auto auxiliary_boundary_corrections =
          ::dg::needs_projection(face_mesh, mortar_mesh, mortar_size)
              ? ::dg::project_from_mortar(
                    auxiliary_boundary_corrections_on_mortar, face_mesh,
                    mortar_mesh, mortar_size)
              : std::move(auxiliary_boundary_corrections_on_mortar);

      // Lift the boundary correction to the volume, but still only provide the
      // data only on the face because it is zero everywhere else. This is the
      // "massless" lifting operation, i.e. it involves an inverse mass matrix.
      // The mass matrix is diagonally approximated ("mass lumping") so it
      // reduces to a division by quadrature weights.
      ::dg::lift_flux(make_not_null(&auxiliary_boundary_corrections),
                      mesh.extents(direction.dimension()),
                      face_normal_magnitude);
      // The `dg::lift_flux` function contains an extra minus sign
      auxiliary_boundary_corrections *= -1.;
      //   Parallel::printf("lifted aux correction:\n%s\n\n",
      //                    auxiliary_boundary_corrections);

      // Add the boundary corrections to the auxiliary variables
      add_slice_to_data(auxiliary_vars, auxiliary_boundary_corrections,
                        mesh.extents(), direction.dimension(), slice_index);
    }  // apply auxiliary boundary corrections on all mortars

    // Compute the primal equation, i.e. the actual DG operator
    Variables<tmpl::list<PrimalFluxes...>> primal_fluxes{num_points};
    std::apply(
        [&primal_fluxes,
         &auxiliary_vars](const auto&... expanded_fluxes_args) noexcept {
          FluxesComputer::apply(
              make_not_null(&get<PrimalFluxes>(primal_fluxes))...,
              expanded_fluxes_args..., get<AuxiliaryVars>(*auxiliary_vars)...);
        },
        fluxes_args);
    divergence(operator_applied_to_vars, primal_fluxes, mesh, inv_jacobian);
    *operator_applied_to_vars *= -1.;
    std::apply(
        [&operator_applied_to_vars, &primal_vars,
         &primal_fluxes](const auto&... expanded_sources_args) noexcept {
          SourcesComputer::apply(
              make_not_null(&get<OperatorTags>(*operator_applied_to_vars))...,
              expanded_sources_args..., get<PrimalVars>(primal_vars)...,
              get<PrimalFluxes>(primal_fluxes)...);
        },
        sources_args);

    // Add boundary corrections to primal equation
    for (auto& [mortar_id, mortar_data] : *all_mortar_data) {
      const auto& [direction, neighbor_id] = mortar_id;
      const bool is_internal =
          (neighbor_id != ElementId<Dim>::external_boundary_id());
      if constexpr (DataIsZero) {
        if (is_internal) {
          continue;
        }
      }
      const auto face_mesh = mesh.slice_away(direction.dimension());
      const size_t slice_index = index_to_slice_at(mesh.extents(), direction);
      const auto [local_data, remote_data] = mortar_data.extract();
      const auto& face_normal_magnitude =
          is_internal ? internal_face_normal_magnitudes.at(direction)
                      : external_face_normal_magnitudes.at(direction);
      const auto& mortar_mesh = all_mortar_meshes.at(mortar_id);
      const auto& mortar_size = all_mortar_sizes.at(mortar_id);

      // This is the _strong_ primal boundary correction avg(n.F_u) - penalty *
      // jump(n.F_v) - n.F_u. Note that the "internal penalty" numerical flux
      // (as opposed to the LLF flux) uses the raw field derivatives without
      // boundary corrections in the average, which is why we can communicate
      // the data so early together with the auxiliary boundary data. In this
      // case the penalty needs to include a factor p^2 / h (see the `penalty`
      // function).
      const auto penalty = elliptic::dg::penalty(
          min(get(get<Tags::ElementSize>(local_data.field_data)),
              get(get<Tags::ElementSize>(remote_data.field_data))),
          std::max(get<Tags::PerpendicularNumPoints>(local_data.extra_data),
                   get<Tags::PerpendicularNumPoints>(remote_data.extra_data)),
          penalty_parameter);
      // Start with the penalty term
      auto primal_boundary_corrections_on_mortar =
          local_data.field_data.template extract_subset<
              tmpl::list<Tags::NormalDotFluxForJump<PrimalMortarVars>...>>();
      primal_boundary_corrections_on_mortar -=
          remote_data.field_data.template extract_subset<
              tmpl::list<Tags::NormalDotFluxForJump<PrimalMortarVars>...>>();
      primal_boundary_corrections_on_mortar *= penalty;
      primal_boundary_corrections_on_mortar +=
          0.5 * local_data.field_data.template extract_subset<
                    tmpl::list<::Tags::NormalDotFlux<PrimalMortarVars>...>>();
      primal_boundary_corrections_on_mortar +=
          0.5 * remote_data.field_data.template extract_subset<
                    tmpl::list<::Tags::NormalDotFlux<PrimalMortarVars>...>>();
      primal_boundary_corrections_on_mortar *= -1.;

      // Project from the mortar back down to the face if needed, lift and add
      // to operator. See auxiliary boundary corrections above for details.
      auto primal_boundary_corrections =
          ::dg::needs_projection(face_mesh, mortar_mesh, mortar_size)
              ? ::dg::project_from_mortar(primal_boundary_corrections_on_mortar,
                                          face_mesh, mortar_mesh, mortar_size)
              : std::move(primal_boundary_corrections_on_mortar);
      ::dg::lift_flux(make_not_null(&primal_boundary_corrections),
                      mesh.extents(direction.dimension()),
                      face_normal_magnitude);
      // Add the boundary corrections to the auxiliary variables
      add_slice_to_data(operator_applied_to_vars, primal_boundary_corrections,
                        mesh.extents(), direction.dimension(), slice_index);
    }  // apply primal boundary corrections on all mortars

    // Finally, invert sign of the operator because. This is the sign that makes
    // the operator _minus_ the Laplacian for a Poisson system.
  }

  template <typename... FixedSourcesTags, typename ApplyBoundaryCondition,
            typename... FluxesArgs, typename... SourcesArgs,
            bool LocalLinearized = Linearized,
            // This function adds nothing to the fixed sources if the operator
            // is linearized, so it shouldn't be used.
            Requires<not LocalLinearized> = nullptr>
  static void impose_inhomogeneous_boundary_conditions_on_source(
      const gsl::not_null<Variables<tmpl::list<FixedSourcesTags...>>*>
          fixed_sources,
      const Element<Dim>& element, const Mesh<Dim>& mesh,
      const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
          inv_jacobian,
      const std::unordered_map<Direction<Dim>, tnsr::i<DataVector, Dim>>&
          external_face_normals,
      const std::unordered_map<Direction<Dim>, Scalar<DataVector>>&
          external_face_normal_magnitudes,
      const ::dg::MortarMap<Dim, Mesh<Dim - 1>>& all_mortar_meshes,
      const ::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>& all_mortar_sizes,
      const double penalty_parameter,
      const ApplyBoundaryCondition& apply_boundary_condition,
      const std::tuple<FluxesArgs...>& fluxes_args,
      const std::tuple<SourcesArgs...>& sources_args,
      const DirectionMap<Dim, std::tuple<FluxesArgs...>>&
          fluxes_args_on_internal_faces,
      const DirectionMap<Dim, std::tuple<FluxesArgs...>>&
          fluxes_args_on_external_faces) noexcept {
    // We just feed zero variables through the nonlinear operator to extract the
    // constant contribution at external boundaries. Since the variables are
    // zero the operator simplifies quite a lot. The simplification is probably
    // not very important for performance because this function will only be
    // called when solving a linear elliptic system and only once during
    // initialization, but we specialize the operator for zero data nonetheless
    // so we can ignore internal boundaries. We would have to copy mortar data
    // around to emulate the communication step, so by just skipping internal
    // boundaries we avoid that.
    const size_t num_points = mesh.number_of_grid_points();
    const Variables<tmpl::list<PrimalFields...>> zero_primal_vars{num_points,
                                                                  0.};
    Variables<tmpl::list<AuxiliaryFields...>> auxiliary_vars{num_points};
    Variables<tmpl::list<FixedSourcesTags...>> operator_applied_to_zero_vars{
        num_points};
    // Set up data on mortars. We only need them at external boundaries.
    ::dg::MortarMap<Dim, MortarData<tmpl::list<PrimalFields...>,
                                    tmpl::list<AuxiliaryFields...>>>
        all_mortar_data{};
    constexpr size_t temporal_id = std::numeric_limits<size_t>::max();
    // Apply the operator to the zero variables, skipping internal boundaries
    prepare_mortar_data<true>(
        make_not_null(&auxiliary_vars), make_not_null(&all_mortar_data),
        zero_primal_vars, element, mesh, inv_jacobian, {},
        external_face_normals, {}, external_face_normal_magnitudes,
        all_mortar_meshes, all_mortar_sizes, temporal_id,
        apply_boundary_condition, fluxes_args, sources_args,
        fluxes_args_on_internal_faces, fluxes_args_on_external_faces);
    apply_operator<true>(
        make_not_null(&operator_applied_to_zero_vars),
        make_not_null(&auxiliary_vars), make_not_null(&all_mortar_data),
        zero_primal_vars, mesh, inv_jacobian, {},
        external_face_normal_magnitudes, all_mortar_meshes, all_mortar_sizes,
        penalty_parameter, temporal_id, fluxes_args, sources_args);
    // Impose the nonlinear (constant) boundary contribution as fixed sources on
    // the RHS of the equations
    *fixed_sources -= operator_applied_to_zero_vars;
  }
};

}  // namespace detail

template <typename System, bool Linearized, typename... Args>
void prepare_mortar_data(Args&&... args) noexcept {
  detail::DgOperatorImpl<System, Linearized>::template prepare_mortar_data<
      false>(std::forward<Args>(args)...);
}

template <typename System, bool Linearized, typename... Args>
void apply_operator(Args&&... args) noexcept {
  detail::DgOperatorImpl<System, Linearized>::template apply_operator<false>(
      std::forward<Args>(args)...);
}

template <typename System, typename... Args>
void impose_inhomogeneous_boundary_conditions_on_source(
    Args&&... args) noexcept {
  detail::DgOperatorImpl<System, false>::
      impose_inhomogeneous_boundary_conditions_on_source(
          std::forward<Args>(args)...);
}

}  // namespace elliptic::dg
