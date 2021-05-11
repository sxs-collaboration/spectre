// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/range/join.hpp>
#include <cstddef>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Elliptic/DiscontinuousGalerkin/Penalty.hpp"
#include "Elliptic/Systems/GetSourcesComputer.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleMortarData.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/*!
 * \brief Functionality related to discontinuous Galerkin discretizations of
 * elliptic equations
 *
 * The following is a brief overview of the elliptic DG schemes that are
 * implemented here. A publication that describes the schemes in detail is in
 * preparation and will be referenced here.
 *
 * The DG schemes apply to any elliptic PDE that can be formulated in
 * first-order flux-form
 *
 * \f{equation}
 * -\partial_i F^i_\alpha + S_\alpha = f_\alpha(x)
 * \f}
 *
 * in terms of fluxes \f$F^i_\alpha\f$, sources \f$S_\alpha\f$ and fixed-sources
 * \f$f_\alpha(x)\f$. The fluxes and sources are functionals of the "primal"
 * system variables \f$u_A(x)\f$ and their corresponding "auxiliary" variables
 * \f$v_A(x)\f$. The uppercase index enumerates the variables such that
 * \f$v_A\f$ is the auxiliary variable corresponding to \f$u_A\f$. Greek indices
 * enumerate _all_ variables. In documentation related to particular elliptic
 * systems we generally use the canonical system-specific symbols for the fields
 * in place of these indices. See the `Poisson::FirstOrderSystem` and the
 * `Elasticity::FirstOrderSystem` for examples.
 *
 * The DG discretization of equations in this first-order form amounts to
 * projecting the equations on the set of basis functions that we also use to
 * represent the fields on the computational grid. The currently implemented DG
 * operator uses Lagrange interpolating polynomials w.r.t.
 * Legendre-Gauss-Lobatto collocation points as basis functions. Support for
 * Legendre-Gauss collocation points can be added if needed. Skipping all
 * further details here, the discretization results in a linear equation
 * \f$A(u)=b\f$ over all grid points and primal variables. Solving the elliptic
 * equations amounts to numerically inverting the DG operator \f$A\f$, typically
 * without ever constructing the full matrix but by employing an iterative
 * linear solver that repeatedly applies the DG operator to "test data". Note
 * that the DG operator applies directly to the primal variables. Auxiliary
 * variables are only computed temporarily and don't inflate the size of the
 * operator. This means the DG operator essentially computes second derivatives
 * of the primal variables, modified by the fluxes and sources of the system
 * as well as by DG boundary corrections that couple grid points across element
 * boundaries.
 *
 * \par Boundary corrections:
 * In this implementation we employ the "internal penalty" DG scheme that
 * couples grid points across nearest-neighbor elements through the fluxes:
 *
 * \f{align}
 * (n_i F^i_v)^* &= \frac{1}{2} n_i \left(
 * F^i_v(u^\mathrm{int}) +
 * F^i_v(u^\mathrm{ext})\right) \\
 * (n_i F^i_u)^* &= \frac{1}{2} n_i \left(
 * F^i_u(\partial_j F^j_v(u^\mathrm{int}) - S_v(u^\mathrm{int})) +
 * F^i_u(\partial_j F^j_v(u^\mathrm{ext}) - S_v(u^\mathrm{ext}))\right)
 * - \sigma n_i \left(
 * F^i_u(n_j F^j_v(u^\mathrm{int})) -
 * F^i_u(n_j F^j_v(u^\mathrm{ext}))\right)
 * \f}
 *
 * Note that \f$n_i\f$ denotes the face normal on the "interior" side of the
 * element under consideration. We assume \f$n^\mathrm{ext}_i=-n_i\f$ in the
 * implementation, i.e. face normals don't depend on the dynamic variables
 * (which may be discontinuous on element faces). This is the case for the
 * problems we are expecting to solve, because those will be on fixed background
 * metrics (e.g. a conformal metric for the XCTS system). Numerically, the face
 * normals on either side of a mortar may nonetheless be different because the
 * two faces adjacent to the mortar may resolve them at different resolutions.
 *
 * Also note that the numerical fluxes intentionally don't depend on the
 * auxiliary field values \f$v\f$. This property allows us to communicate data
 * for both the primal and auxiliary boundary corrections together, instead of
 * communicating them in two steps. If we were to resort to a two-step
 * communication we could replace the divergence in \f$(n_i F^i_u)^*\f$ with
 * \f$v\f$, which would result in a generalized "stabilized central flux" that
 * is slightly less sparse than the internal penalty flux (see e.g.
 * \cite HesthavenWarburton, section 7.2). We could also choose to ignore the
 * fluxes in the penalty term, but preliminary tests suggest that this may hurt
 * convergence.
 *
 * For a Poisson system (see `Poisson::FirstOrderSystem`) this numerical flux
 * reduces to the standard internal penalty flux (see e.g.
 * \cite HesthavenWarburton, section 7.2, or \cite Arnold2002):
 *
 * \f{align}
 * (n_i F^i_{v_j})^* &= n_j u^* = \frac{1}{2} n_j \left(
 * u^\mathrm{int} + u^\mathrm{ext}\right) \\
 * (n_i F^i_u)^* &= n_i v_i^* = \frac{1}{2} n_i \left(
 * \partial_i u^\mathrm{int} + \partial_i u^\mathrm{ext}\right)
 * - \sigma \left(u^\mathrm{int} - u^\mathrm{ext}\right)
 * \f}
 *
 * where a sum over repeated indices is assumed, since the equation is
 * formulated on a Euclidean geometry.
 *
 * The penalty factor \f$\sigma\f$ is responsible for removing zero eigenmodes
 * and impacts the conditioning of the linear operator to be solved. See
 * `elliptic::dg::penalty` for details. For the element size that goes into
 * computing the penalty we choose
 * \f$h=\frac{J_\mathrm{volume}}{J_\mathrm{face}}\f$, i.e. the ratio of Jacobi
 * determinants from logical to inertial coordinates in the element volume and
 * on the element face, both evaluated on the face (see \cite Vincent2019qpd).
 * Since both \f$N_\mathrm{points}\f$ and \f$h\f$ can be different on either
 * side of the element boundary we take the maximum of \f$N_\mathrm{points}\f$
 * and the pointwise minimum of \f$h\f$ across the element boundary as is done
 * in \cite Vincent2019qpd. Note that we use the number of points
 * \f$N_\mathrm{points}\f$ where \cite Vincent2019qpd uses the polynomial degree
 * \f$N_\mathrm{points} - 1\f$ because we found unstable configurations on
 * curved meshes when using the polynomial degree. Optimizing the penalty on
 * curved meshes is subject to further investigation.
 */
namespace elliptic::dg {

namespace Tags {
/// Number of grid points perpendicular to an element face. Used to compute
/// the penalty (see `elliptic::dg::penalty`).
struct PerpendicularNumPoints {
  using type = size_t;
};

/// A measure of element size perpendicular to an element face. Used to compute
/// the penalty (see `elliptic::dg::penalty`).
struct ElementSize {
  using type = Scalar<DataVector>;
};

/// The quantity \f$n_i F^i_u(n_j F^j_v(u))\f$ where \f$F^i\f$ is the system
/// flux for primal variables \f$u\f$ and auxiliary variables \f$v\f$, and
/// \f$n_i\f$ is the face normal. This quantity is projected to mortars to
/// compute the jump term of the numerical flux.
template <typename Tag>
struct NormalDotFluxForJump : db::PrefixTag {
  using type = typename Tag::type;
  using tag = Tag;
};
}  // namespace Tags

/// Data that is projected to mortars and communicated across element
/// boundaries
template <typename PrimalFields, typename PrimalFluxes>
using BoundaryData = ::dg::SimpleBoundaryData<
    tmpl::append<db::wrap_tags_in<::Tags::NormalDotFlux, PrimalFields>,
                 db::wrap_tags_in<::Tags::NormalDotFlux, PrimalFluxes>,
                 db::wrap_tags_in<Tags::NormalDotFluxForJump, PrimalFields>,
                 tmpl::list<Tags::ElementSize>>,
    tmpl::list<Tags::PerpendicularNumPoints>>;

/// Construct `elliptic::dg::BoundaryData` assuming the variable data on the
/// element is zero, and project it to the mortar.
template <typename PrimalMortarFields, typename PrimalMortarFluxes, size_t Dim>
BoundaryData<PrimalMortarFields, PrimalMortarFluxes>
zero_boundary_data_on_mortar(
    const Direction<Dim>& direction, const Mesh<Dim>& mesh,
    const Scalar<DataVector>& face_normal_magnitude,
    const Mesh<Dim - 1>& mortar_mesh,
    const ::dg::MortarSize<Dim - 1>& mortar_size) noexcept {
  const auto face_mesh = mesh.slice_away(direction.dimension());
  const size_t face_num_points = face_mesh.number_of_grid_points();
  BoundaryData<PrimalMortarFields, PrimalMortarFluxes> boundary_data{
      face_num_points};
  boundary_data.field_data.initialize(face_num_points, 0.);
  get<Tags::PerpendicularNumPoints>(boundary_data.extra_data) =
      mesh.extents(direction.dimension());
  get(get<Tags::ElementSize>(boundary_data.field_data)) =
      2. / get(face_normal_magnitude);
  return Spectral::needs_projection(face_mesh, mortar_mesh, mortar_size)
             ? boundary_data.project_to_mortar(face_mesh, mortar_mesh,
                                               mortar_size)
             : std::move(boundary_data);
}

/// Boundary data on both sides of a mortar.
///
/// \note This is a struct (as opposed to a type alias) so it can be used to
/// deduce the template parameters
template <typename TemporalId, typename PrimalFields, typename PrimalFluxes>
struct MortarData
    : ::dg::SimpleMortarData<TemporalId,
                             BoundaryData<PrimalFields, PrimalFluxes>,
                             BoundaryData<PrimalFields, PrimalFluxes>> {};

namespace Tags {
/// Holds `elliptic::dg::MortarData`, i.e. boundary data on both sides of a
/// mortar
template <typename TemporalId, typename PrimalFields, typename PrimalFluxes>
struct MortarData : db::SimpleTag {
  using type = elliptic::dg::MortarData<TemporalId, PrimalFields, PrimalFluxes>;
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
  using SourcesComputer = elliptic::get_sources_computer<System, Linearized>;

  struct AllDirections {
    bool operator()(const Direction<Dim>& /*unused*/) const noexcept {
      return true;
    }
  };

  static constexpr auto full_mortar_size =
      make_array<Dim - 1>(Spectral::MortarSize::Full);

  template <bool DataIsZero, typename... PrimalVars, typename... AuxiliaryVars,
            typename... PrimalFluxesVars, typename... AuxiliaryFluxesVars,
            typename... PrimalMortarVars, typename... PrimalMortarFluxes,
            typename TemporalId, typename ApplyBoundaryCondition,
            typename... FluxesArgs, typename... SourcesArgs,
            typename DirectionsPredicate = AllDirections>
  static void prepare_mortar_data(
      const gsl::not_null<Variables<tmpl::list<AuxiliaryVars...>>*>
          auxiliary_vars,
      const gsl::not_null<Variables<tmpl::list<AuxiliaryFluxesVars...>>*>
          auxiliary_fluxes,
      const gsl::not_null<Variables<tmpl::list<PrimalFluxesVars...>>*>
          primal_fluxes,
      const gsl::not_null<::dg::MortarMap<
          Dim, MortarData<TemporalId, tmpl::list<PrimalMortarVars...>,
                          tmpl::list<PrimalMortarFluxes...>>>*>
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
            sizeof...(AuxiliaryVars) == sizeof...(AuxiliaryFields) and
            sizeof...(PrimalFluxesVars) == sizeof...(PrimalFluxes) and
            sizeof...(AuxiliaryFluxesVars) == sizeof...(AuxiliaryFluxes),
        "The number of variables must match the number of system fields.");
    static_assert(
        (std::is_same_v<typename PrimalVars::type,
                        typename PrimalFields::type> and
         ...) and
            (std::is_same_v<typename AuxiliaryVars::type,
                            typename AuxiliaryFields::type> and
             ...) and
            (std::is_same_v<typename PrimalFluxesVars::type,
                            typename PrimalFluxes::type> and
             ...) and
            (std::is_same_v<typename AuxiliaryFluxesVars::type,
                            typename AuxiliaryFluxes::type> and
             ...),
        "The variables must have the same tensor types as the system fields.");
#ifdef SPECTRE_DEBUG
    for (size_t d = 0; d < Dim; ++d) {
      ASSERT(mesh.basis(d) == Spectral::Basis::Legendre and
                 mesh.quadrature(d) == Spectral::Quadrature::GaussLobatto,
             "The elliptic DG operator is currently only implemented for "
             "Legendre-Gauss-Lobatto grids. Found basis '"
                 << mesh.basis(d) << "' and quadrature '" << mesh.quadrature(d)
                 << "' in dimension " << d << ".");
    }
#endif  // SPECTRE_DEBUG
    const size_t num_points = mesh.number_of_grid_points();

    // This function and the one below allocate various Variables to compute
    // intermediate quantities. It could be a performance optimization to reduce
    // the number of these allocations and/or move some of the memory buffers
    // into the DataBox to keep them around permanently. The latter should be
    // informed by profiling.

    // Compute the auxiliary variables, and from those the primal fluxes. The
    // auxiliary variables are the variables `v` in the auxiliary equations
    // `-div(F_v(u)) + S_v + v = 0` where `F_v` and `S_v` are the auxiliary
    // fluxes and sources. From the auxiliary variables we compute the primal
    // fluxes `F_u(v)` for the primal equation `-div(F_u(v)) + S_u = f(x)`. Note
    // that before taking the second derivative, boundary corrections from the
    // first derivative have to be added to `F_u(v)`. Therefore the second
    // derivative is taken after a communication break in the `apply_operator`
    // function below.
    if constexpr (DataIsZero) {
      (void)auxiliary_vars;
      (void)auxiliary_fluxes;
      primal_fluxes->initialize(num_points, 0.);
    } else {
      // Compute the auxiliary fluxes
      auxiliary_fluxes->initialize(num_points);
      std::apply(
          [&auxiliary_fluxes,
           &primal_vars](const auto&... expanded_fluxes_args) noexcept {
            FluxesComputer::apply(
                make_not_null(&get<AuxiliaryFluxesVars>(*auxiliary_fluxes))...,
                expanded_fluxes_args..., get<PrimalVars>(primal_vars)...);
          },
          fluxes_args);
      divergence(auxiliary_vars, *auxiliary_fluxes, mesh, inv_jacobian);
      // Subtract the sources. Invert signs because the sources computers _add_
      // the sources, i.e. they compute the `+ S_v` term defined above.
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
      // Compute the primal fluxes
      primal_fluxes->initialize(num_points);
      std::apply(
          [&primal_fluxes,
           &auxiliary_vars](const auto&... expanded_fluxes_args) noexcept {
            FluxesComputer::apply(
                make_not_null(&get<PrimalFluxesVars>(*primal_fluxes))...,
                expanded_fluxes_args...,
                get<AuxiliaryVars>(*auxiliary_vars)...);
          },
          fluxes_args);
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
      Variables<tmpl::list<PrimalFluxesVars..., AuxiliaryFluxesVars...>>
          fluxes_on_face{face_num_points};
      Variables<tmpl::list<::Tags::NormalDotFlux<AuxiliaryFields>...>>
          n_dot_aux_fluxes{face_num_points};
      BoundaryData<tmpl::list<PrimalMortarVars...>,
                   tmpl::list<PrimalMortarFluxes...>>
          boundary_data{face_num_points};
      if constexpr (DataIsZero) {
        // Just setting all boundary field data to zero. Variable-independent
        // data such as the element size will be set below.
        boundary_data.field_data.initialize(face_num_points, 0.);
      } else {
        // Compute F_u(n.F_v)
        fluxes_on_face.assign_subset(
            data_on_slice(*auxiliary_fluxes, mesh.extents(),
                          direction.dimension(), slice_index));
        EXPAND_PACK_LEFT_TO_RIGHT(normal_dot_flux(
            make_not_null(
                &get<::Tags::NormalDotFlux<AuxiliaryFields>>(n_dot_aux_fluxes)),
            face_normal, get<AuxiliaryFluxesVars>(fluxes_on_face)));
        std::apply(
            [&boundary_data, &n_dot_aux_fluxes](
                const auto&... expanded_fluxes_args_on_face) noexcept {
              FluxesComputer::apply(
                  make_not_null(&get<::Tags::NormalDotFlux<PrimalMortarFluxes>>(
                      boundary_data.field_data))...,
                  expanded_fluxes_args_on_face...,
                  get<::Tags::NormalDotFlux<AuxiliaryFields>>(
                      n_dot_aux_fluxes)...);
            },
            fluxes_args_on_face);
        // Compute n.F_u
        //
        // For the internal penalty flux we can already slice the n.F_u to the
        // faces at this point, before the boundary corrections have been added
        // to the auxiliary variables. The reason is essentially that the
        // internal penalty flux uses average(grad(u)) in place of average(v),
        // i.e. the raw primal field derivatives without boundary corrections.
        fluxes_on_face.assign_subset(
            data_on_slice(*primal_fluxes, mesh.extents(), direction.dimension(),
                          slice_index));
        EXPAND_PACK_LEFT_TO_RIGHT(normal_dot_flux(
            make_not_null(&get<::Tags::NormalDotFlux<PrimalMortarVars>>(
                boundary_data.field_data)),
            face_normal, get<PrimalFluxesVars>(fluxes_on_face)));
        // Compute n.F_u(n.F_v) for jump term, re-using the memory buffer from
        // above
        EXPAND_PACK_LEFT_TO_RIGHT(normal_dot_flux(
            make_not_null(&get<Tags::NormalDotFluxForJump<PrimalMortarVars>>(
                boundary_data.field_data)),
            face_normal,
            get<::Tags::NormalDotFlux<PrimalMortarFluxes>>(
                boundary_data.field_data)));
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
                Spectral::needs_projection(face_mesh, mortar_mesh, mortar_size)
                    ? boundary_data.project_to_mortar(face_mesh, mortar_mesh,
                                                      mortar_size)
                    : std::move(boundary_data);
            (*all_mortar_data)[mortar_id].local_insert(
                temporal_id, std::move(projected_boundary_data));
          }
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
        // The `apply_boundary_conditions` invocable is expected to impose
        // boundary conditions by modifying the fields and fluxes that are
        // passed by reference. Dirichlet-type boundary conditions are imposed
        // by modifying the fields, and Neumann-type boundary conditions are
        // imposed by modifying the interior n.F_u. Note that all data passed to
        // the boundary conditions is taken from the "interior" side of the
        // boundary, i.e. with a normal vector that points _out_ of the
        // computational domain.
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
                  make_not_null(&get<AuxiliaryFluxesVars>(fluxes_on_face))...,
                  expanded_fluxes_args_on_face...,
                  get<PrimalVars>(dirichlet_vars)...);
            },
            fluxes_args_on_face);
        EXPAND_PACK_LEFT_TO_RIGHT(normal_dot_flux(
            make_not_null(
                &get<::Tags::NormalDotFlux<AuxiliaryFields>>(n_dot_aux_fluxes)),
            face_normal, get<AuxiliaryFluxesVars>(fluxes_on_face)));
        std::apply(
            [&boundary_data, &n_dot_aux_fluxes](
                const auto&... expanded_fluxes_args_on_face) noexcept {
              FluxesComputer::apply(
                  make_not_null(&get<::Tags::NormalDotFlux<PrimalMortarFluxes>>(
                      boundary_data.field_data))...,
                  expanded_fluxes_args_on_face...,
                  get<::Tags::NormalDotFlux<AuxiliaryFields>>(
                      n_dot_aux_fluxes)...);
            },
            fluxes_args_on_face);

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
            make_not_null(&get<::Tags::NormalDotFlux<PrimalMortarFluxes>>(
                boundary_data.field_data)),
            get<::Tags::NormalDotFlux<PrimalMortarFluxes>>(
                all_mortar_data->at(mortar_id)
                    .local_data(temporal_id)
                    .field_data)));

        // Compute n.F_u(n.F_v) for jump term
        EXPAND_PACK_LEFT_TO_RIGHT(normal_dot_flux(
            make_not_null(&get<Tags::NormalDotFluxForJump<PrimalMortarVars>>(
                boundary_data.field_data)),
            face_normal,
            get<::Tags::NormalDotFlux<PrimalMortarFluxes>>(
                boundary_data.field_data)));
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
            typename... PrimalFluxesVars, typename... PrimalMortarVars,
            typename... PrimalMortarFluxes, typename TemporalId,
            typename... FluxesArgs, typename... SourcesArgs,
            typename DirectionsPredicate = AllDirections>
  static void apply_operator(
      const gsl::not_null<Variables<tmpl::list<OperatorTags...>>*>
          operator_applied_to_vars,
      const gsl::not_null<::dg::MortarMap<
          Dim, MortarData<TemporalId, tmpl::list<PrimalMortarVars...>,
                          tmpl::list<PrimalMortarFluxes...>>>*>
          all_mortar_data,
      const Variables<tmpl::list<PrimalVars...>>& primal_vars,
      // Taking the primal fluxes computed in the `prepare_mortar_data` function
      // by const-ref here because other code might use them and so we don't
      // want to modify them by adding boundary corrections. E.g. linearized
      // sources use the nonlinear fields and fluxes as background fields.
      const Variables<tmpl::list<PrimalFluxesVars...>>& primal_fluxes,
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
      const std::tuple<SourcesArgs...>& sources_args,
      const DirectionsPredicate& directions_predicate =
          AllDirections{}) noexcept {
    static_assert(
        sizeof...(PrimalVars) == sizeof...(PrimalFields) and
            sizeof...(PrimalFluxesVars) == sizeof...(PrimalFluxes) and
            sizeof...(PrimalMortarVars) == sizeof...(PrimalFields) and
            sizeof...(PrimalMortarFluxes) == sizeof...(PrimalFluxes) and
            sizeof...(OperatorTags) == sizeof...(PrimalFields),
        "The number of variables must match the number of system fields.");
    static_assert(
        (std::is_same_v<typename PrimalVars::type,
                        typename PrimalFields::type> and
         ...) and
            (std::is_same_v<typename PrimalFluxesVars::type,
                            typename PrimalFluxes::type> and
             ...) and
            (std::is_same_v<typename PrimalMortarVars::type,
                            typename PrimalFields::type> and
             ...) and
            (std::is_same_v<typename PrimalMortarFluxes::type,
                            typename PrimalFluxes::type> and
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
             "Legendre-Gauss-Lobatto grids. Found basis '"
                 << mesh.basis(d) << "' and quadrature '" << mesh.quadrature(d)
                 << "' in dimension " << d << ".");
    }
#endif  // SPECTRE_DEBUG

    // This function and the one above allocate various Variables to compute
    // intermediate quantities. It could be a performance optimization to reduce
    // the number of these allocations and/or move some of the memory buffers
    // into the DataBox to keep them around permanently. The latter should be
    // informed by profiling.

    // Add boundary corrections to the auxiliary variables _before_ computing
    // the second derivative. This is called the "flux" formulation. It is
    // equivalent to discretizing the system in first-order form, i.e. treating
    // the primal and auxiliary variables on the same footing, and then taking a
    // Schur complement of the operator. The Schur complement is possible
    // because the auxiliary equations are essentially the definition of the
    // auxiliary variables and can therefore always be solved for the auxiliary
    // variables by just inverting the mass matrix. This Schur-complement
    // formulation avoids inflating the DG operator with the DOFs of the
    // auxiliary variables. In this form it is very similar to the "primal"
    // formulation where we get rid of the auxiliary variables through a DG
    // theorem and thus add the auxiliary boundary corrections _after_ computing
    // the second derivative. This involves a slightly different lifting
    // operation with differentiation matrices, which we avoid to implement for
    // now by using the flux-formulation.
    auto primal_fluxes_corrected = primal_fluxes;
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
      const auto& mortar_mesh =
          is_internal ? all_mortar_meshes.at(mortar_id) : face_mesh;
      const auto& mortar_size =
          is_internal ? all_mortar_sizes.at(mortar_id) : full_mortar_size;

      // This is the _strong_ auxiliary boundary correction avg(n.F_v) - n.F_v
      auto auxiliary_boundary_corrections_on_mortar =
          Variables<tmpl::list<PrimalMortarFluxes...>>(
              local_data.field_data.template extract_subset<
                  tmpl::list<::Tags::NormalDotFlux<PrimalMortarFluxes>...>>());
      const auto add_remote_contribution = [](auto& lhs,
                                              const auto& rhs) noexcept {
        for (size_t i = 0; i < lhs.size(); ++i) {
          lhs[i] += rhs[i];
        }
      };
      EXPAND_PACK_LEFT_TO_RIGHT(add_remote_contribution(
          get<PrimalMortarFluxes>(auxiliary_boundary_corrections_on_mortar),
          get<::Tags::NormalDotFlux<PrimalMortarFluxes>>(
              remote_data.field_data)));
      auxiliary_boundary_corrections_on_mortar *= -0.5;

      // Project from the mortar back down to the face if needed
      auto auxiliary_boundary_corrections =
          Spectral::needs_projection(face_mesh, mortar_mesh, mortar_size)
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

      // Add the boundary corrections to the auxiliary variables
      add_slice_to_data(make_not_null(&primal_fluxes_corrected),
                        auxiliary_boundary_corrections, mesh.extents(),
                        direction.dimension(), slice_index);
    }  // apply auxiliary boundary corrections on all mortars

    // Compute the primal equation, i.e. the actual DG operator, by taking the
    // second derivative: -div(F_u(v)) + S_u = f(x)
    divergence(operator_applied_to_vars, primal_fluxes_corrected, mesh,
               inv_jacobian);
    // This is the sign flip that makes the operator _minus_ the Laplacian for a
    // Poisson system
    *operator_applied_to_vars *= -1.;
    // Using the non-boundary-corrected primal fluxes to compute sources here
    std::apply(
        [&operator_applied_to_vars, &primal_vars,
         &primal_fluxes](const auto&... expanded_sources_args) noexcept {
          SourcesComputer::apply(
              make_not_null(&get<OperatorTags>(*operator_applied_to_vars))...,
              expanded_sources_args..., get<PrimalVars>(primal_vars)...,
              get<PrimalFluxesVars>(primal_fluxes)...);
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
      const auto& mortar_mesh =
          is_internal ? all_mortar_meshes.at(mortar_id) : face_mesh;
      const auto& mortar_size =
          is_internal ? all_mortar_sizes.at(mortar_id) : full_mortar_size;

      // This is the _strong_ primal boundary correction avg(n.F_u) - penalty *
      // jump(n.F_v) - n.F_u. Note that the "internal penalty" numerical flux
      // (as opposed to the LLF flux) uses the raw field derivatives without
      // boundary corrections in the average, which is why we can communicate
      // the data so early together with the auxiliary boundary data. In this
      // case the penalty needs to include a factor N_points^2 / h (see the
      // `penalty` function).
      const auto penalty = elliptic::dg::penalty(
          min(get(get<Tags::ElementSize>(local_data.field_data)),
              get(get<Tags::ElementSize>(remote_data.field_data))),
          std::max(get<Tags::PerpendicularNumPoints>(local_data.extra_data),
                   get<Tags::PerpendicularNumPoints>(remote_data.extra_data)),
          penalty_parameter);
      // Start with the penalty term
      auto primal_boundary_corrections_on_mortar =
          Variables<tmpl::list<PrimalMortarVars...>>(
              local_data.field_data.template extract_subset<tmpl::list<
                  Tags::NormalDotFluxForJump<PrimalMortarVars>...>>());
      const auto add_remote_jump_contribution = [](auto& lhs,
                                                   const auto& rhs) noexcept {
        for (size_t i = 0; i < lhs.size(); ++i) {
          lhs[i] -= rhs[i];
        }
      };
      EXPAND_PACK_LEFT_TO_RIGHT(add_remote_jump_contribution(
          get<PrimalMortarVars>(primal_boundary_corrections_on_mortar),
          get<Tags::NormalDotFluxForJump<PrimalMortarVars>>(
              remote_data.field_data)));
      primal_boundary_corrections_on_mortar *= penalty;
      const auto add_remote_avg_contribution = [](auto& lhs,
                                                  const auto& rhs) noexcept {
        for (size_t i = 0; i < lhs.size(); ++i) {
          lhs[i] += 0.5 * rhs[i];
        }
      };
      EXPAND_PACK_LEFT_TO_RIGHT(add_remote_avg_contribution(
          get<PrimalMortarVars>(primal_boundary_corrections_on_mortar),
          get<::Tags::NormalDotFlux<PrimalMortarVars>>(local_data.field_data)));
      EXPAND_PACK_LEFT_TO_RIGHT(add_remote_avg_contribution(
          get<PrimalMortarVars>(primal_boundary_corrections_on_mortar),
          get<::Tags::NormalDotFlux<PrimalMortarVars>>(
              remote_data.field_data)));
      primal_boundary_corrections_on_mortar *= -1.;

      // Project from the mortar back down to the face if needed, lift and add
      // to operator. See auxiliary boundary corrections above for details.
      auto primal_boundary_corrections =
          Spectral::needs_projection(face_mesh, mortar_mesh, mortar_size)
              ? ::dg::project_from_mortar(primal_boundary_corrections_on_mortar,
                                          face_mesh, mortar_mesh, mortar_size)
              : std::move(primal_boundary_corrections_on_mortar);
      ::dg::lift_flux(make_not_null(&primal_boundary_corrections),
                      mesh.extents(direction.dimension()),
                      face_normal_magnitude);
      add_slice_to_data(operator_applied_to_vars, primal_boundary_corrections,
                        mesh.extents(), direction.dimension(), slice_index);
    }  // loop over all mortars
  }

  template <typename... FixedSourcesTags, typename ApplyBoundaryCondition,
            typename... FluxesArgs, typename... SourcesArgs,
            bool LocalLinearized = Linearized,
            // This function adds nothing to the fixed sources if the operator
            // is linearized, so it shouldn't be used in that case
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
    // just so we can ignore internal boundaries. For internal boundaries we
    // would unnecessarily have to copy mortar data around to emulate the
    // communication step, so by just skipping internal boundaries we avoid
    // that.
    const size_t num_points = mesh.number_of_grid_points();
    const Variables<tmpl::list<PrimalFields...>> zero_primal_vars{num_points,
                                                                  0.};
    Variables<tmpl::list<PrimalFluxes...>> primal_fluxes_buffer{num_points};
    Variables<tmpl::list<AuxiliaryFields...>> unused_aux_vars_buffer{};
    Variables<tmpl::list<AuxiliaryFluxes...>> unused_aux_fluxes_buffer{};
    Variables<tmpl::list<FixedSourcesTags...>> operator_applied_to_zero_vars{
        num_points};
    // Set up data on mortars. We only need them at external boundaries.
    ::dg::MortarMap<Dim, MortarData<size_t, tmpl::list<PrimalFields...>,
                                    tmpl::list<PrimalFluxes...>>>
        all_mortar_data{};
    constexpr size_t temporal_id = std::numeric_limits<size_t>::max();
    // Apply the operator to the zero variables, skipping internal boundaries
    prepare_mortar_data<true>(
        make_not_null(&unused_aux_vars_buffer),
        make_not_null(&unused_aux_fluxes_buffer),
        make_not_null(&primal_fluxes_buffer),
        make_not_null(&all_mortar_data), zero_primal_vars, element, mesh,
        inv_jacobian, {}, external_face_normals, {},
        external_face_normal_magnitudes, all_mortar_meshes, all_mortar_sizes,
        temporal_id, apply_boundary_condition, fluxes_args, sources_args,
        fluxes_args_on_internal_faces, fluxes_args_on_external_faces);
    apply_operator<true>(make_not_null(&operator_applied_to_zero_vars),
                         make_not_null(&all_mortar_data), zero_primal_vars,
                         primal_fluxes_buffer, mesh, inv_jacobian, {},
                         external_face_normal_magnitudes, all_mortar_meshes,
                         all_mortar_sizes, penalty_parameter, temporal_id,
                         sources_args);
    // Impose the nonlinear (constant) boundary contribution as fixed sources on
    // the RHS of the equations
    *fixed_sources -= operator_applied_to_zero_vars;
  }
};

}  // namespace detail

/*!
 * \brief Prepare data on mortars so they can be communicated to neighbors
 *
 * Call this function on all elements and communicate the mortar data, then call
 * `elliptic::dg::apply_operator`.
 */
template <typename System, bool Linearized, typename... Args>
void prepare_mortar_data(Args&&... args) noexcept {
  detail::DgOperatorImpl<System, Linearized>::template prepare_mortar_data<
      false>(std::forward<Args>(args)...);
}

/*!
 * \brief Apply the elliptic DG operator
 *
 * This function applies the elliptic DG operator on an element, assuming all
 * data on mortars is already available. Use the
 * `elliptic::dg::prepare_mortar_data` function to prepare mortar data on
 * neighboring elements, then communicate the data and insert them on the
 * "remote" side of the mortars before calling this function.
 */
template <typename System, bool Linearized, typename... Args>
void apply_operator(Args&&... args) noexcept {
  detail::DgOperatorImpl<System, Linearized>::template apply_operator<false>(
      std::forward<Args>(args)...);
}

/*!
 * \brief For linear systems, impose inhomogeneous boundary conditions as
 * contributions to the fixed sources (i.e. the RHS of the equations).
 *
 * This function exists because the DG operator must typically be linear, but
 * even for linear elliptic equations we typically apply boundary conditions
 * with a constant, and therefore nonlinear, contribution. Standard examples are
 * inhomogeneous (i.e. non-zero) Dirichlet or Neumann boundary conditions. This
 * nonlinear contribution can be added to the fixed sources, leaving only the
 * linearized boundary conditions in the DG operator. For standard constant
 * Dirichlet or Neumann boundary conditions the linearization is of course just
 * zero.
 *
 * This function essentially feeds zero variables through the nonlinear operator
 * and subtracts the result from the fixed sources: `b -= A(x=0)`.
 */
template <typename System, typename... Args>
void impose_inhomogeneous_boundary_conditions_on_source(
    Args&&... args) noexcept {
  detail::DgOperatorImpl<System, false>::
      impose_inhomogeneous_boundary_conditions_on_source(
          std::forward<Args>(args)...);
}

}  // namespace elliptic::dg
