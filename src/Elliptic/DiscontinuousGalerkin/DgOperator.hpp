// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/Variables/FrameTransform.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Elliptic/Protocols/FirstOrderSystem.hpp"
#include "Elliptic/Systems/GetSourcesComputer.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ApplyMassMatrix.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFromBoundary.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleMortarData.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/WeakDivergence.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/*!
 * \brief Functionality related to discontinuous Galerkin discretizations of
 * elliptic equations
 *
 * The following is a brief overview of the elliptic DG schemes that are
 * implemented here. The scheme is described in detail in \cite Fischer2021voj.
 *
 * The DG schemes apply to any elliptic PDE that can be formulated in
 * first-order flux-form, as detailed by
 * `elliptic::protocols::FirstOrderSystem`.
 * The DG discretization of equations in this first-order form amounts to
 * projecting the equations on the set of basis functions that we also use to
 * represent the fields on the computational grid. The currently implemented DG
 * operator uses Lagrange interpolating polynomials w.r.t. Legendre-Gauss or
 * Legendre-Gauss-Lobatto collocation points as basis functions. Skipping all
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
 * \label{eq:internal_penalty_auxiliary}
 * u^* &= \frac{1}{2} \left(u^\mathrm{int} + u^\mathrm{ext}\right) \\
 * \label{eq:internal_penalty_primal}
 * (n_i F^i)^* &= \frac{1}{2} n_i \left(
 * F^i_\mathrm{int} + F^i_\mathrm{ext} \right)
 * - \sigma n_i F^i(n_j (u^\mathrm{int} - u^\mathrm{ext}))
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
 * communication we could replace the derivatives in \f$(n_i F^i)^*\f$ with
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
 * u^* &= \frac{1}{2} \left(u^\mathrm{int} + u^\mathrm{ext}\right) \\
 * (n_i F^i)^* &= n_i v_i^* = \frac{1}{2} n_i \left(
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
 *
 * \par Discontinuous fluxes:
 * The DG operator also supports systems with potentially discontinuous fluxes,
 * such as elasticity with layered materials. The way to handle the
 * discontinuous fluxes in the DG scheme is described in \cite Vu2023thn.
 * Essentially, we evaluate the penalty term in
 * Eq. $\ref{eq:internal_penalty_primal}$ on both sides of an element boundary
 * and take the average. The other terms in the numerical flux remain unchanged.
 */
namespace elliptic::dg {

/// Data that is projected to mortars and communicated across element
/// boundaries
template <typename PrimalFields, typename PrimalFluxes>
using BoundaryData = ::dg::SimpleBoundaryData<
    tmpl::append<PrimalFields,
                 db::wrap_tags_in<::Tags::NormalDotFlux, PrimalFields>>,
    tmpl::list<>>;

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

// Mass-conservative restriction: R = M^{-1}_face P^T M_mortar
//
// Note that projecting the mortar data times the Jacobian using
// `Spectral::projection_matrix_child_to_parent(operand_is_massive=false)` is
// equivalent to this implementation on Gauss grids. However, on Gauss-Lobatto
// grids we usually diagonally approximate the mass matrix ("mass lumping") but
// `projection_matrix_child_to_parent(operand_is_massive=false)` uses the full
// mass matrix. Therefore, the two implementations differ slightly on
// Gauss-Lobatto grids. Furthermore, note that
// `projection_matrix_child_to_parent(operand_is_massive=false)` already
// includes the factors of two that account for the mortar size, so they must be
// omitted from the mortar Jacobian when using that approach.
template <typename TagsList, size_t FaceDim>
Variables<TagsList> mass_conservative_restriction(
    Variables<TagsList> mortar_vars,
    [[maybe_unused]] const Mesh<FaceDim>& mortar_mesh,
    [[maybe_unused]] const ::dg::MortarSize<FaceDim>& mortar_size,
    [[maybe_unused]] const Scalar<DataVector>& mortar_jacobian,
    [[maybe_unused]] const Mesh<FaceDim> face_mesh,
    [[maybe_unused]] const Scalar<DataVector>& face_jacobian) {
  if constexpr (FaceDim == 0) {
    return mortar_vars;
  } else {
    const auto projection_matrices =
        Spectral::projection_matrix_child_to_parent(mortar_mesh, face_mesh,
                                                    mortar_size, true);
    mortar_vars *= get(mortar_jacobian);
    ::dg::apply_mass_matrix(make_not_null(&mortar_vars), mortar_mesh);
    auto face_vars =
        apply_matrices(projection_matrices, mortar_vars, mortar_mesh.extents());
    face_vars /= get(face_jacobian);
    ::dg::apply_inverse_mass_matrix(make_not_null(&face_vars), face_mesh);
    return face_vars;
  }
}

template <typename System, bool Linearized,
          typename PrimalFields = typename System::primal_fields,
          typename PrimalFluxes = typename System::primal_fluxes>
struct DgOperatorImpl;

template <typename System, bool Linearized, typename... PrimalFields,
          typename... PrimalFluxes>
struct DgOperatorImpl<System, Linearized, tmpl::list<PrimalFields...>,
                      tmpl::list<PrimalFluxes...>> {
  static_assert(
      tt::assert_conforms_to_v<System, elliptic::protocols::FirstOrderSystem>);

  static constexpr size_t Dim = System::volume_dim;
  using FluxesComputer = typename System::fluxes_computer;
  using SourcesComputer = elliptic::get_sources_computer<System, Linearized>;

  struct AllDirections {
    bool operator()(const Direction<Dim>& /*unused*/) const { return true; }
  };

  struct NoDataIsZero {
    bool operator()(const ElementId<Dim>& /*unused*/) const { return false; }
  };

  static constexpr auto full_mortar_size =
      make_array<Dim - 1>(Spectral::MortarSize::Full);

  template <bool AllDataIsZero, typename... DerivTags, typename... PrimalVars,
            typename... PrimalFluxesVars, typename... PrimalMortarVars,
            typename... PrimalMortarFluxes, typename TemporalId,
            typename ApplyBoundaryCondition, typename... FluxesArgs,
            typename... SourcesArgs, typename DataIsZero = NoDataIsZero,
            typename DirectionsPredicate = AllDirections>
  static void prepare_mortar_data(
      const gsl::not_null<Variables<tmpl::list<DerivTags...>>*> deriv_vars,
      const gsl::not_null<Variables<tmpl::list<PrimalFluxesVars...>>*>
          primal_fluxes,
      const gsl::not_null<::dg::MortarMap<
          Dim, MortarData<TemporalId, tmpl::list<PrimalMortarVars...>,
                          tmpl::list<PrimalMortarFluxes...>>>*>
          all_mortar_data,
      const Variables<tmpl::list<PrimalVars...>>& primal_vars,
      const Element<Dim>& element, const Mesh<Dim>& mesh,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian,
      const DirectionMap<Dim, tnsr::i<DataVector, Dim>>& face_normals,
      const ::dg::MortarMap<Dim, Mesh<Dim - 1>>& all_mortar_meshes,
      const ::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>& all_mortar_sizes,
      const TemporalId& temporal_id,
      const ApplyBoundaryCondition& apply_boundary_condition,
      const std::tuple<FluxesArgs...>& fluxes_args,
      const DataIsZero& data_is_zero = NoDataIsZero{},
      const DirectionsPredicate& directions_predicate = AllDirections{}) {
    static_assert(
        sizeof...(PrimalVars) == sizeof...(PrimalFields) and
            sizeof...(PrimalFluxesVars) == sizeof...(PrimalFluxes),
        "The number of variables must match the number of system fields.");
    static_assert(
        (std::is_same_v<typename PrimalVars::type,
                        typename PrimalFields::type> and
         ...) and
            (std::is_same_v<typename PrimalFluxesVars::type,
                            typename PrimalFluxes::type> and
             ...),
        "The variables must have the same tensor types as the system fields.");
#ifdef SPECTRE_DEBUG
    for (size_t d = 0; d < Dim; ++d) {
      ASSERT(mesh.basis(d) == Spectral::Basis::Legendre and
                 (mesh.quadrature(d) == Spectral::Quadrature::GaussLobatto or
                  mesh.quadrature(d) == Spectral::Quadrature::Gauss),
             "The elliptic DG operator is currently only implemented for "
             "Legendre-Gauss(-Lobatto) grids. Found basis '"
                 << mesh.basis(d) << "' and quadrature '" << mesh.quadrature(d)
                 << "' in dimension " << d << ".");
    }
#endif  // SPECTRE_DEBUG
    const auto& element_id = element.id();
    const bool local_data_is_zero = data_is_zero(element_id);
    ASSERT(Linearized or not local_data_is_zero,
           "Only a linear operator can take advantage of the knowledge that "
           "the operand is zero. Don't return 'true' in 'data_is_zero' unless "
           "you also set 'Linearized' to 'true'.");
    const size_t num_points = mesh.number_of_grid_points();

    // This function and the one below allocate various Variables to compute
    // intermediate quantities. It could be a performance optimization to reduce
    // the number of these allocations and/or move some of the memory buffers
    // into the DataBox to keep them around permanently. The latter should be
    // informed by profiling.

    // Compute partial derivatives grad(u) of the system variables, and from
    // those the fluxes F^i(grad(u)) in the volume. We will take the divergence
    // of the fluxes in the `apply_operator` function below to compute the full
    // elliptic equation -div(F) + S = f(x).
    if (AllDataIsZero or local_data_is_zero) {
      primal_fluxes->initialize(num_points, 0.);
    } else {
      // Compute partial derivatives of the variables
      partial_derivatives(deriv_vars, primal_vars, mesh, inv_jacobian);
      // Compute the fluxes
      primal_fluxes->initialize(num_points);
      std::apply(
          [&primal_fluxes, &primal_vars, &deriv_vars,
           &element_id](const auto&... expanded_fluxes_args) {
            if constexpr (FluxesComputer::is_discontinuous) {
              FluxesComputer::apply(
                  make_not_null(&get<PrimalFluxesVars>(*primal_fluxes))...,
                  expanded_fluxes_args..., element_id,
                  get<PrimalVars>(primal_vars)...,
                  get<DerivTags>(*deriv_vars)...);
            } else {
              (void)element_id;
              FluxesComputer::apply(
                  make_not_null(&get<PrimalFluxesVars>(*primal_fluxes))...,
                  expanded_fluxes_args..., get<PrimalVars>(primal_vars)...,
                  get<DerivTags>(*deriv_vars)...);
            }
          },
          fluxes_args);
    }

    // Populate the mortar data on this element's side of the boundary so it's
    // ready to be sent to neighbors.
    for (const auto& direction : [&element]() -> const auto& {
           if constexpr (AllDataIsZero) {
             // Skipping internal boundaries for all-zero data because they
             // won't contribute boundary corrections anyway (data on both sides
             // of the boundary is the same). For all-zero data we are
             // interested in external boundaries, to extract inhomogeneous
             // boundary conditions from a non-linear operator.
             return element.external_boundaries();
           } else {
             (void)element;
             return Direction<Dim>::all_directions();
           };
         }()) {
      if (not directions_predicate(direction)) {
        continue;
      }
      const bool is_internal = element.neighbors().contains(direction);
      // Skip directions altogether when both this element and all neighbors in
      // the direction have zero data. These boundaries won't contribute
      // corrections, because the data is the same on both sides. External
      // boundaries also count as zero data here, because they are linearized
      // (see assert above).
      if (local_data_is_zero and
          (not is_internal or
           alg::all_of(element.neighbors().at(direction), data_is_zero))) {
        continue;
      }
      const auto face_mesh = mesh.slice_away(direction.dimension());
      const size_t face_num_points = face_mesh.number_of_grid_points();
      const auto& face_normal = face_normals.at(direction);
      Variables<tmpl::list<PrimalFluxesVars...>> primal_fluxes_on_face{};
      BoundaryData<tmpl::list<PrimalMortarVars...>,
                   tmpl::list<PrimalMortarFluxes...>>
          boundary_data{};
      if (AllDataIsZero or local_data_is_zero) {
        if (is_internal) {
          // We manufacture zero boundary data directly on the mortars below.
          // Nothing to do here.
        } else {
          boundary_data.field_data.initialize(face_num_points, 0.);
        }
      } else {
        boundary_data.field_data.initialize(face_num_points);
        primal_fluxes_on_face.initialize(face_num_points);
        // Project fields to faces
        // Note: need to convert tags of `Variables` because
        // `project_contiguous_data_to_boundary` requires that the face and
        // volume tags are subsets.
        Variables<
            tmpl::list<PrimalVars..., ::Tags::NormalDotFlux<PrimalVars>...>>
            boundary_data_ref{};
        boundary_data_ref.set_data_ref(boundary_data.field_data.data(),
                                       boundary_data.field_data.size());
        ::dg::project_contiguous_data_to_boundary(
            make_not_null(&boundary_data_ref), primal_vars, mesh, direction);
        // Compute n_i F^i on faces
        ::dg::project_contiguous_data_to_boundary(
            make_not_null(&primal_fluxes_on_face), *primal_fluxes, mesh,
            direction);
        EXPAND_PACK_LEFT_TO_RIGHT(normal_dot_flux(
            make_not_null(&get<::Tags::NormalDotFlux<PrimalMortarVars>>(
                boundary_data.field_data)),
            face_normal, get<PrimalFluxesVars>(primal_fluxes_on_face)));
      }

      if (is_internal) {
        if constexpr (not AllDataIsZero) {
          // Project boundary data on internal faces to mortars
          for (const auto& neighbor_id : element.neighbors().at(direction)) {
            if (local_data_is_zero and data_is_zero(neighbor_id)) {
              continue;
            }
            const ::dg::MortarId<Dim> mortar_id{direction, neighbor_id};
            const auto& mortar_mesh = all_mortar_meshes.at(mortar_id);
            const auto& mortar_size = all_mortar_sizes.at(mortar_id);
            if (local_data_is_zero) {
              // No need to project anything. We just manufacture zero boundary
              // data on the mortar.
              BoundaryData<tmpl::list<PrimalMortarVars...>,
                           tmpl::list<PrimalMortarFluxes...>>
                  zero_boundary_data{};
              zero_boundary_data.field_data.initialize(
                  mortar_mesh.number_of_grid_points(), 0.);
              (*all_mortar_data)[mortar_id].local_insert(
                  temporal_id, std::move(zero_boundary_data));
              continue;
            }
            // When no projection is necessary we can safely move the boundary
            // data from the face as there is only a single neighbor in this
            // direction
            auto projected_boundary_data =
                Spectral::needs_projection(face_mesh, mortar_mesh, mortar_size)
                    // NOLINTNEXTLINE
                    ? boundary_data.project_to_mortar(face_mesh, mortar_mesh,
                                                      mortar_size)
                    : std::move(boundary_data);  // NOLINT
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
        // imposed by modifying the interior n.F. Note that all data passed to
        // the boundary conditions is taken from the "interior" side of the
        // boundary, i.e. with a normal vector that points _out_ of the
        // computational domain.
        Variables<tmpl::list<DerivTags...>> deriv_vars_on_boundary{};
        if (AllDataIsZero or local_data_is_zero) {
          deriv_vars_on_boundary.initialize(face_num_points, 0.);
        } else {
          deriv_vars_on_boundary.initialize(face_num_points);
          ::dg::project_contiguous_data_to_boundary(
              make_not_null(&deriv_vars_on_boundary), *deriv_vars, mesh,
              direction);
        }
        apply_boundary_condition(
            direction,
            make_not_null(&get<PrimalMortarVars>(boundary_data.field_data))...,
            make_not_null(&get<::Tags::NormalDotFlux<PrimalMortarVars>>(
                boundary_data.field_data))...,
            get<DerivTags>(deriv_vars_on_boundary)...);

        // Invert the sign of the fluxes to account for the inverted normal on
        // exterior faces. Also multiply by 2 and add the interior fluxes to
        // impose the boundary conditions on the _average_ instead of just
        // setting the fields on the exterior:
        //   (Dirichlet)  u_D = avg(u) = 1/2 (u_int + u_ext)
        //                => u_ext = 2 u_D - u_int
        //   (Neumann)    (n.F)_N = avg(n.F) = 1/2 [(n.F)_int - (n.F)_ext]
        //                => (n.F)_ext = -2 (n.F)_N + (n.F)_int]
        const auto impose_on_average = [](const auto exterior_field,
                                          const auto& interior_field) {
          for (size_t i = 0; i < interior_field.size(); ++i) {
            (*exterior_field)[i] *= 2.;
            (*exterior_field)[i] -= interior_field[i];
          }
        };
        EXPAND_PACK_LEFT_TO_RIGHT(impose_on_average(
            make_not_null(&get<PrimalMortarVars>(boundary_data.field_data)),
            get<PrimalMortarVars>(all_mortar_data->at(mortar_id)
                                      .local_data(temporal_id)
                                      .field_data)));
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

        // Store the exterior boundary data on the mortar
        all_mortar_data->at(mortar_id).remote_insert(temporal_id,
                                                     std::move(boundary_data));
      }  // if (is_internal)
    }    // loop directions
  }

  // --- This is essentially a break to communicate the mortar data ---

  template <bool AllDataIsZero, typename... OperatorTags,
            typename... PrimalVars, typename... PrimalFluxesVars,
            typename... PrimalMortarVars, typename... PrimalMortarFluxes,
            typename TemporalId, typename... FluxesArgs,
            typename... SourcesArgs, typename DataIsZero = NoDataIsZero,
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
      const Element<Dim>& element, const Mesh<Dim>& mesh,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian,
      const Scalar<DataVector>& det_inv_jacobian,
      const Scalar<DataVector>& det_jacobian,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>& det_times_inv_jacobian,
      const DirectionMap<Dim, tnsr::i<DataVector, Dim>>& face_normals,
      const DirectionMap<Dim, tnsr::I<DataVector, Dim>>& face_normal_vectors,
      const DirectionMap<Dim, Scalar<DataVector>>& face_normal_magnitudes,
      const DirectionMap<Dim, Scalar<DataVector>>& face_jacobians,
      const DirectionMap<Dim,
                         InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                                         Frame::Inertial>>&
          face_jacobian_times_inv_jacobians,
      const ::dg::MortarMap<Dim, Mesh<Dim - 1>>& all_mortar_meshes,
      const ::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>& all_mortar_sizes,
      const ::dg::MortarMap<Dim, Scalar<DataVector>>& mortar_jacobians,
      const ::dg::MortarMap<Dim, Scalar<DataVector>>& penalty_factors,
      const bool massive, const ::dg::Formulation formulation,
      const TemporalId& /*temporal_id*/,
      const DirectionMap<Dim, std::tuple<FluxesArgs...>>& fluxes_args_on_faces,
      const std::tuple<SourcesArgs...>& sources_args,
      const DataIsZero& data_is_zero = NoDataIsZero{},
      const DirectionsPredicate& directions_predicate = AllDirections{}) {
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
                 (mesh.quadrature(d) == Spectral::Quadrature::GaussLobatto or
                  mesh.quadrature(d) == Spectral::Quadrature::Gauss),
             "The elliptic DG operator is currently only implemented for "
             "Legendre-Gauss(-Lobatto) grids. Found basis '"
                 << mesh.basis(d) << "' and quadrature '" << mesh.quadrature(d)
                 << "' in dimension " << d << ".");
    }
#endif  // SPECTRE_DEBUG
    const auto& element_id = element.id();
    const bool local_data_is_zero = data_is_zero(element_id);
    ASSERT(Linearized or not local_data_is_zero,
           "Only a linear operator can take advantage of the knowledge that "
           "the operand is zero. Don't return 'true' in 'data_is_zero' unless "
           "you also set 'Linearized' to 'true'.");
    const size_t num_points = mesh.number_of_grid_points();

    // This function and the one above allocate various Variables to compute
    // intermediate quantities. It could be a performance optimization to reduce
    // the number of these allocations and/or move some of the memory buffers
    // into the DataBox to keep them around permanently. The latter should be
    // informed by profiling.

    // Compute volume terms: -div(F) + S
    if (local_data_is_zero) {
      operator_applied_to_vars->initialize(num_points, 0.);
    } else {
      // "Massive" operators retain the factors from the volume integral:
      //   \int_volume div(F) \phi_p = w_p det(J)_p div(F)_p
      // Here, `w` are the quadrature weights (the diagonal logical mass matrix
      // with mass-lumping) and det(J) is the Jacobian determinant. The
      // quantities are evaluated at the grid point `p`.
      if (formulation == ::dg::Formulation::StrongInertial) {
        // Compute strong divergence:
        //   div(F) = (J^\hat{i}_i)_p \sum_q (D_\hat{i})_pq (F^i)_q.
        divergence(operator_applied_to_vars, primal_fluxes, mesh,
                   massive ? det_times_inv_jacobian : inv_jacobian);
        // This is the sign flip that makes the operator _minus_ the Laplacian
        // for a Poisson system
        *operator_applied_to_vars *= -1.;
      } else {
        // Compute weak divergence:
        //   F^i \partial_i \phi = 1/w_p \sum_q
        //     (D^T_\hat{i})_pq (w det(J) J^\hat{i}_i F^i)_q
        weak_divergence(operator_applied_to_vars, primal_fluxes, mesh,
                        det_times_inv_jacobian);
        if (not massive) {
          *operator_applied_to_vars *= get(det_inv_jacobian);
        }
      }
      if constexpr (not std::is_same_v<SourcesComputer, void>) {
        Variables<tmpl::list<OperatorTags...>> sources{num_points, 0.};
        std::apply(
            [&sources, &primal_vars,
             &primal_fluxes](const auto&... expanded_sources_args) {
              SourcesComputer::apply(
                  make_not_null(&get<OperatorTags>(sources))...,
                  expanded_sources_args..., get<PrimalVars>(primal_vars)...,
                  get<PrimalFluxesVars>(primal_fluxes)...);
            },
            sources_args);
        if (massive) {
          sources *= get(det_jacobian);
        }
        *operator_applied_to_vars += sources;
      }
    }
    if (massive) {
      ::dg::apply_mass_matrix(operator_applied_to_vars, mesh);
    }

    // Add boundary corrections
    // Keeping track if any corrections were applied here, for an optimization
    // below
    bool has_any_boundary_corrections = false;
    Variables<tmpl::list<transform::Tags::TransformedFirstIndex<
        PrimalFluxesVars, Frame::ElementLogical>...>>
        lifted_logical_aux_boundary_corrections{num_points, 0.};
    for (auto& [mortar_id, mortar_data] : *all_mortar_data) {
      const auto& direction = mortar_id.direction();
      const auto& neighbor_id = mortar_id.id();
      const bool is_internal =
          (neighbor_id != ElementId<Dim>::external_boundary_id());
      if constexpr (AllDataIsZero) {
        if (is_internal) {
          continue;
        }
      }
      if (not directions_predicate(direction)) {
        continue;
      }
      // When the data on both sides of the mortar is zero then we don't need to
      // handle this mortar at all.
      if (local_data_is_zero and
          (not is_internal or data_is_zero(neighbor_id))) {
        continue;
      }
      has_any_boundary_corrections = true;

      const auto face_mesh = mesh.slice_away(direction.dimension());
      auto [local_data, remote_data] = mortar_data.extract();
      const size_t face_num_points = face_mesh.number_of_grid_points();
      const auto& face_normal = face_normals.at(direction);
      const auto& face_normal_vector = face_normal_vectors.at(direction);
      const auto& fluxes_args_on_face = fluxes_args_on_faces.at(direction);
      const auto& face_normal_magnitude = face_normal_magnitudes.at(direction);
      const auto& face_jacobian = face_jacobians.at(direction);
      const auto& face_jacobian_times_inv_jacobian =
          face_jacobian_times_inv_jacobians.at(direction);
      const auto& mortar_mesh =
          is_internal ? all_mortar_meshes.at(mortar_id) : face_mesh;
      const auto& mortar_size =
          is_internal ? all_mortar_sizes.at(mortar_id) : full_mortar_size;

      // This is the strong auxiliary boundary correction:
      //   G^i = F^i(n_j (avg(u) - u))
      // where
      //   avg(u) - u = -0.5 * (u_int - u_ext)
      auto avg_vars_on_mortar = Variables<tmpl::list<PrimalMortarVars...>>(
          local_data.field_data
              .template extract_subset<tmpl::list<PrimalMortarVars...>>());
      const auto add_remote_contribution = [](auto& lhs, const auto& rhs) {
        for (size_t i = 0; i < lhs.size(); ++i) {
          lhs[i] -= rhs[i];
        }
      };
      EXPAND_PACK_LEFT_TO_RIGHT(add_remote_contribution(
          get<PrimalMortarVars>(avg_vars_on_mortar),
          get<PrimalMortarVars>(remote_data.field_data)));
      avg_vars_on_mortar *= -0.5;

      // Project from the mortar back down to the face if needed
      const auto avg_vars_on_face =
          Spectral::needs_projection(face_mesh, mortar_mesh, mortar_size)
              ? mass_conservative_restriction(
                    std::move(avg_vars_on_mortar), mortar_mesh, mortar_size,
                    mortar_jacobians.at(mortar_id), face_mesh, face_jacobian)
              : std::move(avg_vars_on_mortar);

      // Apply fluxes to get G^i
      Variables<tmpl::list<PrimalFluxesVars...>> auxiliary_boundary_corrections{
          face_num_points};
      std::apply(
          [&auxiliary_boundary_corrections, &face_normal, &face_normal_vector,
           &avg_vars_on_face,
           &element_id](const auto&... expanded_fluxes_args_on_face) {
            if constexpr (FluxesComputer::is_discontinuous) {
              FluxesComputer::apply(make_not_null(&get<PrimalFluxesVars>(
                                        auxiliary_boundary_corrections))...,
                                    expanded_fluxes_args_on_face..., element_id,
                                    face_normal, face_normal_vector,
                                    get<PrimalMortarVars>(avg_vars_on_face)...);
            } else {
              (void)element_id;
              FluxesComputer::apply(make_not_null(&get<PrimalFluxesVars>(
                                        auxiliary_boundary_corrections))...,
                                    expanded_fluxes_args_on_face...,
                                    face_normal, face_normal_vector,
                                    get<PrimalMortarVars>(avg_vars_on_face)...);
            }
          },
          fluxes_args_on_face);

      // Lifting for the auxiliary boundary correction:
      //   \int_face G^i \partial_i \phi
      // We first transform the flux index to the logical frame, apply the
      // quadrature weights and the Jacobian for the face integral, then take
      // the logical weak divergence in the volume after lifting (below the loop
      // over faces).
      auto logical_aux_boundary_corrections =
          transform::first_index_to_different_frame(
              auxiliary_boundary_corrections, face_jacobian_times_inv_jacobian);
      ::dg::apply_mass_matrix(make_not_null(&logical_aux_boundary_corrections),
                              face_mesh);
      if (mesh.quadrature(0) == Spectral::Quadrature::GaussLobatto) {
        add_slice_to_data(
            make_not_null(&lifted_logical_aux_boundary_corrections),
            logical_aux_boundary_corrections, mesh.extents(),
            direction.dimension(),
            index_to_slice_at(mesh.extents(), direction));
      } else {
        ::dg::lift_boundary_terms_gauss_points(
            make_not_null(&lifted_logical_aux_boundary_corrections),
            logical_aux_boundary_corrections, mesh, direction);
      }

      // This is the strong primal boundary correction:
      //   -n.H = -avg(n.F) + n.F + penalty * n.F(n_j jump(u))
      // Note that the "internal penalty" numerical flux
      // (as opposed to the LLF flux) uses the raw field derivatives without
      // boundary corrections in the average, which is why we can communicate
      // the data so early together with the auxiliary boundary data. In this
      // case the penalty needs to include a factor N_points^2 / h (see the
      // `penalty` function).
      const auto& penalty_factor = penalty_factors.at(mortar_id);
      // Compute jump on mortar:
      //   penalty * jump(u) = penalty * (u_int - u_ext)
      const auto add_remote_jump_contribution =
          [&penalty_factor](auto& lhs, const auto& rhs) {
            for (size_t i = 0; i < lhs.size(); ++i) {
              lhs[i] -= rhs[i];
              lhs[i] *= get(penalty_factor);
            }
          };
      EXPAND_PACK_LEFT_TO_RIGHT(add_remote_jump_contribution(
          get<PrimalMortarVars>(local_data.field_data),
          get<PrimalMortarVars>(remote_data.field_data)));
      // Compute average on mortar:
      //   (strong)  -avg(n.F) + n.F = 0.5 * (n.F)_int + 0.5 * (n.F)_ext
      //   (weak)    -avg(n.F) = -0.5 * (n.F)_int + 0.5 * (n.F)_ext
      const auto add_avg_contribution = [](auto& lhs, const auto& rhs,
                                           const double factor) {
        for (size_t i = 0; i < lhs.size(); ++i) {
          lhs[i] *= factor;
          lhs[i] += 0.5 * rhs[i];
        }
      };
      EXPAND_PACK_LEFT_TO_RIGHT(add_avg_contribution(
          get<::Tags::NormalDotFlux<PrimalMortarVars>>(local_data.field_data),
          get<::Tags::NormalDotFlux<PrimalMortarVars>>(remote_data.field_data),
          formulation == ::dg::Formulation::StrongInertial ? 0.5 : -0.5));

      // Project from the mortar back down to the face if needed, lift and add
      // to operator. See auxiliary boundary corrections above for details.
      auto primal_boundary_corrections_on_face =
          Spectral::needs_projection(face_mesh, mortar_mesh, mortar_size)
              ? mass_conservative_restriction(
                    std::move(local_data.field_data), mortar_mesh, mortar_size,
                    mortar_jacobians.at(mortar_id), face_mesh, face_jacobian)
              : std::move(local_data.field_data);

      // Compute fluxes for jump term: n.F(n_j jump(u))
      // If the fluxes are trivial (just the spatial metric), we can skip this
      // step because the face normal is normalized.
      if constexpr (not FluxesComputer::is_trivial) {
        // We reuse the memory buffer from above for the result.
        std::apply(
            [&auxiliary_boundary_corrections, &face_normal, &face_normal_vector,
             &primal_boundary_corrections_on_face,
             &element_id](const auto&... expanded_fluxes_args_on_face) {
              if constexpr (FluxesComputer::is_discontinuous) {
                FluxesComputer::apply(
                    make_not_null(&get<PrimalFluxesVars>(
                        auxiliary_boundary_corrections))...,
                    expanded_fluxes_args_on_face..., element_id, face_normal,
                    face_normal_vector,
                    get<PrimalMortarVars>(
                        primal_boundary_corrections_on_face)...);
              } else {
                (void)element_id;
                FluxesComputer::apply(
                    make_not_null(&get<PrimalFluxesVars>(
                        auxiliary_boundary_corrections))...,
                    expanded_fluxes_args_on_face..., face_normal,
                    face_normal_vector,
                    get<PrimalMortarVars>(
                        primal_boundary_corrections_on_face)...);
              }
            },
            fluxes_args_on_face);
        if constexpr (FluxesComputer::is_discontinuous) {
          if (is_internal) {
            // For penalty term with discontinuous fluxes: evaluate the fluxes
            // on the other side of the boundary as well and take average
            Variables<tmpl::list<PrimalFluxesVars...>> fluxes_other_side{
                face_num_points};
            std::apply(
                [&fluxes_other_side, &face_normal, &face_normal_vector,
                 &primal_boundary_corrections_on_face,
                 &local_neighbor_id =
                     neighbor_id](const auto&... expanded_fluxes_args_on_face) {
                  FluxesComputer::apply(
                      make_not_null(
                          &get<PrimalFluxesVars>(fluxes_other_side))...,
                      expanded_fluxes_args_on_face..., local_neighbor_id,
                      face_normal, face_normal_vector,
                      get<PrimalMortarVars>(
                          primal_boundary_corrections_on_face)...);
                },
                fluxes_args_on_face);
            auxiliary_boundary_corrections += fluxes_other_side;
            auxiliary_boundary_corrections *= 0.5;
          }
        }
        EXPAND_PACK_LEFT_TO_RIGHT(normal_dot_flux(
            make_not_null(
                &get<PrimalMortarVars>(primal_boundary_corrections_on_face)),
            face_normal,
            get<PrimalFluxesVars>(auxiliary_boundary_corrections)));
      }

      // Add penalty term to average term
      Variables<tmpl::list<PrimalMortarVars...>> primal_boundary_corrections{};
      // First half of the memory allocated above is filled with the penalty
      // term, so just use that memory here.
      primal_boundary_corrections.set_data_ref(
          primal_boundary_corrections_on_face.data(), avg_vars_on_face.size());
      // Second half of the memory is filled with the average term. Add that to
      // the penalty term.
      const auto add_avg_term = [](auto& lhs, const auto& rhs) {
        for (size_t i = 0; i < lhs.size(); ++i) {
          lhs[i] += rhs[i];
        }
      };
      EXPAND_PACK_LEFT_TO_RIGHT(
          add_avg_term(get<PrimalMortarVars>(primal_boundary_corrections),
                       get<::Tags::NormalDotFlux<PrimalMortarVars>>(
                           primal_boundary_corrections_on_face)));

      // Lifting for the primal boundary correction:
      //   \int_face n.H \phi
      if (massive) {
        // We apply the quadrature weights and Jacobian for the face integral,
        // then lift to the volume
        primal_boundary_corrections *= get(face_jacobian);
        ::dg::apply_mass_matrix(make_not_null(&primal_boundary_corrections),
                                face_mesh);
        if (mesh.quadrature(0) == Spectral::Quadrature::GaussLobatto) {
          add_slice_to_data(operator_applied_to_vars,
                            primal_boundary_corrections, mesh.extents(),
                            direction.dimension(),
                            index_to_slice_at(mesh.extents(), direction));
        } else {
          ::dg::lift_boundary_terms_gauss_points(operator_applied_to_vars,
                                                 primal_boundary_corrections,
                                                 mesh, direction);
        }
      } else {
        // Apply an extra inverse mass matrix to the boundary corrections (with
        // mass lumping, so it's diagonal).
        // For Gauss-Lobatto grids this divides out the quadrature weights and
        // Jacobian on the face, leaving only factors perpendicular to the face.
        // Those are handled by `dg::lift_flux` (with an extra minus sign since
        // the function was written for evolution systems).
        // For Gauss grids the quadrature weights and Jacobians are handled
        // by `::dg::lift_boundary_terms_gauss_points` (which was also written
        // for evolution systems, hence the extra minus sign).
        primal_boundary_corrections *= -1.;
        if (mesh.quadrature(0) == Spectral::Quadrature::GaussLobatto) {
          ::dg::lift_flux(make_not_null(&primal_boundary_corrections),
                          mesh.extents(direction.dimension()),
                          face_normal_magnitude);
          add_slice_to_data(operator_applied_to_vars,
                            primal_boundary_corrections, mesh.extents(),
                            direction.dimension(),
                            index_to_slice_at(mesh.extents(), direction));
        } else {
          // We already have the `face_jacobian = det(J) * magnitude(n)` here,
          // so just pass a constant 1 for `magnitude(n)`. This could be
          // optimized to avoid allocating the vector of ones.
          ::dg::lift_boundary_terms_gauss_points(
              operator_applied_to_vars, det_inv_jacobian, mesh, direction,
              primal_boundary_corrections,
              Scalar<DataVector>{face_mesh.number_of_grid_points(), 1.},
              face_jacobian);
        }
      }
    }  // loop over all mortars

    if (not has_any_boundary_corrections) {
      // No need to handle auxiliary boundary corrections; return early
      return;
    }

    // Apply weak divergence to lifted auxiliary boundary corrections and add to
    // operator
    if (massive) {
      logical_weak_divergence(operator_applied_to_vars,
                              lifted_logical_aux_boundary_corrections, mesh,
                              true);
    } else {
      // Possible optimization: eliminate this allocation by building the
      // inverse mass matrix into `logical_weak_divergence`
      Variables<tmpl::list<OperatorTags...>> massless_aux_boundary_corrections{
          num_points};
      logical_weak_divergence(make_not_null(&massless_aux_boundary_corrections),
                              lifted_logical_aux_boundary_corrections, mesh);
      massless_aux_boundary_corrections *= get(det_inv_jacobian);
      ::dg::apply_inverse_mass_matrix(
          make_not_null(&massless_aux_boundary_corrections), mesh);
      *operator_applied_to_vars += massless_aux_boundary_corrections;
    }
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
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian,
      const Scalar<DataVector>& det_inv_jacobian,
      const Scalar<DataVector>& det_jacobian,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>& det_times_inv_jacobian,
      const DirectionMap<Dim, tnsr::i<DataVector, Dim>>& face_normals,
      const DirectionMap<Dim, tnsr::I<DataVector, Dim>>& face_normal_vectors,
      const DirectionMap<Dim, Scalar<DataVector>>& face_normal_magnitudes,
      const DirectionMap<Dim, Scalar<DataVector>>& face_jacobians,
      const DirectionMap<Dim,
                         InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                                         Frame::Inertial>>&
          face_jacobian_times_inv_jacobians,
      const ::dg::MortarMap<Dim, Mesh<Dim - 1>>& all_mortar_meshes,
      const ::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>& all_mortar_sizes,
      const ::dg::MortarMap<Dim, Scalar<DataVector>>& penalty_factors,
      const bool massive, const ::dg::Formulation formulation,
      const ApplyBoundaryCondition& apply_boundary_condition,
      const std::tuple<FluxesArgs...>& fluxes_args,
      const std::tuple<SourcesArgs...>& sources_args,
      const DirectionMap<Dim, std::tuple<FluxesArgs...>>&
          fluxes_args_on_faces) {
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
    Variables<tmpl::list<
        ::Tags::deriv<PrimalFields, tmpl::size_t<Dim>, Frame::Inertial>...>>
        unused_deriv_vars_buffer{};
    Variables<tmpl::list<FixedSourcesTags...>> operator_applied_to_zero_vars{
        num_points};
    // Set up data on mortars. We only need them at external boundaries.
    ::dg::MortarMap<Dim, MortarData<size_t, tmpl::list<PrimalFields...>,
                                    tmpl::list<PrimalFluxes...>>>
        all_mortar_data{};
    constexpr size_t temporal_id = std::numeric_limits<size_t>::max();
    // Apply the operator to the zero variables, skipping internal boundaries
    prepare_mortar_data<true>(make_not_null(&unused_deriv_vars_buffer),
                              make_not_null(&primal_fluxes_buffer),
                              make_not_null(&all_mortar_data), zero_primal_vars,
                              element, mesh, inv_jacobian, face_normals,
                              all_mortar_meshes, all_mortar_sizes, temporal_id,
                              apply_boundary_condition, fluxes_args);
    apply_operator<true>(
        make_not_null(&operator_applied_to_zero_vars),
        make_not_null(&all_mortar_data), zero_primal_vars, primal_fluxes_buffer,
        element, mesh, inv_jacobian, det_inv_jacobian, det_jacobian,
        det_times_inv_jacobian, face_normals, face_normal_vectors,
        face_normal_magnitudes, face_jacobians,
        face_jacobian_times_inv_jacobians, all_mortar_meshes, all_mortar_sizes,
        {}, penalty_factors, massive, formulation, temporal_id,
        fluxes_args_on_faces, sources_args);
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
void prepare_mortar_data(Args&&... args) {
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
void apply_operator(Args&&... args) {
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
void impose_inhomogeneous_boundary_conditions_on_source(Args&&... args) {
  detail::DgOperatorImpl<System, false>::
      impose_inhomogeneous_boundary_conditions_on_source(
          std::forward<Args>(args)...);
}

}  // namespace elliptic::dg
