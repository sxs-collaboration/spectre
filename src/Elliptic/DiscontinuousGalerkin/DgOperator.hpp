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
#include "Elliptic/DiscontinuousGalerkin/Penalty.hpp"
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
 * u^* &= \frac{1}{2} \left(u^\mathrm{int} + u^\mathrm{ext}\right) \\
 * (n_i F^i)^* &= \frac{1}{2} n_i \left(
 * F^i_\mathrm{int} + F^i_\mathrm{ext} \right)
 * - \sigma n_i F^i(n_j (u^\mathrm{int} - u^\mathrm{ext})) \right)
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
struct ElementSize : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// The quantity \f$n_i F^i(n_j u)\f$ where \f$F^i\f$ is the system flux, and
/// \f$n_i\f$ is the face normal. This quantity is projected to mortars to
/// compute the jump term of the numerical flux.
template <typename Tag>
struct NormalDotFluxForJump : db::PrefixTag, db::SimpleTag {
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
zero_boundary_data_on_mortar(const Direction<Dim>& direction,
                             const Mesh<Dim>& mesh,
                             const Scalar<DataVector>& face_normal_magnitude,
                             const Mesh<Dim - 1>& mortar_mesh,
                             const ::dg::MortarSize<Dim - 1>& mortar_size) {
  const auto face_mesh = mesh.slice_away(direction.dimension());
  const size_t face_num_points = face_mesh.number_of_grid_points();
  BoundaryData<PrimalMortarFields, PrimalMortarFluxes> boundary_data{
      face_num_points};
  boundary_data.field_data.initialize(face_num_points, 0.);
  get<Tags::PerpendicularNumPoints>(boundary_data.extra_data) =
      mesh.extents(direction.dimension());
  // Possible optimization: Store face-normal magnitude on mortars in DataBox,
  // so we don't have to project it here.
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
      const DirectionMap<Dim, tnsr::I<DataVector, Dim>>& face_normal_vectors,
      const DirectionMap<Dim, Scalar<DataVector>>& face_normal_magnitudes,
      const ::dg::MortarMap<Dim, Mesh<Dim - 1>>& all_mortar_meshes,
      const ::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>& all_mortar_sizes,
      const TemporalId& temporal_id,
      const ApplyBoundaryCondition& apply_boundary_condition,
      const std::tuple<FluxesArgs...>& fluxes_args,
      const DirectionMap<Dim, std::tuple<FluxesArgs...>>& fluxes_args_on_faces,
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
    const bool local_data_is_zero = data_is_zero(element.id());
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
          [&primal_fluxes, &primal_vars,
           &deriv_vars](const auto&... expanded_fluxes_args) {
            FluxesComputer::apply(
                make_not_null(&get<PrimalFluxesVars>(*primal_fluxes))...,
                expanded_fluxes_args..., get<PrimalVars>(primal_vars)...,
                get<DerivTags>(*deriv_vars)...);
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
      const auto& face_normal_vector = face_normal_vectors.at(direction);
      const auto& face_normal_magnitude = face_normal_magnitudes.at(direction);
      const auto& fluxes_args_on_face = fluxes_args_on_faces.at(direction);
      Variables<tmpl::list<PrimalFluxesVars...>> primal_fluxes_on_face{};
      Variables<tmpl::list<PrimalVars...>> vars_on_face{};
      BoundaryData<tmpl::list<PrimalMortarVars...>,
                   tmpl::list<PrimalMortarFluxes...>>
          boundary_data{face_num_points};
      if (AllDataIsZero or local_data_is_zero) {
        // Just setting all boundary field data to zero. Variable-independent
        // data such as the element size will be set below.
        boundary_data.field_data.initialize(face_num_points, 0.);
      } else {
        primal_fluxes_on_face.initialize(face_num_points);
        vars_on_face.initialize(face_num_points);
        // Compute F^i(n_j u) on faces
        ::dg::project_contiguous_data_to_boundary(make_not_null(&vars_on_face),
                                                  primal_vars, mesh, direction);
        std::apply(
            [&boundary_data, &face_normal, &face_normal_vector,
             &vars_on_face](const auto&... expanded_fluxes_args_on_face) {
              FluxesComputer::apply(
                  make_not_null(&get<::Tags::NormalDotFlux<PrimalMortarFluxes>>(
                      boundary_data.field_data))...,
                  expanded_fluxes_args_on_face..., face_normal,
                  face_normal_vector, get<PrimalVars>(vars_on_face)...);
            },
            fluxes_args_on_face);
        // Compute n_i F^i on faces
        ::dg::project_contiguous_data_to_boundary(
            make_not_null(&primal_fluxes_on_face), *primal_fluxes, mesh,
            direction);
        EXPAND_PACK_LEFT_TO_RIGHT(normal_dot_flux(
            make_not_null(&get<::Tags::NormalDotFlux<PrimalMortarVars>>(
                boundary_data.field_data)),
            face_normal, get<PrimalFluxesVars>(primal_fluxes_on_face)));
        // Compute n_i F^i(n_j u) for jump term, re-using the F^i(n_j u) from
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
      // optimization could be to store them in the DataBox. In particular, when
      // the `local_data_is_zero` we don't have do projections at all if we
      // store the face-normal magnitude on mortars in the DataBox.
      get<Tags::PerpendicularNumPoints>(boundary_data.extra_data) =
          mesh.extents(direction.dimension());
      get(get<Tags::ElementSize>(boundary_data.field_data)) =
          2. / get(face_normal_magnitude);

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
        if (AllDataIsZero or local_data_is_zero) {
          vars_on_face.initialize(face_num_points, 0.);
        }
        apply_boundary_condition(
            direction, make_not_null(&get<PrimalVars>(vars_on_face))...,
            make_not_null(&get<::Tags::NormalDotFlux<PrimalMortarVars>>(
                boundary_data.field_data))...);

        // The n.F (Neumann-type conditions) are done, but we have to compute
        // fluxes from the Dirichlet fields on the face. We re-use the memory
        // buffer from above.
        std::apply(
            [&boundary_data, &face_normal, &face_normal_vector,
             &vars_on_face](const auto&... expanded_fluxes_args_on_face) {
              FluxesComputer::apply(
                  make_not_null(&get<::Tags::NormalDotFlux<PrimalMortarFluxes>>(
                      boundary_data.field_data))...,
                  expanded_fluxes_args_on_face..., face_normal,
                  face_normal_vector, get<PrimalVars>(vars_on_face)...);
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

        // Compute n_i F^i(n_j u) for jump term
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
      const DirectionMap<Dim, Scalar<DataVector>>& face_normal_magnitudes,
      const DirectionMap<Dim, Scalar<DataVector>>& face_jacobians,
      const DirectionMap<Dim,
                         InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                                         Frame::Inertial>>&
          face_jacobian_times_inv_jacobians,
      const ::dg::MortarMap<Dim, Mesh<Dim - 1>>& all_mortar_meshes,
      const ::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>& all_mortar_sizes,
      const ::dg::MortarMap<Dim, Scalar<DataVector>>& mortar_jacobians,
      const double penalty_parameter, const bool massive,
      const ::dg::Formulation formulation, const TemporalId& /*temporal_id*/,
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
    const bool local_data_is_zero = data_is_zero(element.id());
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
      ASSERT(formulation == ::dg::Formulation::StrongInertial,
             "Only the strong formulation is currently implemented.");
      divergence(operator_applied_to_vars, primal_fluxes, mesh, inv_jacobian);
      // This is the sign flip that makes the operator _minus_ the Laplacian for
      // a Poisson system
      *operator_applied_to_vars *= -1.;
      if constexpr (not std::is_same_v<SourcesComputer, void>) {
        std::apply(
            [&operator_applied_to_vars, &primal_vars,
             &primal_fluxes](const auto&... expanded_sources_args) {
              SourcesComputer::apply(
                  make_not_null(
                      &get<OperatorTags>(*operator_applied_to_vars))...,
                  expanded_sources_args..., get<PrimalVars>(primal_vars)...,
                  get<PrimalFluxesVars>(primal_fluxes)...);
            },
            sources_args);
      }
      if (massive) {
        *operator_applied_to_vars /= get(det_inv_jacobian);
        ::dg::apply_mass_matrix(operator_applied_to_vars, mesh);
      }
    }

    // Add boundary corrections
    // Keeping track if any corrections were applied here, for an optimization
    // below
    bool has_any_boundary_corrections = false;
    Variables<tmpl::list<transform::Tags::TransformedFirstIndex<
        PrimalFluxesVars, Frame::ElementLogical>...>>
        lifted_logical_aux_boundary_corrections{num_points, 0.};
    for (auto& [mortar_id, mortar_data] : *all_mortar_data) {
      const auto& [direction, neighbor_id] = mortar_id;
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
      const auto [local_data, remote_data] = mortar_data.extract();
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
      auto auxiliary_boundary_corrections_on_mortar =
          Variables<tmpl::list<PrimalMortarFluxes...>>(
              local_data.field_data.template extract_subset<
                  tmpl::list<::Tags::NormalDotFlux<PrimalMortarFluxes>...>>());
      const auto add_remote_contribution = [](auto& lhs, const auto& rhs) {
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
      const auto auxiliary_boundary_corrections =
          Spectral::needs_projection(face_mesh, mortar_mesh, mortar_size)
              ? mass_conservative_restriction(
                    std::move(auxiliary_boundary_corrections_on_mortar),
                    mortar_mesh, mortar_size, mortar_jacobians.at(mortar_id),
                    face_mesh, face_jacobian)
              : std::move(auxiliary_boundary_corrections_on_mortar);

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
      //   n.H = avg(n.F) - n.F - penalty * n.F(n_j jump(u)).
      // Note that the "internal penalty" numerical flux
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
      const auto add_remote_jump_contribution = [](auto& lhs, const auto& rhs) {
        for (size_t i = 0; i < lhs.size(); ++i) {
          lhs[i] -= rhs[i];
        }
      };
      EXPAND_PACK_LEFT_TO_RIGHT(add_remote_jump_contribution(
          get<PrimalMortarVars>(primal_boundary_corrections_on_mortar),
          get<Tags::NormalDotFluxForJump<PrimalMortarVars>>(
              remote_data.field_data)));
      primal_boundary_corrections_on_mortar *= penalty;
      const auto add_remote_avg_contribution = [](auto& lhs, const auto& rhs) {
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

      // Project from the mortar back down to the face if needed, lift and add
      // to operator. See auxiliary boundary corrections above for details.
      auto primal_boundary_corrections =
          Spectral::needs_projection(face_mesh, mortar_mesh, mortar_size)
              ? mass_conservative_restriction(
                    std::move(primal_boundary_corrections_on_mortar),
                    mortar_mesh, mortar_size, mortar_jacobians.at(mortar_id),
                    face_mesh, face_jacobian)
              : std::move(primal_boundary_corrections_on_mortar);

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
      const double penalty_parameter, const bool massive,
      const ::dg::Formulation formulation,
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
    prepare_mortar_data<true>(
        make_not_null(&unused_deriv_vars_buffer),
        make_not_null(&primal_fluxes_buffer), make_not_null(&all_mortar_data),
        zero_primal_vars, element, mesh, inv_jacobian, face_normals,
        face_normal_vectors, face_normal_magnitudes, all_mortar_meshes,
        all_mortar_sizes, temporal_id, apply_boundary_condition, fluxes_args,
        fluxes_args_on_faces);
    apply_operator<true>(
        make_not_null(&operator_applied_to_zero_vars),
        make_not_null(&all_mortar_data), zero_primal_vars, primal_fluxes_buffer,
        element, mesh, inv_jacobian, det_inv_jacobian, face_normal_magnitudes,
        face_jacobians, face_jacobian_times_inv_jacobians, all_mortar_meshes,
        all_mortar_sizes, {}, penalty_parameter, massive, formulation,
        temporal_id, sources_args);
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
