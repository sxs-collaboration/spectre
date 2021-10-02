// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution {
namespace dg {
/// \ingroup ConservativeGroup
/// \brief Calculate \f$\partial u/\partial t\f$ for a conservative system.
///
/// The time evolution of the variables \f$u\f$ of any conservative
/// system is given by \f$\partial_t u = - \partial_i F^i + S\f$,
/// where \f$F^i\f$ are the fluxes and \f$S\f$ are the sources.
///
/// Source terms are only added for variables in the
/// `System::sourced_variables` type list.
template <typename System, ::dg::Formulation DgFormulation>
struct ConservativeDuDt {
  static constexpr size_t volume_dim = System::volume_dim;
  using frame = Frame::Inertial;

  template <template <class> class StepPrefix>
  using return_tags = tmpl::list<
      db::add_tag_prefix<StepPrefix, typename System::variables_tag>>;

  using argument_tags = tmpl::list<
      domain::Tags::Mesh<volume_dim>,
      domain::Tags::InverseJacobian<volume_dim, Frame::ElementLogical,
                                    Frame::Inertial>,
      db::add_tag_prefix<::Tags::Flux, typename System::variables_tag,
                         tmpl::size_t<volume_dim>, frame>,
      db::add_tag_prefix<::Tags::Source, typename System::variables_tag>>;

  template <template <class> class StepPrefix, typename... VarsTags, size_t Dim>
  static void apply(
      const gsl::not_null<Variables<tmpl::list<StepPrefix<VarsTags>...>>*>
          dt_vars,
      const Mesh<Dim> mesh,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>& inverse_jacobian,
      const Variables<tmpl::list<::Tags::Flux<VarsTags, tmpl::size_t<Dim>,
                                              Frame::Inertial>...>>& fluxes,
      const Variables<tmpl::list<::Tags::Source<VarsTags>...>>& sources) {
    static_assert(DgFormulation == ::dg::Formulation::StrongInertial,
                  "Curently only support StrongInertial DG formulation in "
                  "ConservativeDuDt.");

    const Variables<tmpl::list<::Tags::div<
        ::Tags::Flux<VarsTags, tmpl::size_t<Dim>, Frame::Inertial>>...>>
        div_fluxes = divergence(fluxes, mesh, inverse_jacobian);

    ASSERT(dt_vars->size() == div_fluxes.size(),
           "The time derivative and the flux divergence have different sizes. "
           "Time derivative: "
               << dt_vars->size() << " div fluxes: " << div_fluxes.size() << " "
               << dt_vars->number_of_grid_points() << " "
               << div_fluxes.number_of_grid_points());
    *dt_vars = -div_fluxes;
    tmpl::for_each<typename System::sourced_variables>(
        [&dt_vars, &sources](auto tag_v) {
          using tag = typename decltype(tag_v)::type;
          const auto& source = get<::Tags::Source<tag>>(sources);
          auto& dt_var = get<StepPrefix<tag>>(*dt_vars);
          ASSERT(source.size() == dt_var.size(),
                 "The source and the time derivative variable must have the "
                 "same number of independent components. Time derivative: "
                     << dt_var.size() << " source: " << source.size());
          for (size_t i = 0; i < source.size(); ++i) {
            dt_var[i] += source[i];
          }
        });
  }
};
}  // namespace dg
}  // namespace evolution
