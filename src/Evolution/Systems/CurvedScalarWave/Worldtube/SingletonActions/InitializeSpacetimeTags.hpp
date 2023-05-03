// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube::Initialization {

/*!
 * \brief Initializes the inverse spacetime metric and trace of the spacetime
 * christoffel symbol in the co-rotating grid frame at the position of the
 * scalar charge.
 *
 * \details This assumes a circular orbit so the spacetime quantities are
 * constant in the co-rotating grid frame due to the spherical symmetry of the
 * background spacetime. The mass of the central, non-spinning black hole is
 * hard-coded to be 1 and. For non-circular orbits, this should probably be a
 * compute tag that computes the background quantities at the current
 * coordinates of the scalar charge each time step.
 */
struct InitializeSpacetimeTags {
  static constexpr size_t Dim = 3;

  using compute_tags = tmpl::list<>;
  using simple_tags_from_options = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  using mutable_global_cache_tags = tmpl::list<>;
  using argument_tags = tmpl::list<Tags::ExcisionSphere<Dim>>;
  using simple_tags = tmpl::list<
      gr::Tags::InverseSpacetimeMetric<double, Dim, Frame::Grid>,
      gr::Tags::TraceSpacetimeChristoffelSecondKind<double, Dim, Frame::Grid>>;
  using return_tags = simple_tags;

  static void apply(const gsl::not_null<tnsr::AA<double, Dim, Frame::Grid>*>
                        inverse_spacetime_metric,
                    const gsl::not_null<tnsr::A<double, Dim, Frame::Grid>*>
                        trace_spacetime_christoffel,
                    const ExcisionSphere<Dim>& excision_sphere);
};
}  // namespace CurvedScalarWave::Worldtube::Initialization
