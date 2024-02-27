// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube {

/*!
 * \brief Computes the geodesic acceleration of the particle, see
 * `Tags::GeodesicAccelerationCompute`. This mutator is run on the worldtube
 * singleton chare.
 */
struct UpdateAcceleration {
  static constexpr size_t Dim = 3;
  using variables_tag = ::Tags::Variables<
      tmpl::list<Tags::EvolvedPosition<Dim>, Tags::EvolvedVelocity<Dim>>>;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
  using return_tags = tmpl::list<dt_variables_tag>;
  using argument_tags = tmpl::list<Tags::ParticlePositionVelocity<Dim>,
                                   Tags::GeodesicAcceleration<Dim>>;
  static void apply(
      gsl::not_null<
          Variables<tmpl::list<::Tags::dt<Tags::EvolvedPosition<Dim>>,
                               ::Tags::dt<Tags::EvolvedVelocity<Dim>>>>*>
          dt_evolved_vars,
      const std::array<tnsr::I<double, Dim>, 2>& pos_vel,
      const tnsr::I<double, Dim, Frame::Inertial>& geodesic_acc);
};
}  // namespace CurvedScalarWave::Worldtube
