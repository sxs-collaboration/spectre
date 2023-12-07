// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "Evolution/DgSubcell/GetTciDecision.hpp"
#include "Evolution/DgSubcell/NeighborReconstructedFaceSolution.hpp"
#include "Evolution/DgSubcell/NeighborTciDecision.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Tags.hpp"
#include "Evolution/Executables/GrMhd/GhValenciaDivClean/GhValenciaDivCleanBase.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Tags.hpp"
#include "Utilities/TMPL.hpp"

template <bool UseControlSystems, typename... InterpolationTargetTags>
struct EvolutionMetavars
    : public GhValenciaDivCleanTemplateBase<
          EvolutionMetavars<UseControlSystems, InterpolationTargetTags...>,
          true, UseControlSystems> {
  using base = GhValenciaDivCleanTemplateBase<
      EvolutionMetavars<UseControlSystems, InterpolationTargetTags...>, true,
      UseControlSystems>;
  using const_global_cache_tags = typename base::const_global_cache_tags;
  using observed_reduction_data_tags =
      typename base::observed_reduction_data_tags;
  using component_list = typename base::component_list;
  using factory_creation = typename base::factory_creation;
  using registration = typename base::registration;

  static constexpr Options::String help{
      "Evolve the Valencia formulation of the GRMHD system with divergence "
      "cleaning, coupled to a dynamic spacetime evolved with the Generalized "
      "Harmonic formulation\n"};
};
