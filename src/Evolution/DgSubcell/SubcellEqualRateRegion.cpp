// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/SubcellEqualRateRegion.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <unordered_map>

#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/EqualRateRegions.tpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/NonconformingEqualRateRegions.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"

template <size_t VolumeDim>
class DomainCreator;
namespace evolution::dg::subcell {
class SubcellOptions;
}  // namespace evolution::dg::subcell

namespace evolution::dg::subcell {
template <size_t Dim>
SubcellEqualRateRegion<Dim>::SubcellEqualRateRegion(
    const SubcellOptions& subcell_options,
    const std::unique_ptr<DomainCreator<Dim>>& domain_creator) {
  // We need the version of the subcell options from the cache, but we
  // only have access to the parsed options here, so we have to
  // recreate it.
  const auto real_subcell_options =
      Tags::SubcellOptions<Dim>::create_from_options(subcell_options,
                                                     domain_creator);
  only_dg_block_ids_ = real_subcell_options.only_dg_block_ids();
}

template <size_t Dim>
std::unordered_map<std::string, size_t> SubcellEqualRateRegion<Dim>::regions()
    const {
  std::unordered_map<std::string, size_t> result{};
  result.emplace("Subcell", 0);
  return result;
}

template <size_t Dim>
bool SubcellEqualRateRegion<Dim>::is_in_region(
    const size_t region, const ElementId<Dim>& element_id) const {
  ASSERT(region == 0, "SubcellEqualRateRegion only defines one region");
  return not alg::found(only_dg_block_ids_, element_id.block_id());
}

template <size_t Dim>
void SubcellEqualRateRegion<Dim>::pup(PUP::er& p) {
  p | only_dg_block_ids_;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) template class SubcellEqualRateRegion<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace evolution::dg::subcell

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

// This doesn't scale, but as long as there are only two region
// generators we can just instantiate the combinations here.
#define INSTANTIATE(_, data)                                                  \
  template class evolution::dg::EqualRateRegions<                             \
      DIM(data),                                                              \
      tmpl::list<evolution::dg::subcell::SubcellEqualRateRegion<DIM(data)>>>; \
  template class evolution::dg::EqualRateRegions<                             \
      DIM(data),                                                              \
      tmpl::list<evolution::dg::NonconformingEqualRateRegions<DIM(data)>,     \
                 evolution::dg::subcell::SubcellEqualRateRegion<DIM(data)>>>; \
  template class evolution::dg::EqualRateRegions<                             \
      DIM(data),                                                              \
      tmpl::list<evolution::dg::subcell::SubcellEqualRateRegion<DIM(data)>,   \
                 evolution::dg::NonconformingEqualRateRegions<DIM(data)>>>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
