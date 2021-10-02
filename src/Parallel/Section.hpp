// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>
#include <optional>
#include <pup.h>

#include "Parallel/ParallelComponentHelpers.hpp"

namespace Parallel {

/*!
 * \brief A subset of chares in a parallel component
 *
 * The section is identified at compile time by the parallel component and a
 * `SectionIdTag`. The `SectionIdTag` describes the quantity that partitions the
 * chares into one or more sections. For example, the `SectionIdTag` could be
 * the block ID in the computational domain, so elements are partitioned per
 * block. Chares can be a member of multiple sections.
 *
 * - [Details on sections in the Charm++
 *   documentation](https://charm.readthedocs.io/en/latest/charm++/manual.html#sections-subsets-of-a-chare-array-group)
 *
 * Here's an example how to work with sections in an array parallel component:
 *
 * \snippet Test_SectionReductions.cpp sections_example
 *
 * \warning The Charm++ documentation indicates some [creation order
 * restrictions](https://charm.readthedocs.io/en/latest/charm++/manual.html#creation-order-restrictions)
 * for sections that may become relevant if we encounter issues with race
 * conditions in the future.
 */
template <typename ParallelComponent, typename SectionIdTag>
struct Section {
 private:
  using chare_type = typename ParallelComponent::chare_type;
  using charm_type = charm_types_with_parameters<
      ParallelComponent,
      typename get_array_index<chare_type>::template f<ParallelComponent>>;
  using IdType = typename SectionIdTag::type;

 public:
  using parallel_component = ParallelComponent;
  using cproxy_section = typename charm_type::cproxy_section;
  using section_id_tag = SectionIdTag;

  Section(IdType id, cproxy_section proxy)
      : id_(id), proxy_(std::move(proxy)), cookie_(proxy_.ckGetSectionInfo()) {}

  Section() = default;
  Section(Section&& rhs) = default;
  Section& operator=(Section&& rhs) = default;
  // The copy constructors currently copy the cookie as well. This seems to work
  // fine, but if issues come up with updating the cookie we can consider
  // re-creating it when copying the section to each element.
  Section(const Section&) = default;
  Section& operator=(const Section&) = default;
  ~Section() = default;

  /// The section ID corresponding to the `SectionIdTag`
  const IdType& id() const { return id_; }

  /// @{
  /// The Charm++ section proxy
  const cproxy_section& proxy() const { return proxy_; }
  cproxy_section& proxy() { return proxy_; }
  /// @}

  /*!
   * \brief The Charm++ section cookie that keeps track of reductions
   *
   * The section cookie must be stored on each element and updated when
   * performing reductions. For details on Charm++ sections and section
   * reductions see:
   * https://charm.readthedocs.io/en/latest/charm++/manual.html?#sections-subsets-of-a-chare-array-group
   */
  /// @{
  const CkSectionInfo& cookie() const { return cookie_; }
  CkSectionInfo& cookie() { return cookie_; }
  /// @}

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) {
    p | id_;
    p | proxy_;
    p | cookie_;
  }

 private:
  IdType id_{};
  cproxy_section proxy_{};
  CkSectionInfo cookie_{};
};

}  // namespace Parallel
