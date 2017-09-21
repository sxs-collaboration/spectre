// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>
#include <limits>

#include "Parallel/Info.hpp"
#include "Parallel/Printf.hpp"

#include "Parallel/ConstGlobalCache.decl.h"
#include "tests/Unit/Parallel/ParallelTestChares.decl.h"

template <typename Metavariables>
class TestChare : public CBase_TestChare<Metavariables> {
 public:
  explicit TestChare(Parallel::CProxy_ConstGlobalCache<Metavariables>
                         const_global_cache_proxy);

  explicit TestChare(CkMigrateMessage* /*msg*/) {}

  void set_id(int id) noexcept { id_ = id; }

  int my_id() const noexcept { return id_; }

 private:
  Parallel::CProxy_ConstGlobalCache<Metavariables> const_global_cache_proxy_;
  int id_{std::numeric_limits<int>::max()};
};

template <typename Metavariables>
class TestGroupChare : public CBase_TestGroupChare<Metavariables> {
 public:
  explicit TestGroupChare(Parallel::CProxy_ConstGlobalCache<Metavariables>
                              const_global_cache_proxy);

  explicit TestGroupChare(CkMigrateMessage* /*msg*/) {}

  int my_proc() const noexcept { return Parallel::my_proc(); }

 private:
  Parallel::CProxy_ConstGlobalCache<Metavariables> const_global_cache_proxy_;
};

template <typename Metavariables>
class TestNodeGroupChare : public CBase_TestNodeGroupChare<Metavariables> {
 public:
  explicit TestNodeGroupChare(Parallel::CProxy_ConstGlobalCache<Metavariables>
                                  const_global_cache_proxy);

  explicit TestNodeGroupChare(CkMigrateMessage* /*msg*/) {}

  int my_node() const noexcept { return Parallel::my_node(); }

 private:
  Parallel::CProxy_ConstGlobalCache<Metavariables> const_global_cache_proxy_;
};

template <typename Metavariables>
class TestArrayChare : public CBase_TestArrayChare<Metavariables> {
 public:
  explicit TestArrayChare(Parallel::CProxy_ConstGlobalCache<Metavariables>
                              const_global_cache_proxy);

  explicit TestArrayChare(CkMigrateMessage* /*msg*/) {}

  int my_index() const noexcept { return this->thisIndex; }

 private:
  Parallel::CProxy_ConstGlobalCache<Metavariables> const_global_cache_proxy_;
};

template <typename Metavariables>
class TestBoundArrayChare : public CBase_TestBoundArrayChare<Metavariables> {
 public:
  explicit TestBoundArrayChare(Parallel::CProxy_ConstGlobalCache<Metavariables>
                                   const_global_cache_proxy);

  explicit TestBoundArrayChare(CkMigrateMessage* /*msg*/) {}

  int my_index() const noexcept { return this->thisIndex; }

 private:
  Parallel::CProxy_ConstGlobalCache<Metavariables> const_global_cache_proxy_;
};

// ================================================================
// Template Definitions
// ================================================================

template <typename Metavariables>
TestChare<Metavariables>::TestChare(
    Parallel::CProxy_ConstGlobalCache<Metavariables> const_global_cache_proxy)
    : const_global_cache_proxy_(std::move(const_global_cache_proxy)) {
  Parallel::printf("Constructing TestChare on processor %i\n",
                   Parallel::my_proc());
}

template <typename Metavariables>
TestGroupChare<Metavariables>::TestGroupChare(
    Parallel::CProxy_ConstGlobalCache<Metavariables> const_global_cache_proxy)
    : const_global_cache_proxy_(std::move(const_global_cache_proxy)) {
  Parallel::printf("Constructing TestGroupChare on processor %i\n",
                   Parallel::my_proc());
}

template <typename Metavariables>
TestNodeGroupChare<Metavariables>::TestNodeGroupChare(
    Parallel::CProxy_ConstGlobalCache<Metavariables> const_global_cache_proxy)
    : const_global_cache_proxy_(std::move(const_global_cache_proxy)) {
  Parallel::printf("Constructing TestNodeGroupChare on processor %i\n",
                   Parallel::my_proc());
}

template <typename Metavariables>
TestArrayChare<Metavariables>::TestArrayChare(
    Parallel::CProxy_ConstGlobalCache<Metavariables> const_global_cache_proxy)
    : const_global_cache_proxy_(std::move(const_global_cache_proxy)) {
  Parallel::printf("Constructing TestArrayChare on processor %i\n",
                   Parallel::my_proc());
}

template <typename Metavariables>
TestBoundArrayChare<Metavariables>::TestBoundArrayChare(
    Parallel::CProxy_ConstGlobalCache<Metavariables> const_global_cache_proxy)
    : const_global_cache_proxy_(std::move(const_global_cache_proxy)) {
  Parallel::printf("Constructing TestBoundArrayChare on processor %i\n",
                   Parallel::my_proc());
}

#define CK_TEMPLATES_ONLY
#include "tests/Unit/Parallel/ParallelTestChares.def.h"
#undef CK_TEMPLATES_ONLY
