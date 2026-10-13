#pragma once

#include <cfloat>
#include <unordered_map>

#define USEARCH_USE_NUMKONG 0
#define USEARCH_USE_OPENMP  0

#include "usearch/index.hpp"
#include "usearch/index_dense.hpp"
#include "usearch/index_plugins.hpp"

#undef USEARCH_USE_NUMKONG
#undef USEARCH_USE_OPENMP
