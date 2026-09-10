#pragma once

#include "duckdb/common/unique_ptr.hpp"
#include "hnsw/hnsw_index_scan.hpp"

namespace duckdb {

class LogicalOperator;

//! Extracts the constant values from the MARK join generated for a positive
//! constant IN predicate and returns the join's left-hand table scan.
unique_ptr<HNSWConstantInFilter> HNSWTryExtractConstantInFilter(unique_ptr<LogicalOperator> &root,
                                                                unique_ptr<LogicalOperator> *&get_ptr);

} // namespace duckdb
