#pragma once

#include "duckdb/common/helper.hpp"
#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/storage/storage_index.hpp"

namespace duckdb {

class Index;

// This is created by the optimizer rule
struct HNSWIndexScanBindData final : public TableScanBindData {
	explicit HNSWIndexScanBindData(TableCatalogEntry &table, Index &index, idx_t limit,
	                               unsafe_unique_array<float> query)
	    : TableScanBindData(table), index(index), limit(limit), query(std::move(query)) {
	}

	//! The index to use
	Index &index;

	//! The limit of the scan
	idx_t limit;

	//! The query vector
	unsafe_unique_array<float> query;

	//! Optional filter pushdown: set by the optimizer when static table_filters are present.
	//! When non-empty, a FilterBitmap is built at scan-init time and passed to filtered_ef_search.
	unique_ptr<TableFilterSet> pushed_filters;
	vector<idx_t>       filter_logical_col_ids; // logical col IDs (keys in pushed_filters->filters)
	vector<LogicalType> filter_col_types;

public:
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<HNSWIndexScanBindData>();
		return &other.table == &table;
	}
};

struct HNSWIndexScanFunction {
	static TableFunction GetFunction();
};

} // namespace duckdb
