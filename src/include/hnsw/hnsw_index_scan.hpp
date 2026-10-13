#pragma once

#include "duckdb/common/column_index.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/types/value_map.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/storage/table/index_entry.hpp"

namespace duckdb {

struct HNSWConstantInFilter {
	HNSWConstantInFilter(ColumnIndex column_index_p, value_set_t values_p)
	    : column_index(std::move(column_index_p)), values(std::move(values_p)) {
	}

	//! The base-table column tested by the IN predicate.
	ColumnIndex column_index;
	//! The non-NULL constants from the IN predicate.
	value_set_t values;
};

// This is created by the optimizer rule
struct HNSWIndexScanBindData final : public TableScanBindData {
	explicit HNSWIndexScanBindData(TableCatalogEntry &table, shared_ptr<IndexEntry> index_entry,
	                               Identifier index_name_p, idx_t limit, unsafe_unique_array<float> query,
	                               bool prefilter_p, unique_ptr<HNSWConstantInFilter> constant_in_filter_p = nullptr)
	    : TableScanBindData(table), index_name(std::move(index_name_p)), index_entry(std::move(index_entry)),
	      limit(limit), query(std::move(query)), prefilter(prefilter_p),
	      constant_in_filter(std::move(constant_in_filter_p)) {
	}

	//! The index name used for display purposes
	Identifier index_name;

	//! The index to use
	shared_ptr<IndexEntry> index_entry;

	//! The limit of the scan
	idx_t limit;

	//! The query vector
	unsafe_unique_array<float> query;

	//! Whether static table filters should be evaluated before the HNSW search
	bool prefilter;

	//! A constant IN predicate that DuckDB lowered to a MARK join before extension optimization.
	unique_ptr<HNSWConstantInFilter> constant_in_filter;

public:
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<HNSWIndexScanBindData>();
		return &other.table == &table;
	}
};

struct HNSWIndexScanFunction {
	static TableFunction GetFunction();
	static bool PrefilterEnabled(ClientContext &context);
};

} // namespace duckdb
