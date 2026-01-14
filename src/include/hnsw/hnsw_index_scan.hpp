#pragma once

#include "duckdb/function/table_function.hpp"
#include "duckdb/function/table/table_scan.hpp"

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
