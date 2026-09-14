#pragma once

#include "duckdb/common/helper.hpp"
#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/storage/table/index_entry.hpp"

namespace duckdb {

// This is created by the optimizer rule
struct HNSWIndexScanBindData final : public TableScanBindData {
	explicit HNSWIndexScanBindData(TableCatalogEntry &table, shared_ptr<IndexEntry> index_entry,
	                               Identifier index_name_p, idx_t limit, unsafe_unique_array<float> query,
	                               bool prefilter_p)
	    : TableScanBindData(table), index_name(std::move(index_name_p)), index_entry(std::move(index_entry)),
	      limit(limit), query(std::move(query)), prefilter(prefilter_p) {
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
