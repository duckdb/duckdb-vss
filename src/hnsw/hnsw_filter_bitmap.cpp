#include "hnsw/hnsw_filter_bitmap.hpp"

#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/allocator.hpp"
#include "duckdb/common/constants.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/storage_index.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/transaction/local_storage.hpp"

namespace duckdb {

//! Marks row_ids from a scanned chunk that pass the given ExpressionExecutor filter into the bitmap.
//! sel is a scratch buffer; chunk has filter columns at positions 0..n-1, row_id at position n.
static void MarkMatchingRows(ExpressionExecutor &executor, DataChunk &chunk, SelectionVector &sel,
                             idx_t filter_col_count, FilterBitmap &bitmap) {
	// The executor expression references columns 0..filter_col_count-1.
	// We pass the whole chunk; the row_id at position filter_col_count is ignored.
	idx_t valid_count = executor.SelectExpression(chunk, sel);

	auto &row_id_vec = chunk.data[filter_col_count]; // last column = row_id
	// DuckDB may return row_ids as a SequenceVector (lazy base+increment representation).
	// Flatten materialises the values into a FlatVector so GetData returns correct row IDs.
	row_id_vec.Flatten(chunk.size());
	auto row_ids_ptr = FlatVector::GetData<row_t>(row_id_vec);
	for (idx_t i = 0; i < valid_count; i++) {
		bitmap.Set(row_ids_ptr[sel.get_index(i)]);
	}
}

unique_ptr<FilterBitmap> BuildFilterBitmap(ClientContext &context, TableCatalogEntry &table,
                                           const TableFilterSet &filters,
                                           const vector<idx_t> &logical_col_ids,
                                           const vector<LogicalType> &col_types) {
	D_ASSERT(!filters.filters.empty());
	D_ASSERT(logical_col_ids.size() == col_types.size());

	auto &storage = table.GetStorage();
	auto &transaction = DuckTransaction::Get(context, table.catalog);

	// Pre-allocate bitmap to the current total committed row count (safe upper bound for main storage).
	// Local storage row_ids can exceed this; FilterBitmap::Set grows the vector on demand.
	auto bitmap = make_uniq<FilterBitmap>();
	const idx_t total_rows = storage.GetTotalRows();
	bitmap->bits.assign(total_rows, 0);

	// Build scan column IDs: [filter_col_0, filter_col_1, ..., row_id]
	vector<StorageIndex> scan_col_ids;
	scan_col_ids.reserve(logical_col_ids.size() + 1);
	for (idx_t i = 0; i < logical_col_ids.size(); i++) {
		auto storage_oid = table.GetColumn(LogicalIndex(logical_col_ids[i])).StorageOid();
		scan_col_ids.emplace_back(storage_oid);
	}
	scan_col_ids.emplace_back(DConstants::INVALID_INDEX); // row_id column marker

	// Build scan types: [filter_col_types..., ROW_TYPE]
	vector<LogicalType> scan_types = col_types;
	scan_types.push_back(LogicalType::ROW_TYPE);
	const idx_t filter_col_count = col_types.size();

	// Build filter expressions, each bound to its chunk-position index (0, 1, ...) rather
	// than the original logical column ID.  This is critical: after scanning, chunk columns
	// are laid out at positions 0..n-1, independent of the source table's column numbering.
	vector<unique_ptr<Expression>> filter_exprs;
	filter_exprs.reserve(filters.filters.size());
	for (idx_t i = 0; i < logical_col_ids.size(); i++) {
		idx_t orig_col_id = logical_col_ids[i];
		auto it = filters.filters.find(orig_col_id);
		if (it == filters.filters.end()) {
			continue;
		}
		auto col_ref = make_uniq<BoundReferenceExpression>(col_types[i], i);
		filter_exprs.push_back(it->second->ToExpression(*col_ref));
	}

	if (filter_exprs.empty()) {
		// No evaluable filters (shouldn't normally happen given the caller's guard)
		return bitmap;
	}

	// Combine multiple filters into a single AND expression so we can use SelectExpression
	unique_ptr<Expression> combined_expr;
	if (filter_exprs.size() == 1) {
		combined_expr = std::move(filter_exprs[0]);
	} else {
		auto conjunction = make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
		for (auto &expr : filter_exprs) {
			conjunction->children.push_back(std::move(expr));
		}
		combined_expr = std::move(conjunction);
	}

	ExpressionExecutor executor(context, *combined_expr);
	SelectionVector sel(STANDARD_VECTOR_SIZE);
	DataChunk chunk;
	chunk.Initialize(Allocator::Get(context), scan_types);

	// Build a POSITION-remapped TableFilterSet for zone-map pruning.
	// ScanFilterInfo::Initialize treats filter keys as indices into scan_col_ids, NOT logical col IDs.
	// We scan with [label_storage_col, row_id], so label is at scan position 0, not logical position 1.
	TableFilterSet remapped_filters;
	for (idx_t i = 0; i < logical_col_ids.size(); i++) {
		idx_t orig_col_id = logical_col_ids[i];
		auto it = filters.filters.find(orig_col_id);
		if (it != filters.filters.end()) {
			remapped_filters.filters.emplace(i, it->second->Copy());
		}
	}

	// --- Scan main (committed) storage ---
	// Passing the remapped TableFilterSet enables zone-map pruning (row-group skipping).
	// Row-level filtering is applied via ExpressionExecutor above.
	{
		TableScanState scan_state;
		storage.InitializeScan(context, transaction, scan_state, scan_col_ids,
		                       optional_ptr<TableFilterSet>(&remapped_filters));

		while (true) {
			chunk.Reset();
			storage.Scan(transaction, chunk, scan_state);
			if (chunk.size() == 0) {
				break;
			}
			MarkMatchingRows(executor, chunk, sel, filter_col_count, *bitmap);
		}
	}

	// --- Scan local (transaction-local / uncommitted) storage ---
	// Rows inserted within the current transaction are in local storage and may also
	// have been added to the HNSW index.  They must be included in the bitmap so that
	// filtered_ef_search does not exclude them.
	// FilterBitmap::Set grows the vector on demand for row_ids beyond the pre-allocated size.
	{
		auto &local_storage = LocalStorage::Get(context, table.catalog);
		if (local_storage.Find(storage)) {
			// TableScanState acts as the required parent for local_state (CollectionScanState).
			TableScanState local_ts;
			local_storage.InitializeScan(storage, local_ts.local_state,
			                             optional_ptr<TableFilterSet>(&remapped_filters));

			DataChunk local_chunk;
			local_chunk.Initialize(Allocator::Get(context), scan_types);

			while (true) {
				local_chunk.Reset();
				local_storage.Scan(local_ts.local_state, scan_col_ids, local_chunk);
				if (local_chunk.size() == 0) {
					break;
				}
				MarkMatchingRows(executor, local_chunk, sel, filter_col_count, *bitmap);
			}
		}
	}

	return bitmap;
}

} // namespace duckdb
