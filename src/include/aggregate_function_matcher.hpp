#pragma once
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/optimizer/matcher/expression_matcher.hpp"

namespace duckdb {

class AggregateFunctionExpressionMatcher : public ExpressionMatcher {
public:
	AggregateFunctionExpressionMatcher()
	    : ExpressionMatcher(ExpressionClass::BOUND_AGGREGATE), policy(SetMatcher::Policy::INVALID) {
	}
	//! The matchers for the child expressions
	vector<unique_ptr<ExpressionMatcher>> matchers;
	//! The set matcher matching policy to use
	SetMatcher::Policy policy;
	//! The function name to match
	unique_ptr<FunctionMatcher> function;

	bool Match(Expression &expr_p, vector<reference<Expression>> &bindings) override;
};

inline bool AggregateFunctionExpressionMatcher::Match(Expression &expr_p, vector<reference<Expression>> &bindings) {
	if (!ExpressionMatcher::Match(expr_p, bindings)) {
		return false;
	}
	auto &expr = expr_p.Cast<BoundAggregateExpression>();
	if (!FunctionMatcher::Match(function, expr.Function().name)) {
		return false;
	}
	if (!SetMatcher::Match(matchers, expr.GetChildrenMutable(), bindings, policy)) {
		return false;
	}
	return true;
}

} // namespace duckdb
