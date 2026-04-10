#include "duckdb/optimizer/expression_heuristics.hpp"

#include "duckdb/planner/expression/list.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"
#include "duckdb/planner/filter/struct_filter.hpp"

#include <set>

namespace duckdb {

unique_ptr<LogicalOperator> ExpressionHeuristics::Rewrite(unique_ptr<LogicalOperator> op) {
	VisitOperator(*op);
	return op;
}

void ExpressionHeuristics::VisitOperator(LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_FILTER) {
		// reorder all filter expressions
		if (op.expressions.size() > 1) {
			ReorderExpressions(op.expressions);
		}
	}

	// traverse recursively through the operator tree
	VisitOperatorChildren(op);
	VisitOperatorExpressions(op);
}

unique_ptr<Expression> ExpressionHeuristics::VisitReplace(BoundConjunctionExpression &expr,
                                                          unique_ptr<Expression> *expr_ptr) {
	ReorderExpressions(expr.children);
	return nullptr;
}

void ExpressionHeuristics::ReorderExpressions(vector<unique_ptr<Expression>> &expressions) {
	struct ExpressionCosts {
		unique_ptr<Expression> expr;
		idx_t cost;
		idx_t original_index;

		bool operator==(const ExpressionCosts &p) const {
			return cost == p.cost;
		}
		bool operator<(const ExpressionCosts &p) const {
			if (cost != p.cost) {
				return cost < p.cost;
			}
			return original_index < p.original_index;
		}
	};

	idx_t num_expressions = expressions.size();

	// Build all ExpressionCosts upfront; non-throwing go into the candidate set,
	// throwing expressions go into a separate vector (preserving original order).
	// Expressions are moved into ExpressionCosts, no separate temp storage needed.
	std::set<ExpressionCosts> candidates;
	vector<ExpressionCosts> can_throw_exprs;
	for (idx_t i = 0; i < num_expressions; i++) {
		idx_t cost = Cost(*expressions[i]);
		bool can_throw = expressions[i]->CanThrow();
		ExpressionCosts entry {std::move(expressions[i]), cost, i};
		if (can_throw) {
			can_throw_exprs.push_back(std::move(entry));
		} else {
			candidates.insert(std::move(entry));
		}
	}

	if (can_throw_exprs.empty()) {
		// Fast path: no throwing expressions, extract in ascending cost order
		idx_t output_idx = 0;
		while (!candidates.empty()) {
			auto node = candidates.extract(candidates.begin());
			expressions[output_idx++] = std::move(node.value().expr);
		}
		return;
	}

	// is_placed[i]: whether the expression originally at position i has been written to the output
	vector<bool> is_placed(num_expressions, false);
	// all_placed_before: all expressions at original positions 0..all_placed_before-1 have been placed
	idx_t all_placed_before = 0;
	// next_throw_idx: index into can_throw_exprs[], the next throwing expression to consider
	idx_t next_throw_idx = 0;
	idx_t output_idx = 0;

	// Add the first throwing expression to candidates if it has no predecessors
	if (next_throw_idx < can_throw_exprs.size() &&
	    all_placed_before >= can_throw_exprs[next_throw_idx].original_index) {
		candidates.insert(std::move(can_throw_exprs[next_throw_idx]));
		next_throw_idx++;
	}

	while (!candidates.empty()) {
		// Extract the cheapest expression from candidates
		auto node = candidates.extract(candidates.begin());
		auto &cheapest = node.value();

		// Write it to the output
		idx_t orig_idx = cheapest.original_index;
		expressions[output_idx++] = std::move(cheapest.expr);
		is_placed[orig_idx] = true;

		// Advance all_placed_before past all consecutively placed positions
		while (all_placed_before < num_expressions && is_placed[all_placed_before]) {
			all_placed_before++;
		}

		// Add the next throwing expression if all its original predecessors have been placed
		if (next_throw_idx < can_throw_exprs.size() &&
		    all_placed_before >= can_throw_exprs[next_throw_idx].original_index) {
			candidates.insert(std::move(can_throw_exprs[next_throw_idx]));
			next_throw_idx++;
		}
	}
}

idx_t ExpressionHeuristics::ExpressionCost(BoundBetweenExpression &expr) {
	return Cost(*expr.input) + Cost(*expr.lower) + Cost(*expr.upper) + 10;
}

idx_t ExpressionHeuristics::ExpressionCost(BoundCaseExpression &expr) {
	// CASE WHEN check THEN result_if_true ELSE result_if_false END
	idx_t case_cost = 0;
	for (auto &case_check : expr.case_checks) {
		case_cost += Cost(*case_check.then_expr);
		case_cost += Cost(*case_check.when_expr);
	}
	case_cost += Cost(*expr.else_expr);
	return case_cost;
}

idx_t ExpressionHeuristics::ExpressionCost(BoundCastExpression &expr) {
	// OPERATOR_CAST
	// determine cast cost by comparing cast_expr.source_type and cast_expr_target_type
	idx_t cast_cost = 0;
	if (expr.return_type != expr.source_type()) {
		// if cast from or to varchar
		// TODO: we might want to add more cases
		if (expr.return_type.id() == LogicalTypeId::VARCHAR || expr.source_type().id() == LogicalTypeId::VARCHAR ||
		    expr.return_type.id() == LogicalTypeId::BLOB || expr.source_type().id() == LogicalTypeId::BLOB) {
			cast_cost = 200;
		} else {
			cast_cost = 5;
		}
	}
	return Cost(*expr.child) + cast_cost;
}

idx_t ExpressionHeuristics::ExpressionCost(BoundComparisonExpression &expr) {
	// COMPARE_EQUAL, COMPARE_NOTEQUAL, COMPARE_GREATERTHAN, COMPARE_GREATERTHANOREQUALTO, COMPARE_LESSTHAN,
	// COMPARE_LESSTHANOREQUALTO
	return Cost(*expr.left) + 5 + Cost(*expr.right);
}

idx_t ExpressionHeuristics::ExpressionCost(BoundConjunctionExpression &expr) {
	// CONJUNCTION_AND, CONJUNCTION_OR
	idx_t cost = 5;
	for (auto &child : expr.children) {
		cost += Cost(*child);
	}
	return cost;
}

idx_t ExpressionHeuristics::ExpressionCost(BoundFunctionExpression &expr) {
	unordered_map<std::string, idx_t> function_costs = {
	    {"+", 5},       {"-", 5},    {"&", 5},          {"#", 5},
	    {">>", 5},      {"<<", 5},   {"abs", 5},        {"*", 10},
	    {"%", 10},      {"/", 15},   {"date_part", 20}, {"year", 20},
	    {"round", 100}, {"~~", 200}, {"!~~", 200},      {"regexp_matches", 200},
	    {"||", 200}};

	idx_t cost_children = 0;
	for (auto &child : expr.children) {
		cost_children += Cost(*child);
	}

	auto cost_function = function_costs.find(expr.function.name);
	if (cost_function != function_costs.end()) {
		return cost_children + cost_function->second;
	} else {
		return cost_children + 1000;
	}
}

idx_t ExpressionHeuristics::ExpressionCost(BoundOperatorExpression &expr, ExpressionType expr_type) {
	idx_t sum = 0;
	for (auto &child : expr.children) {
		sum += Cost(*child);
	}

	// OPERATOR_IS_NULL, OPERATOR_IS_NOT_NULL
	if (expr_type == ExpressionType::OPERATOR_IS_NULL || expr_type == ExpressionType::OPERATOR_IS_NOT_NULL) {
		return sum + 5;
	} else if (expr_type == ExpressionType::COMPARE_IN || expr_type == ExpressionType::COMPARE_NOT_IN) {
		// COMPARE_IN, COMPARE_NOT_IN
		return sum + (expr.children.size() - 1) * 100;
	} else if (expr_type == ExpressionType::OPERATOR_NOT) {
		// OPERATOR_NOT
		return sum + 10; // TODO: evaluate via measured runtimes
	} else {
		return sum + 1000;
	}
}

idx_t ExpressionHeuristics::ExpressionCost(PhysicalType return_type, idx_t multiplier) {
	// TODO: ajust values according to benchmark results
	switch (return_type) {
	case PhysicalType::VARCHAR:
		return 5 * multiplier;
	case PhysicalType::FLOAT:
	case PhysicalType::DOUBLE:
		return 2 * multiplier;
	default:
		return 1 * multiplier;
	}
}

idx_t ExpressionHeuristics::Cost(Expression &expr) {
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::BOUND_CASE: {
		auto &case_expr = expr.Cast<BoundCaseExpression>();
		return ExpressionCost(case_expr);
	}
	case ExpressionClass::BOUND_BETWEEN: {
		auto &between_expr = expr.Cast<BoundBetweenExpression>();
		return ExpressionCost(between_expr);
	}
	case ExpressionClass::BOUND_CAST: {
		auto &cast_expr = expr.Cast<BoundCastExpression>();
		return ExpressionCost(cast_expr);
	}
	case ExpressionClass::BOUND_COMPARISON: {
		auto &comp_expr = expr.Cast<BoundComparisonExpression>();
		return ExpressionCost(comp_expr);
	}
	case ExpressionClass::BOUND_CONJUNCTION: {
		auto &conj_expr = expr.Cast<BoundConjunctionExpression>();
		return ExpressionCost(conj_expr);
	}
	case ExpressionClass::BOUND_FUNCTION: {
		auto &func_expr = expr.Cast<BoundFunctionExpression>();
		return ExpressionCost(func_expr);
	}
	case ExpressionClass::BOUND_OPERATOR: {
		auto &op_expr = expr.Cast<BoundOperatorExpression>();
		return ExpressionCost(op_expr, expr.GetExpressionType());
	}
	case ExpressionClass::BOUND_COLUMN_REF: {
		auto &col_expr = expr.Cast<BoundColumnRefExpression>();
		return ExpressionCost(col_expr.return_type.InternalType(), 8);
	}
	case ExpressionClass::BOUND_CONSTANT: {
		auto &const_expr = expr.Cast<BoundConstantExpression>();
		return ExpressionCost(const_expr.return_type.InternalType(), 1);
	}
	case ExpressionClass::BOUND_PARAMETER: {
		auto &const_expr = expr.Cast<BoundParameterExpression>();
		return ExpressionCost(const_expr.return_type.InternalType(), 1);
	}
	case ExpressionClass::BOUND_REF: {
		auto &col_expr = expr.Cast<BoundColumnRefExpression>();
		return ExpressionCost(col_expr.return_type.InternalType(), 8);
	}
	default: {
		break;
	}
	}

	// return a very high value if nothing matches
	return 1000;
}

idx_t ExpressionHeuristics::Cost(const TableFilter &filter) {
	switch (filter.filter_type) {
	case TableFilterType::DYNAMIC_FILTER:
	case TableFilterType::OPTIONAL_FILTER:
		return 0;
	case TableFilterType::CONJUNCTION_OR: {
		auto &conjunction_and = filter.Cast<ConjunctionOrFilter>();
		idx_t cost = 5;
		for (auto &child_filter : conjunction_and.child_filters) {
			cost += Cost(*child_filter);
		}
		return cost;
	}
	case TableFilterType::CONJUNCTION_AND: {
		auto &conjunction_and = filter.Cast<ConjunctionAndFilter>();
		idx_t cost = 5;
		for (auto &child_filter : conjunction_and.child_filters) {
			cost += Cost(*child_filter);
		}
		return cost;
	}
	case TableFilterType::CONSTANT_COMPARISON: {
		auto &constant_filter = filter.Cast<ConstantFilter>();
		return ExpressionCost(constant_filter.constant.type().InternalType(), 1);
	}
	case TableFilterType::IS_NULL:
	case TableFilterType::IS_NOT_NULL:
		return 5;
	case TableFilterType::STRUCT_EXTRACT: {
		auto &struct_filter = filter.Cast<StructFilter>();
		return Cost(*struct_filter.child_filter);
	}
	default:
		return 1000;
	}
}

static bool FilterCanThrow(const TableFilter &filter) {
	switch (filter.filter_type) {
	case TableFilterType::EXPRESSION_FILTER: {
		auto &expr_filter = filter.Cast<ExpressionFilter>();
		return expr_filter.expr->CanThrow();
	}
	case TableFilterType::CONJUNCTION_OR: {
		auto &conj = filter.Cast<ConjunctionOrFilter>();
		for (auto &child : conj.child_filters) {
			if (FilterCanThrow(*child)) {
				return true;
			}
		}
		return false;
	}
	case TableFilterType::CONJUNCTION_AND: {
		auto &conj = filter.Cast<ConjunctionAndFilter>();
		for (auto &child : conj.child_filters) {
			if (FilterCanThrow(*child)) {
				return true;
			}
		}
		return false;
	}
	case TableFilterType::STRUCT_EXTRACT: {
		auto &struct_filter = filter.Cast<StructFilter>();
		return FilterCanThrow(*struct_filter.child_filter);
	}
	case TableFilterType::OPTIONAL_FILTER: {
		auto &optional_filter = filter.Cast<OptionalFilter>();
		if (optional_filter.child_filter) {
			return FilterCanThrow(*optional_filter.child_filter);
		}
		return false;
	}
	default:
		return false;
	}
}

vector<idx_t> ExpressionHeuristics::GetInitialOrder(const TableFilterSet &table_filters) {
	struct FilterCost {
		idx_t index;
		idx_t cost;

		bool operator<(const FilterCost &other) const {
			if (cost != other.cost) {
				return cost < other.cost;
			}
			return index < other.index;
		}
	};

	idx_t num_filters = table_filters.filters.size();

	// Build FilterCost entries; non-throwing go into the candidate set,
	// throwing filters go into a separate vector (preserving original order).
	std::set<FilterCost> candidates;
	vector<FilterCost> can_throw_filters;
	idx_t filter_index = 0;
	for (auto &entry : table_filters.filters) {
		FilterCost fc {filter_index, Cost(*entry.second)};
		if (FilterCanThrow(*entry.second)) {
			can_throw_filters.push_back(fc);
		} else {
			candidates.insert(fc);
		}
		filter_index++;
	}

	if (can_throw_filters.empty()) {
		// Fast path: no throwing filters, extract in ascending cost order
		vector<idx_t> initial_permutation;
		for (auto &fc : candidates) {
			initial_permutation.push_back(fc.index);
		}
		return initial_permutation;
	}

	// is_placed[i]: whether the filter originally at position i has been placed
	vector<bool> is_placed(num_filters, false);
	// all_placed_before: all filters at original positions 0..all_placed_before-1 have been placed
	idx_t all_placed_before = 0;
	// next_throw_idx: index into can_throw_filters[], the next throwing filter to consider
	idx_t next_throw_idx = 0;
	vector<idx_t> initial_permutation;

	// Add the first throwing filter to candidates if it has no predecessors
	if (next_throw_idx < can_throw_filters.size() &&
	    all_placed_before >= can_throw_filters[next_throw_idx].index) {
		candidates.insert(can_throw_filters[next_throw_idx]);
		next_throw_idx++;
	}

	while (!candidates.empty()) {
		// Extract the cheapest filter from candidates
		auto it = candidates.begin();
		FilterCost cheapest = *it;
		candidates.erase(it);

		// Place it in the output
		initial_permutation.push_back(cheapest.index);
		is_placed[cheapest.index] = true;

		// Advance all_placed_before past all consecutively placed positions
		while (all_placed_before < num_filters && is_placed[all_placed_before]) {
			all_placed_before++;
		}

		// Add the next throwing filter if all its original predecessors have been placed
		if (next_throw_idx < can_throw_filters.size() &&
		    all_placed_before >= can_throw_filters[next_throw_idx].index) {
			candidates.insert(can_throw_filters[next_throw_idx]);
			next_throw_idx++;
		}
	}

	return initial_permutation;
}

} // namespace duckdb
