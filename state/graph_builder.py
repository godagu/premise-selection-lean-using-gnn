import networkx as nx
from state.schema import ParsedState
from state.graph_shcema import EdgeType, NodeType
from state.expression_parser import Expression, ExpressionParseError, parse_expression

# builds a multidirectional graph from parsed lean state
# starts with main skeleton of state, context and goal and then adds context and goal subtrees
def build_state_graph(parsed: ParsedState) -> nx.MultiDiGraph:
	# the skeleton of the graph: STATE, CONTEXT, GOAL and edges between them
    graph = nx.MultiDiGraph()
    
    graph.add_node("state", node_type=NodeType.STATE.value)
    graph.add_node("context", node_type=NodeType.CONTEXT.value)
    graph.add_node("goal", node_type=NodeType.GOAL.value, text=parsed.goal)
    
    graph.add_edge("state", "context", relation=EdgeType.HAS_CTX.value)
    graph.add_edge("state", "goal", relation=EdgeType.HAS_GOAL.value)

    declarations_by_name = {}

    # Each name gets its own declaration node; identical type text is shared.
    # Proposition declarations attach directly to their expression trees.
    for declaration_index, declaration in enumerate(parsed.declarations):
        for name_index, name in enumerate(declaration.names):
            declaration_id = f"declaration_{declaration_index}_{name_index}"
            graph.add_node(
                declaration_id,
                node_type=NodeType.DECL.value,
                name=name,
                declaration_index=declaration_index,
                name_index=name_index,
                type_text=declaration.type_expr,
                val_expr=declaration.val_expr,
            )
            graph.add_edge(
                "context",
                declaration_id,
                relation=EdgeType.DECL.value,
            )
            declarations_by_name[name] = declaration_id


    # analyze and attach type escpressions or propostions to declarations
    type_nodes_by_text = {}
    for declaration_index, declaration in enumerate(parsed.declarations):
        try:
            type_expression = _lower_expression(parse_expression(declaration.type_expr))
        except ExpressionParseError as error:
            type_expression = None
            parse_error = str(error)
        else:
            parse_error = None

        for name_index, name in enumerate(declaration.names):
            declaration_id = f"declaration_{declaration_index}_{name_index}"

            # if a type is proposition, unwrap it into an AST tree 
            if type_expression is not None and _is_proposition_expression(type_expression):
                expression_id = _add_expression_to_graph(
                    graph,
                    type_expression,
                    declarations_by_name,
                    node_prefix=f"context_expr_{declaration_index}_{name_index}",
                )
                graph.add_edge(declaration_id, expression_id, relation=EdgeType.HAS_EXPR.value)
                graph.nodes[declaration_id]["type_expression_status"] = "parsed"
                continue

            # else save it as text node
            type_id = type_nodes_by_text.get(declaration.type_expr)
            if type_id is None:
                type_id = f"type_expr_{len(type_nodes_by_text)}"
                type_nodes_by_text[declaration.type_expr] = type_id
                graph.add_node(
                    type_id,
                    node_type=NodeType.TYPE_EXPR.value,
                    text=declaration.type_expr,
                    expression_status="text_only" if parse_error else "opaque_type",
                )
                if parse_error:
                    graph.nodes[type_id]["parse_error"] = parse_error
            graph.add_edge(declaration_id, type_id, relation=EdgeType.HAS_TYPE.value)

    # analize and attach the goal
    try:
        goal_expression = _lower_expression(parse_expression(parsed.goal))
    except ExpressionParseError as error:
        # if parsing was not succesful, leave it as raw text
        graph.nodes["goal"]["expression_status"] = "text_only"
        graph.nodes["goal"]["parse_error"] = str(error)
        graph.add_node(
            "goal_expr_raw",
            node_type=NodeType.RAW_EXPR.value,
            text=parsed.goal,
        )
        graph.add_edge("goal", "goal_expr_raw", relation=EdgeType.HAS_EXPR.value)
    # if successfully parsed, recursively generate AST tree
    else:
        expression_id = _add_expression_to_graph(
            graph,
            goal_expression,
            declarations_by_name,
        )
        graph.add_edge(
            "goal",
            expression_id,
            relation=EdgeType.HAS_EXPR.value,
        )
        graph.nodes["goal"]["expression_status"] = "parsed"

    return graph

# node kinds produced by the parser that the graph schema has no dedicated node type for;
# they are rewritten into existing kinds (application of a symbol, binary/unary operator)
_STRUCTURAL_KINDS = {
    "tuple", "anonymous_constructor", "delimited", "list_literal", "curly_group", "structure_instance",
}

def _lower_expression(expression: Expression) -> Expression:
    children = tuple(_lower_expression(child) for child in expression.children)
    kind = expression.kind
    if kind in _STRUCTURAL_KINDS:
        # ⟨a, b⟩  ->  application(application(sym "⟨⟩", a), b)
        result = Expression("identifier", expression.value or kind)
        for child in children:
            result = Expression("application", "", (result, child))
        return result
    if kind == "type_ascription":
        return Expression("operator", ":", children)
    if kind in {"field_assign", "named_argument"}:
        return Expression("operator", ":=", (Expression("identifier", expression.value), *children))
    if kind == "as_pattern":
        return Expression("operator", "@", children)
    if kind == "binder_predicate":
        return Expression("unary", expression.value, children)
    return Expression(kind, expression.value, children)

# checks if expression is logical proposition
def _is_proposition_expression(expression: Expression) -> bool:
    if expression.kind == "relation":
        return True
    if expression.kind == "unary":
        return expression.value == "¬" and _is_proposition_expression(expression.children[0])
    if expression.kind == "binder":
        return _is_proposition_expression(expression.children[-1])
    if expression.kind == "operator" and expression.value in {"∧", "∨", "→"}:
        return all(_is_proposition_expression(child) for child in expression.children)
    return False

# recursive helper function for building graph
def _add_expression_to_graph(
    graph: nx.MultiDiGraph,
    expression: Expression,
    declarations_by_name: dict[str, str],
    path: tuple[int, ...] = (0,),
    node_prefix: str = "goal_expr",
    bound_variables: dict[str, str] | None = None,
) -> str:
    node_id = node_prefix + "_" + "_".join(map(str, path))

    if bound_variables is None:
        bound_variables = {}

    # parse bound variables and identifiers
    if expression.kind in {"bound_var", "bound_identifier"}:
        graph.add_node(node_id, node_type=NodeType.VAR.value, name=expression.value)
        if expression.kind == "bound_identifier" and expression.value in bound_variables:
            graph.add_edge(
                node_id,
                bound_variables[expression.value],
                relation=EdgeType.REF_BINDER.value,
            )
        return node_id

    # do bind sentence parsing
    if expression.kind == "do_bind":
        graph.add_node(node_id, node_type=NodeType.DO_BIND.value, name=expression.value)
        variable_id = node_prefix + "_" + "_".join(map(str, path + (0,)))
        graph.add_node(variable_id, node_type=NodeType.VAR.value, name=expression.value)
        value_id = _add_expression_to_graph(
            graph,
            expression.children[0],
            declarations_by_name,
            path + (1,),
            node_prefix,
            bound_variables,
        )
        graph.add_edge(node_id, variable_id, relation=EdgeType.DO_BIND_VAR.value)
        graph.add_edge(node_id, value_id, relation=EdgeType.DO_VALUE.value)
        return node_id

    # a simbple statement in do block
    if expression.kind == "do_expr":
        graph.add_node(node_id, node_type=NodeType.DO_EXPR.value)
        value_id = _add_expression_to_graph(
            graph,
            expression.children[0],
            declarations_by_name,
            path + (0,),
            node_prefix,
            bound_variables,
        )
        graph.add_edge(node_id, value_id, relation=EdgeType.DO_VALUE.value)
        return node_id
    
    # local let definition
    if expression.kind == "let":
        graph.add_node(node_id, node_type=NodeType.LET_EXPR.value, name=expression.value)
        variable_id = node_prefix + "_" + "_".join(map(str, path + (0,)))
        graph.add_node(variable_id, node_type=NodeType.VAR.value, name=expression.value)
        value_id = _add_expression_to_graph(
            graph,
            expression.children[0],
            declarations_by_name,
            path + (1,),
            node_prefix,
            bound_variables,
        )
        body_bound_variables = {**bound_variables, expression.value: variable_id}
        body_id = _add_expression_to_graph(
            graph,
            expression.children[1],
            declarations_by_name,
            path + (2,),
            node_prefix,
            body_bound_variables,
        )
        graph.add_edge(node_id, variable_id, relation=EdgeType.LET_VAR.value)
        graph.add_edge(node_id, value_id, relation=EdgeType.LET_VALUE.value)
        graph.add_edge(node_id, body_id, relation=EdgeType.LET_BODY.value)
        return node_id

    #  identifiers (variables or symbols)
    if expression.kind == "identifier":
        declaration_id = declarations_by_name.get(expression.value)
        node_type = NodeType.VAR if declaration_id is not None else NodeType.SYM
        graph.add_node(node_id, node_type=node_type.value, name=expression.value)
        if declaration_id is not None:
            graph.add_edge(
                node_id,
                declaration_id,
                relation=EdgeType.REF_DECL.value,
            )
        return node_id
    
    # handling of literals
    if expression.kind == "literal":
        graph.add_node(
            node_id,
            node_type=NodeType.LIT.value,
            value=expression.value,
        )
        return node_id

    if expression.kind == "application":
        graph.add_node(node_id, node_type=NodeType.APP.value)
        child_edge_types = (EdgeType.APP_FN, EdgeType.APP_ARG)
    elif expression.kind == "relation":
        graph.add_node(
            node_id,
            node_type=NodeType.OP_REL.value,
            operator=expression.value,
        )
        child_edge_types = (EdgeType.LHS, EdgeType.RHS)
    elif expression.kind == "operator":
        graph.add_node(
            node_id,
            node_type=NodeType.OP_BIN.value,
            operator=expression.value,
        )
        child_edge_types = (EdgeType.LHS, EdgeType.RHS)
    elif expression.kind == "unary":
        graph.add_node(
            node_id,
            node_type=NodeType.OP_UNARY.value,
            operator=expression.value,
        )
        child_edge_types = (EdgeType.OPERAND,)
    elif expression.kind == "conditional":
        graph.add_node(
            node_id,
            node_type=NodeType.IF.value,
            operator=expression.value,
        )
        child_edge_types = (EdgeType.IF_COND, EdgeType.IF_THEN, EdgeType.IF_ELSE)
    elif expression.kind == "projection":
        graph.add_node(
            node_id,
            node_type=NodeType.PROJECTION.value,
            name=expression.value,
        )
        child_edge_types = (EdgeType.PROJ_BASE,)
    elif expression.kind == "index":
        graph.add_node(
            node_id, 
            node_type=NodeType.INDEX.value, 
            operator="[]"
        )
        child_edge_types = (EdgeType.INDEX_BASE, EdgeType.INDEX_ARG)
    elif expression.kind == "match":
        graph.add_node(
            node_id, 
            node_type=NodeType.MATCH.value
        )
        child_edge_types = (EdgeType.MATCH_SCRUTINEE,) + (EdgeType.MATCH_CASE,) * (len(expression.children) - 1)
    elif expression.kind == "do":
        graph.add_node(
            node_id, 
            node_type=NodeType.DO.value
        )
        child_edge_types = tuple(EdgeType.DO_STMT for _ in expression.children)
    elif expression.kind == "binder":
        graph.add_node(
            node_id, 
            node_type=NodeType.BINDER.value, 
            operator=expression.value
        )
        if expression.value in {"fun_pattern", "match_case"}:
            child_edge_types = (EdgeType.BINDER_VAR,) * (len(expression.children) - 1) + (EdgeType.BINDER_BODY,)
        elif len(expression.children) == 3:
            child_edge_types = (EdgeType.BINDER_VAR, EdgeType.BINDER_TYPE, EdgeType.BINDER_BODY)
        else:
            child_edge_types = (EdgeType.BINDER_VAR, EdgeType.BINDER_BODY)
    else:
        raise ValueError(f"Unsupported expression node kind: {expression.kind!r}")

    # recursively add child nodes and join edges
    for child_index, (child, edge_type) in enumerate(zip(expression.children, child_edge_types)):
        child_bound_variables = bound_variables
        if expression.kind == "binder" and child_index == len(expression.children) - 1:
            variable_count = len(expression.children) - 1
            if expression.value not in {"fun_pattern", "match_case"}:
                variable_count = 1
            child_bound_variables = dict(bound_variables)
            for variable_index, binder_variable in enumerate(expression.children[:variable_count]):
                binder_variable_id = node_prefix + "_" + "_".join(
                    map(str, path + (variable_index,))
                )
                child_bound_variables[binder_variable.value] = binder_variable_id
        elif expression.kind == "do" and child_index > 0:
            for previous_index, previous_statement in enumerate(expression.children[:child_index]):
                if previous_statement.kind == "do_bind":
                    variable_id = node_prefix + "_" + "_".join(map(str, path + (previous_index, 0)))
                    child_bound_variables = {
                        **child_bound_variables,
                        previous_statement.value: variable_id,
                    }
        child_id = _add_expression_to_graph(
            graph,
            child,
            declarations_by_name,
            path + (child_index,),
            node_prefix,
            child_bound_variables,
        )
        graph.add_edge(node_id, child_id, relation=edge_type.value)

    return node_id