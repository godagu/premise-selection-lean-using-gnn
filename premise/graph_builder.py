import re

import networkx as nx

from premise.graph_schema import PremiseEdgeType, PremiseNodeType
from premise.schema import ParsedPremise
from state.expression_parser import Expression, ExpressionParseError, parse_expression
from state.graph_builder import build_state_graph
from state.graph_shcema import NodeType
from state.schema import ContextDeclaration, ParsedState

# Lean auto-bound / `variable` names: a single letter, optionally with digits or primes (a, b, α, x1, x').
# These are used in statements without being declared in the code, e.g.  a ∨ b ↔ (¬a → b)
_AUTO_BOUND = re.compile(r"^[A-Za-z\u03b1-\u03c9\u0391-\u03a9\U0001D538-\U0001D56B][0-9']*$")  # incl. 𝕜 𝔸 ...
_NOT_AUTO_BOUND = {"π", "λ", "Π", "Σ"}
_UNKNOWN_TYPE = "_"  # placeholder type of binders without a type
# universe levels (Sort u, Type (max u v), Type _) are not variables of the statement
_UNIVERSE = re.compile(r"\b(Sort|Type)\s+(?:\([^()]*\)|[A-Za-z_][\w']*(?:\s*\+\s*\d+)?)")


def _identifiers(expression: Expression):
    if expression.kind == "identifier":
        yield expression.value
    for child in expression.children:
        yield from _identifiers(child)


def _find_auto_bound(parsed: ParsedPremise) -> list[str]:
    """Undeclared single-letter identifiers in binder types and the conclusion, in order of appearance."""
    declared = {name for binder in parsed.binders for name in binder.names}
    texts = [binder.type_expr for binder in parsed.binders] + [parsed.conclusion]
    found: list[str] = []
    for text in texts:
        if not text or text == _UNKNOWN_TYPE:
            continue
        try:
            names = list(_identifiers(parse_expression(_UNIVERSE.sub(r"\1", text))))
        except ExpressionParseError:
            continue
        for name in names:
            if name not in declared and name not in _NOT_AUTO_BOUND and _AUTO_BOUND.match(name) and name not in found:
                found.append(name)
    return found


def build_premise_graph(parsed: ParsedPremise, include_body: bool = False) -> nx.MultiDiGraph:
    """
    Builds the premise graph with the state graph builder: binders -> context, conclusion -> goal.
    The root keeps the id "state" (so state/visualize_state.py works unchanged) but has node_type PREMISE.
    """
    declarations: list[ContextDeclaration] = []
    binder_kinds: list[str] = []
    for binder in parsed.binders:
        declarations.append(ContextDeclaration(names=binder.names, type_expr=binder.type_expr or _UNKNOWN_TYPE))
        binder_kinds.append(binder.bracket)

    auto_bound = _find_auto_bound(parsed)
    for name in auto_bound:
        declarations.append(ContextDeclaration(names=[name], type_expr=_UNKNOWN_TYPE))
        binder_kinds.append("auto")

    graph = build_state_graph(ParsedState(declarations=declarations, goal=parsed.conclusion))

    root = graph.nodes["state"]
    root["node_type"] = PremiseNodeType.PREMISE.value
    root.update(
        full_name=parsed.full_name,
        name=parsed.name,
        premise_kind=parsed.kind,
        attrs=list(parsed.attrs),
        modifiers=list(parsed.modifiers),
        has_conclusion=bool(parsed.conclusion),
    )
    if not parsed.conclusion:  # structure / class / bare `def x := ...`: nothing to use as a goal
        graph.nodes["goal"]["expression_status"] = "absent"
        graph.nodes["goal"].pop("parse_error", None)
        graph.remove_node("goal_expr_raw")

    for declaration_index, kind in enumerate(binder_kinds):
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("declaration_index") == declaration_index and attrs.get("node_type") == NodeType.DECL.value:
                attrs["binder_kind"] = kind
                if kind == "auto":
                    attrs["auto_bound"] = True

    if include_body and parsed.body:
        graph.add_node("body", node_type=NodeType.RAW_EXPR.value, text=parsed.body)
        graph.add_edge("state", "body", relation=PremiseEdgeType.HAS_BODY.value)

    graph.graph.update(full_name=parsed.full_name, path=parsed.path, kind="premise")
    return graph