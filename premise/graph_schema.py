# The premise graph deliberately reuses the state graph vocabulary (state/graph_shcema.py):
#   root --has_ctx--> context --decl--> declaration nodes      (binders play the role of the context)
#   root --has_goal--> goal --has_expr--> conclusion AST        (conclusion plays the role of the goal)
# so one encoder can embed both. Only genuinely new types live here; the string values are what
# ends up in the graph, so they can be added to the same embedding tables as the state ones.
from enum import Enum


class PremiseNodeType(str, Enum):
    PREMISE = "PREMISE"  # replaces STATE as the graph root


class PremiseEdgeType(str, Enum):
    HAS_BODY = "has_body"  # root -> RAW_EXPR text node, only when include_body=True


# attribute values stored on declaration nodes, for the binder bracket kind
BINDER_KINDS = ("explicit", "implicit", "inst", "strict", "auto")