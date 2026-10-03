from enum import Enum
from typing import NamedTuple, List

class NodeType(str, Enum):
    # skeleton and context
    STATE = "STATE"
    CONTEXT = "CONTEXT"
    GOAL = "GOAL"
    DECL_VAR = "DECL_VAR"
    DECL_HYP = "DECL_HYP"
    DECL_INST = "DECL_INST"
    DECL_LET = "DECL_LET"

    # AST expression nodes
    OP_REL = "OP_REL" # relations (=, <...)
    OP_BIN = "OP_BIN" # binary operators (+, -, *, ...)
    APP = "APP" # function application
    BINDER = "BINDER" # quantifiers, lambda

    # AST leaves and types
    SYM = "SYM" # function symbols, constant
    VAR = "VAR" # variable
    LIT = "LIT" # literal (number, string, 0, 1 ...)
    TYPE_EXPR = "TYPE_EXPR" # types (Type u_1...)

class EdgeType(str, Enum):
    # Structural skeleton
    HAS_CTX = "has_ctx"
    HAS_GOAL = "has_goal"
    DECL = "decl"
    INST = "inst"
    LET_BIND = "let_bind"

    # universal AST edges
    HAS_EXPR = "has_expr"
    LHS = "lhs" # left argument
    RHS = "rhs" # right argument
    APP_FN = "app_fn" # function being applied
    APP_ARG = "app_arg" # argument being applied
    BINDER_BODY = "binder_body" # body of a binder
    BINDER_VAR = "binder_var" # variable bound by a binder


    # cross reference
    REF_DECL = "ref_decl" # reference to a declaration

    # type dependency
    HAS_TYPE = "has_type" # expression has a type

class EdgeRelation(NamedTuple):
    source: NodeType
    edge: EdgeType
    dst: NodeType

CANONICAL_EDGE_RELATIONS: List[EdgeRelation] = [
    # skeleton
    EdgeRelation(NodeType.STATE, EdgeType.HAS_CTX, NodeType.CONTEXT), # state -> context
    EdgeRelation(NodeType.STATE, EdgeType.HAS_GOAL, NodeType.GOAL), # state -> goal
    EdgeRelation(NodeType.CONTEXT, EdgeType.DECL, NodeType.DECL_VAR), # variable declaration (e.g., x : ℕ)
    EdgeRelation(NodeType.CONTEXT, EdgeType.DECL, NodeType.DECL_HYP), # hypothesis declaration (e.g., h : x > 0)
    EdgeRelation(NodeType.CONTEXT, EdgeType.INST, NodeType.DECL_INST), # typeclass instance declaration (e.g., inst✝ : Group G)
    EdgeRelation(NodeType.CONTEXT, EdgeType.LET_BIND, NodeType.DECL_LET), # local let-binding (e.g., y : ℕ := 5)

    # goal to AST root
    EdgeRelation(NodeType.GOAL, EdgeType.HAS_EXPR, NodeType.OP_REL), # goal is a relation (e.g., ⊢ a = b)
    EdgeRelation(NodeType.GOAL, EdgeType.HAS_EXPR, NodeType.OP_BIN), # goal is a binary operator (e.g., ⊢ a + b)
    EdgeRelation(NodeType.GOAL, EdgeType.HAS_EXPR, NodeType.APP), # goal is a function application (e.g., ⊢ P x)
    EdgeRelation(NodeType.GOAL, EdgeType.HAS_EXPR, NodeType.VAR), # goal is a single variable: ⊢ x

    # -----------------------------------------------------------------
    # 3. RELATIONAL OPERATORS (OP_REL: =, <, ≤, ↔, ∈)
    # -----------------------------------------------------------------
    # relational operators
    EdgeRelation(NodeType.OP_REL, EdgeType.LHS, NodeType.OP_BIN), # left side is a binary operator (e.g., (a + b) = c)
    EdgeRelation(NodeType.OP_REL, EdgeType.LHS, NodeType.APP), # left side is a function application (e.g., (card G) = n)
    EdgeRelation(NodeType.OP_REL, EdgeType.LHS, NodeType.VAR), # left side is a variable (e.g., x = y)
    EdgeRelation(NodeType.OP_REL, EdgeType.RHS, NodeType.OP_BIN), # right side is a binary operator (e.g., a = (b + c))
    EdgeRelation(NodeType.OP_REL, EdgeType.RHS, NodeType.APP), # right side is a function application (e.g., n = (card G))
    EdgeRelation(NodeType.OP_REL, EdgeType.RHS, NodeType.VAR), # right side is a variable (e.g., x = y)

    # binary algabraic operators
    EdgeRelation(NodeType.OP_BIN, EdgeType.LHS, NodeType.OP_BIN), # left side is a binary operator (e.g., (a + b) * c)
    EdgeRelation(NodeType.OP_BIN, EdgeType.LHS, NodeType.APP), # left side is a function application (e.g., (card G) * n)
    EdgeRelation(NodeType.OP_BIN, EdgeType.LHS, NodeType.VAR), # left side is a variable (e.g., x * y)
    EdgeRelation(NodeType.OP_BIN, EdgeType.RHS, NodeType.OP_BIN), # right side is a binary operator (e.g., a * (b + c))
    EdgeRelation(NodeType.OP_BIN, EdgeType.RHS, NodeType.APP), # right side is a function application (e.g., n * (card G))
    EdgeRelation(NodeType.OP_BIN, EdgeType.RHS, NodeType.VAR), # right side is a variable (e.g., x * y)

    # function application and binders
    EdgeRelation(NodeType.APP, EdgeType.APP_FN, NodeType.SYM), # application -> function symbol (e.g., card G -> card)
    EdgeRelation(NodeType.APP, EdgeType.APP_ARG, NodeType.VAR), # application -> argument (e.g., card G -> G)
    EdgeRelation(NodeType.APP, EdgeType.APP_ARG, NodeType.OP_BIN), # application -> argument is binary operator (e.g., f (a + b) -> (a + b))
    EdgeRelation(NodeType.APP, EdgeType.APP_ARG, NodeType.APP), # application -> argument is application (e.g., f (g x) -> (g x))
    EdgeRelation(NodeType.BINDER, EdgeType.BINDER_VAR, NodeType.VAR), # binder -> bound variable (e.g., ∀ x, P x -> x)
    EdgeRelation(NodeType.BINDER, EdgeType.BINDER_BODY, NodeType.OP_REL), # binder -> formula body (e.g., ∀ x, x = x -> (x = x))

    # cross references and types
    EdgeRelation(NodeType.VAR, EdgeType.REF_DECL, NodeType.DECL_VAR), # variable in goal -> variable in context (e.g., goal 'a' -> context 'a : G')
    EdgeRelation(NodeType.VAR, EdgeType.REF_DECL, NodeType.DECL_HYP), # variable in goal -> hypothesis in context (e.g., goal 'h' -> context 'h : x > 0')    
    EdgeRelation(NodeType.VAR, EdgeType.REF_DECL, NodeType.DECL_INST), # variable in goal -> instance in context (e.g., goal 'inst✝' -> context 'inst✝ : Group G')
    EdgeRelation(NodeType.DECL_VAR, EdgeType.HAS_TYPE, NodeType.TYPE_EXPR), # declaration -> type (e.g., (x : ℕ) -> ℕ)
    EdgeRelation(NodeType.DECL_HYP, EdgeType.HAS_TYPE, NodeType.TYPE_EXPR), # hypothesis -> type (e.g., (h : x > 0) -> (x > 0))
]