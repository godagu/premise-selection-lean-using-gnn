from enum import Enum
from typing import NamedTuple, List

class NodeType(str, Enum):
    # skeleton and context
    STATE = "STATE"
    CONTEXT = "CONTEXT"
    GOAL = "GOAL"
    DECL = "DECL"
    DECL_VAR = "DECL_VAR"
    DECL_HYP = "DECL_HYP"
    DECL_INST = "DECL_INST"
    DECL_LET = "DECL_LET"

    # AST expression nodes
    OP_REL = "OP_REL" # relations (=, <...)
    OP_BIN = "OP_BIN" # binary operators (+, -, *, ...)
    OP_UNARY = "OP_UNARY" # unary operators (¬, ↑, ↓, -)
    APP = "APP" # function application
    BINDER = "BINDER" # quantifiers, lambda
    IF = "IF" # conditional expression
    PROJECTION = "PROJECTION" # field projection such as x.fst
    INDEX = "INDEX" # indexing into a collection
    DO = "DO" # do-notation block
    DO_BIND = "DO_BIND" # monadic bind introduced by do-let
    DO_EXPR = "DO_EXPR" # expression statement in a do block
    LET_EXPR = "LET_EXPR" # local let binding with a body
    MATCH = "MATCH" # pattern match expression
    RAW_EXPR = "RAW_EXPR" # expression text preserved when parsing is unsupported

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
    BINDER_TYPE = "binder_type" # type annotation of a bound variable
    OPERAND = "operand" # operand of a unary operator
    IF_COND = "if_cond" # condition of an if expression
    IF_THEN = "if_then" # then branch of an if expression
    IF_ELSE = "if_else" # else branch of an if expression
    PROJ_BASE = "proj_base" # value projected from
    INDEX_BASE = "index_base" # collection being indexed
    INDEX_ARG = "index_arg" # index into the collection
    DO_STMT = "do_stmt" # ordered statement in a do block
    DO_BIND_VAR = "do_bind_var" # variable introduced by a do bind
    DO_VALUE = "do_value" # value or computation in a do statement
    LET_VAR = "let_var" # variable introduced by a local let
    LET_VALUE = "let_value" # assigned value in a local let
    LET_BODY = "let_body" # body scoped by a local let
    MATCH_SCRUTINEE = "match_scrutinee" # value being matched
    MATCH_CASE = "match_case" # pattern branch


    # cross reference
    REF_DECL = "ref_decl" # reference to a declaration
    REF_BINDER = "ref_binder" # reference to a bound variable

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
    EdgeRelation(NodeType.CONTEXT, EdgeType.DECL, NodeType.DECL), # unclassified local declaration
    EdgeRelation(NodeType.CONTEXT, EdgeType.DECL, NodeType.DECL_VAR), # variable declaration (e.g., x : ℕ)
    EdgeRelation(NodeType.CONTEXT, EdgeType.DECL, NodeType.DECL_HYP), # hypothesis declaration (e.g., h : x > 0)
    EdgeRelation(NodeType.CONTEXT, EdgeType.INST, NodeType.DECL_INST), # typeclass instance declaration (e.g., inst✝ : Group G)
    EdgeRelation(NodeType.CONTEXT, EdgeType.LET_BIND, NodeType.DECL_LET), # local let-binding (e.g., y : ℕ := 5)
    EdgeRelation(NodeType.DECL, EdgeType.HAS_EXPR, NodeType.OP_REL), # proposition declaration -> relation
    EdgeRelation(NodeType.DECL, EdgeType.HAS_EXPR, NodeType.OP_UNARY), # proposition declaration -> negation
    EdgeRelation(NodeType.DECL, EdgeType.HAS_EXPR, NodeType.OP_BIN), # proposition declaration -> logical operator
    EdgeRelation(NodeType.DECL, EdgeType.HAS_EXPR, NodeType.BINDER), # proposition declaration -> quantified proposition

    # goal to AST root
    EdgeRelation(NodeType.GOAL, EdgeType.HAS_EXPR, NodeType.OP_REL), # goal is a relation (e.g., ⊢ a = b)
    EdgeRelation(NodeType.GOAL, EdgeType.HAS_EXPR, NodeType.OP_BIN), # goal is a binary operator (e.g., ⊢ a + b)
    EdgeRelation(NodeType.GOAL, EdgeType.HAS_EXPR, NodeType.OP_UNARY), # goal is a unary expression
    EdgeRelation(NodeType.GOAL, EdgeType.HAS_EXPR, NodeType.APP), # goal is a function application (e.g., ⊢ P x)
    EdgeRelation(NodeType.GOAL, EdgeType.HAS_EXPR, NodeType.VAR), # goal is a single variable: ⊢ x
    EdgeRelation(NodeType.GOAL, EdgeType.HAS_EXPR, NodeType.BINDER), # goal is quantified
    EdgeRelation(NodeType.OP_REL, EdgeType.RHS, NodeType.DO), # equality or relation to a do block
    EdgeRelation(NodeType.GOAL, EdgeType.HAS_EXPR, NodeType.RAW_EXPR), # goal preserved as raw text when unsupported
        EdgeRelation(NodeType.TYPE_EXPR, EdgeType.HAS_EXPR, NodeType.OP_REL), # context type contains a relation
        EdgeRelation(NodeType.TYPE_EXPR, EdgeType.HAS_EXPR, NodeType.OP_BIN), # context type contains an operator
        EdgeRelation(NodeType.TYPE_EXPR, EdgeType.HAS_EXPR, NodeType.OP_UNARY), # context type contains a unary expression
        EdgeRelation(NodeType.TYPE_EXPR, EdgeType.HAS_EXPR, NodeType.BINDER), # context type contains a binder
    EdgeRelation(NodeType.IF, EdgeType.IF_COND, NodeType.OP_REL), # if condition is a relation
    EdgeRelation(NodeType.IF, EdgeType.IF_COND, NodeType.APP), # conditional test is an application
    EdgeRelation(NodeType.IF, EdgeType.IF_THEN, NodeType.SYM), # then branch is a symbol
    EdgeRelation(NodeType.IF, EdgeType.IF_ELSE, NodeType.SYM), # else branch is a symbol
    EdgeRelation(NodeType.IF, EdgeType.IF_THEN, NodeType.APP), # then branch is an application
    EdgeRelation(NodeType.IF, EdgeType.IF_THEN, NodeType.IF), # then branch is a nested conditional
    EdgeRelation(NodeType.IF, EdgeType.IF_ELSE, NodeType.APP), # else branch is an application
    EdgeRelation(NodeType.IF, EdgeType.IF_ELSE, NodeType.IF), # else branch is a nested conditional

    # -----------------------------------------------------------------
    # 3. RELATIONAL OPERATORS (OP_REL: =, <, ≤, ↔, ∈)
    # -----------------------------------------------------------------
    # relational operators
    EdgeRelation(NodeType.OP_REL, EdgeType.LHS, NodeType.OP_BIN), # left side is a binary operator (e.g., (a + b) = c)
    EdgeRelation(NodeType.OP_REL, EdgeType.LHS, NodeType.APP), # left side is a function application (e.g., (card G) = n)
    EdgeRelation(NodeType.OP_REL, EdgeType.LHS, NodeType.OP_REL), # left side is another relation (e.g., (a = b) ↔ (c = d))
    EdgeRelation(NodeType.OP_REL, EdgeType.LHS, NodeType.VAR), # left side is a variable (e.g., x = y)
    EdgeRelation(NodeType.OP_REL, EdgeType.LHS, NodeType.INDEX), # left side is an indexed collection value
    EdgeRelation(NodeType.OP_REL, EdgeType.LHS, NodeType.PROJECTION), # left side is a projected value
    EdgeRelation(NodeType.OP_REL, EdgeType.RHS, NodeType.OP_BIN), # right side is a binary operator (e.g., a = (b + c))
    EdgeRelation(NodeType.OP_REL, EdgeType.RHS, NodeType.APP), # right side is a function application (e.g., n = (card G))
    EdgeRelation(NodeType.OP_REL, EdgeType.RHS, NodeType.OP_REL), # right side is another relation
    EdgeRelation(NodeType.OP_REL, EdgeType.RHS, NodeType.VAR), # right side is a variable (e.g., x = y)
    EdgeRelation(NodeType.OP_REL, EdgeType.RHS, NodeType.SYM), # right side is a symbol
    EdgeRelation(NodeType.OP_REL, EdgeType.RHS, NodeType.BINDER), # right side is quantified

    # binary algabraic operators
    EdgeRelation(NodeType.OP_BIN, EdgeType.LHS, NodeType.OP_BIN), # left side is a binary operator (e.g., (a + b) * c)
    EdgeRelation(NodeType.OP_BIN, EdgeType.LHS, NodeType.APP), # left side is a function application (e.g., (card G) * n)
    EdgeRelation(NodeType.OP_BIN, EdgeType.LHS, NodeType.VAR), # left side is a variable (e.g., x * y)
    EdgeRelation(NodeType.OP_BIN, EdgeType.LHS, NodeType.SYM), # left side is a symbol
    EdgeRelation(NodeType.OP_BIN, EdgeType.LHS, NodeType.PROJECTION), # left side is a projected value
    EdgeRelation(NodeType.OP_BIN, EdgeType.LHS, NodeType.OP_REL), # left side is a relation in a logical connective
    EdgeRelation(NodeType.OP_BIN, EdgeType.RHS, NodeType.OP_BIN), # right side is a binary operator (e.g., a * (b + c))
    EdgeRelation(NodeType.OP_BIN, EdgeType.RHS, NodeType.APP), # right side is a function application (e.g., n * (card G))
    EdgeRelation(NodeType.OP_BIN, EdgeType.RHS, NodeType.VAR), # right side is a variable (e.g., x * y)
    EdgeRelation(NodeType.OP_BIN, EdgeType.RHS, NodeType.SYM), # right side is a symbol
    EdgeRelation(NodeType.OP_BIN, EdgeType.RHS, NodeType.LIT), # right side is a literal
    EdgeRelation(NodeType.OP_BIN, EdgeType.RHS, NodeType.PROJECTION), # right side is a projected value
    EdgeRelation(NodeType.OP_BIN, EdgeType.RHS, NodeType.OP_REL), # right side is a relation in a logical connective

    # function application and binders
    EdgeRelation(NodeType.APP, EdgeType.APP_FN, NodeType.SYM), # application -> function symbol (e.g., card G -> card)
    EdgeRelation(NodeType.APP, EdgeType.APP_FN, NodeType.BINDER), # application -> lambda function
    EdgeRelation(NodeType.APP, EdgeType.APP_FN, NodeType.APP), # application -> partially applied function
    EdgeRelation(NodeType.APP, EdgeType.APP_FN, NodeType.VAR), # application -> variable function
    EdgeRelation(NodeType.APP, EdgeType.APP_FN, NodeType.PROJECTION), # projection used as a function
    EdgeRelation(NodeType.APP, EdgeType.APP_ARG, NodeType.VAR), # application -> argument (e.g., card G -> G)
    EdgeRelation(NodeType.APP, EdgeType.APP_ARG, NodeType.SYM), # application -> named argument
    EdgeRelation(NodeType.APP, EdgeType.APP_ARG, NodeType.OP_BIN), # application -> argument is binary operator (e.g., f (a + b) -> (a + b))
    EdgeRelation(NodeType.APP, EdgeType.APP_ARG, NodeType.APP), # application -> argument is application (e.g., f (g x) -> (g x))
    EdgeRelation(NodeType.APP, EdgeType.APP_ARG, NodeType.BINDER), # application -> lambda argument
    EdgeRelation(NodeType.APP, EdgeType.APP_ARG, NodeType.PROJECTION), # application -> projected argument
    EdgeRelation(NodeType.OP_UNARY, EdgeType.OPERAND, NodeType.VAR), # unary operator -> variable operand
    EdgeRelation(NodeType.OP_UNARY, EdgeType.OPERAND, NodeType.SYM), # unary operator -> symbol operand
    EdgeRelation(NodeType.OP_UNARY, EdgeType.OPERAND, NodeType.LIT), # unary operator -> literal operand
    EdgeRelation(NodeType.OP_UNARY, EdgeType.OPERAND, NodeType.APP), # unary operator -> application operand
    EdgeRelation(NodeType.OP_UNARY, EdgeType.OPERAND, NodeType.OP_BIN), # unary operator -> binary expression operand
    EdgeRelation(NodeType.OP_UNARY, EdgeType.OPERAND, NodeType.OP_REL), # unary operator -> relation operand
    EdgeRelation(NodeType.OP_UNARY, EdgeType.OPERAND, NodeType.OP_UNARY), # nested unary operators
    EdgeRelation(NodeType.OP_UNARY, EdgeType.OPERAND, NodeType.BINDER), # negation of a quantified proposition
    EdgeRelation(NodeType.BINDER, EdgeType.BINDER_VAR, NodeType.VAR), # binder -> bound variable (e.g., ∀ x, P x -> x)
    EdgeRelation(NodeType.BINDER, EdgeType.BINDER_BODY, NodeType.OP_REL), # binder -> formula body (e.g., ∀ x, x = x -> (x = x))
    EdgeRelation(NodeType.BINDER, EdgeType.BINDER_BODY, NodeType.OP_BIN), # binder -> conjunction or disjunction
    EdgeRelation(NodeType.BINDER, EdgeType.BINDER_BODY, NodeType.OP_UNARY), # binder -> negated formula
    EdgeRelation(NodeType.BINDER, EdgeType.BINDER_BODY, NodeType.BINDER), # binder -> nested binder
    EdgeRelation(NodeType.BINDER, EdgeType.BINDER_BODY, NodeType.APP), # binder -> predicate application or lambda body
    EdgeRelation(NodeType.BINDER, EdgeType.BINDER_TYPE, NodeType.SYM), # binder variable type is a symbol
    EdgeRelation(NodeType.BINDER, EdgeType.BINDER_TYPE, NodeType.VAR), # binder variable type is a variable
    EdgeRelation(NodeType.BINDER, EdgeType.BINDER_BODY, NodeType.LET_EXPR), # function clause body with local let
    EdgeRelation(NodeType.BINDER, EdgeType.BINDER_BODY, NodeType.IF), # function clause body with conditionals
    EdgeRelation(NodeType.IF, EdgeType.IF_ELSE, NodeType.MATCH), # conditional else branch is a match
    EdgeRelation(NodeType.MATCH, EdgeType.MATCH_SCRUTINEE, NodeType.APP), # match on a function application
    EdgeRelation(NodeType.MATCH, EdgeType.MATCH_SCRUTINEE, NodeType.VAR), # match on a variable
    EdgeRelation(NodeType.MATCH, EdgeType.MATCH_CASE, NodeType.BINDER), # match branch with a bound pattern
    EdgeRelation(NodeType.LET_EXPR, EdgeType.LET_VAR, NodeType.VAR), # local let -> introduced variable
    EdgeRelation(NodeType.LET_EXPR, EdgeType.LET_VALUE, NodeType.OP_BIN), # local let -> computed value
    EdgeRelation(NodeType.LET_EXPR, EdgeType.LET_BODY, NodeType.IF), # local let -> conditional body
    EdgeRelation(NodeType.DO, EdgeType.DO_STMT, NodeType.DO_BIND), # do block -> bind statement
    EdgeRelation(NodeType.DO, EdgeType.DO_STMT, NodeType.DO_EXPR), # do block -> expression statement
    EdgeRelation(NodeType.DO_BIND, EdgeType.DO_BIND_VAR, NodeType.VAR), # bind statement -> introduced variable
    EdgeRelation(NodeType.DO_BIND, EdgeType.DO_VALUE, NodeType.APP), # bind statement -> computation
    EdgeRelation(NodeType.DO_BIND, EdgeType.DO_VALUE, NodeType.VAR), # bind statement -> variable computation
    EdgeRelation(NodeType.DO_BIND, EdgeType.DO_VALUE, NodeType.SYM), # bind statement -> symbol computation
    EdgeRelation(NodeType.DO_EXPR, EdgeType.DO_VALUE, NodeType.APP), # expression statement -> application
    EdgeRelation(NodeType.DO_EXPR, EdgeType.DO_VALUE, NodeType.VAR), # expression statement -> variable

    # cross references and types
    EdgeRelation(NodeType.VAR, EdgeType.REF_BINDER, NodeType.VAR), # occurrence -> binder variable
    EdgeRelation(NodeType.PROJECTION, EdgeType.PROJ_BASE, NodeType.VAR), # field projection from a variable
    EdgeRelation(NodeType.PROJECTION, EdgeType.PROJ_BASE, NodeType.SYM), # field projection from an unresolved receiver
    EdgeRelation(NodeType.PROJECTION, EdgeType.PROJ_BASE, NodeType.APP), # field projection from an application result
    EdgeRelation(NodeType.PROJECTION, EdgeType.PROJ_BASE, NodeType.BINDER), # field projection from a function clause
    EdgeRelation(NodeType.PROJECTION, EdgeType.PROJ_BASE, NodeType.IF), # field projection from a conditional result
    EdgeRelation(NodeType.PROJECTION, EdgeType.PROJ_BASE, NodeType.PROJECTION), # chained projection
    EdgeRelation(NodeType.INDEX, EdgeType.INDEX_BASE, NodeType.LIT), # index into an empty-list literal
    EdgeRelation(NodeType.INDEX, EdgeType.INDEX_BASE, NodeType.VAR), # index into a variable collection
    EdgeRelation(NodeType.INDEX, EdgeType.INDEX_BASE, NodeType.APP), # index into an applied collection
    EdgeRelation(NodeType.INDEX, EdgeType.INDEX_BASE, NodeType.PROJECTION), # index into a projected collection
    EdgeRelation(NodeType.INDEX, EdgeType.INDEX_ARG, NodeType.VAR), # index is a variable
    EdgeRelation(NodeType.INDEX, EdgeType.INDEX_ARG, NodeType.LIT), # index is a literal
    EdgeRelation(NodeType.INDEX, EdgeType.INDEX_ARG, NodeType.OP_BIN), # index is an arithmetic expression
    EdgeRelation(NodeType.VAR, EdgeType.REF_DECL, NodeType.DECL), # variable in an expression -> unclassified context declaration
    EdgeRelation(NodeType.VAR, EdgeType.REF_DECL, NodeType.DECL_VAR), # variable in goal -> variable in context (e.g., goal 'a' -> context 'a : G')
    EdgeRelation(NodeType.VAR, EdgeType.REF_DECL, NodeType.DECL_HYP), # variable in goal -> hypothesis in context (e.g., goal 'h' -> context 'h : x > 0')    
    EdgeRelation(NodeType.VAR, EdgeType.REF_DECL, NodeType.DECL_INST), # variable in goal -> instance in context (e.g., goal 'inst✝' -> context 'inst✝ : Group G')
    EdgeRelation(NodeType.DECL_VAR, EdgeType.HAS_TYPE, NodeType.TYPE_EXPR), # declaration -> type (e.g., (x : ℕ) -> ℕ)
    EdgeRelation(NodeType.DECL_HYP, EdgeType.HAS_TYPE, NodeType.TYPE_EXPR), # hypothesis -> type (e.g., (h : x > 0) -> (x > 0))
    EdgeRelation(NodeType.DECL, EdgeType.HAS_TYPE, NodeType.TYPE_EXPR), # unclassified declaration -> type text
]