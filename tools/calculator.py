import ast
from langchain.tools import tool
import operator

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos
}

MAX_EXPR_LEN = 200
MAX_EXPONENT = 1000

def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if (isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and
    not isinstance(node.value, bool)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"Exponent cannot exceed {MAX_EXPONENT}")
        return OPERATORS[type(node.op)](left, right)
    raise ValueError(f"Unsupported syntax: {type(node).__name__}")

@tool
def calculate(expression: str) -> str:
    """Evaluate exact arithmetic expressions.

    Supports +, -, *, /, //, %, **, and parentheses.
    Always use this tool instead of doing mental math.

    Args:
        expression: A valid math string, e.g., '(1250 * 1.15) - 42**2'.
    """
    if len(expression) > MAX_EXPR_LEN:
        return f"Expression length exceeds {MAX_EXPR_LEN} characters"
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return (str(_eval_node(tree)))
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as exc:
        return f"Error: {exc}"