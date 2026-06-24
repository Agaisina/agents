import ast
import re
import json
import operator as _op
import requests
import logging

from langchain_core.tools import tool
from src.agents.domain.models import CalculatorInput, CurrencyInput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe arithmetic evaluator — replaces eval() (OWASP A03 Code Injection fix)
# ---------------------------------------------------------------------------
_BINARY_OPS: dict = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
}
_UNARY_OPS: dict = {
    ast.UAdd: _op.pos,
    ast.USub: _op.neg,
}


def _eval_ast_node(node: ast.expr) -> float:
    """Recursively evaluate a numeric AST node (constants, binary ops, unary ops)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        return _BINARY_OPS[type(node.op)](
            _eval_ast_node(node.left), _eval_ast_node(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_ast_node(node.operand))
    raise ValueError(f"Unsupported operation: {ast.dump(node)}")


def _safe_eval_expression(expression: str) -> float:
    """Parse and evaluate a numeric arithmetic expression without using eval()."""
    tree = ast.parse(expression, mode="eval")
    return _eval_ast_node(tree.body)


@tool("safe_calculator", args_schema=CalculatorInput)
def safe_calculator(expression: str) -> str:
    """
    Evaluates simple mathematical expressions safely.
    ALWAYS use this tool to calculate Total Cost of Ownership (TCO), apply the 15% flight tolerance rule, or sum budget items.
    """
    clean_expr = expression.replace(" ", "")
    
    if not re.match(r'^[\d\.\+\-\*\/\(\)]+$', clean_expr):
        return json.dumps({
            "status": "error",
            "message": "Security Error: Invalid characters detected. Only numbers and (+, -, *, /) are allowed."
        })
    
    try:
        result = _safe_eval_expression(clean_expr)
        
        return json.dumps({
            "status": "success",
            "expression_evaluated": clean_expr,
            "result_value": float(result)
        })
        
    except ZeroDivisionError:
        return json.dumps({"status": "error", "message": "Math Error: Division by zero."})
    except Exception as e:
        return json.dumps({"status": "error", "message": "Math Error: Invalid expression formulation."})


@tool("convert_currency", args_schema=CurrencyInput)
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Converts monetary values between different currencies in real-time.
    ALWAYS use this tool when expenses are in a foreign currency. All corporate policy limits are in EUR, so you must convert to EUR before auditing.
    """
    from_curr = from_currency.strip().upper()
    to_curr = to_currency.strip().upper()
    
    if from_curr == to_curr:
        return json.dumps({
            "status": "success",
            "original_amount": amount,
            "converted_amount": amount,
            "currency": to_curr
        })

    try:
        url = f"https://api.frankfurter.app/latest?amount={amount}&from={from_curr}&to={to_curr}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            converted_amount = data["rates"][to_curr]
            
            return json.dumps({
                "status": "success",
                "original_amount": amount,
                "original_currency": from_curr,
                "converted_amount": converted_amount,
                "target_currency": to_curr
            })
            
        elif response.status_code == 404:
            return json.dumps({
                "status": "error", 
                "message": f"Currency code unsupported: {from_curr} to {to_curr}."
            })
        else:
            return json.dumps({
                "status": "error", 
                "message": f"API returned status code {response.status_code}."
            })
            
    except Exception as e:
        logger.error(f"Currency API Error: {e}")
        return json.dumps({
            "status": "error", 
            "message": "Connection error to currency API. Assume a 1:1 exchange rate for estimation and warn the user."
        })