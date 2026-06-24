import json
from unittest.mock import MagicMock, patch

import requests

from src.agents.domain.models import HotelList
from src.agents.infrastructure.tools.operator_tools import safe_calculator, convert_currency


#-----------------------------------------#
#             safe_calculator             #
#-----------------------------------------#


def test_safe_calculator_addition():
    result = json.loads(safe_calculator.func("10+5"))
    assert result["status"] == "success"
    assert result["result_value"] == 15.0


def test_safe_calculator_subtraction():
    result = json.loads(safe_calculator.func("100 - 37.5"))
    assert result["status"] == "success"
    assert result["result_value"] == 62.5


def test_safe_calculator_multiplication():
    result = json.loads(safe_calculator.func("150 * 3"))
    assert result["status"] == "success"
    assert result["result_value"] == 450.0


def test_safe_calculator_division():
    result = json.loads(safe_calculator.func("360 / 3"))
    assert result["status"] == "success"
    assert result["result_value"] == 120.0


def test_safe_calculator_complex_expression():
    result = json.loads(safe_calculator.func("120 * 2 + 45 * 3"))
    assert result["status"] == "success"
    assert result["result_value"] == 375.0


def test_safe_calculator_division_by_zero():
    result = json.loads(safe_calculator.func("100/0"))
    assert result["status"] == "error"
    assert "Division by zero" in result["message"]


def test_safe_calculator_rejects_invalid_characters():
    result = json.loads(safe_calculator.func("__import__('os')"))
    assert result["status"] == "error"
    assert "Security Error" in result["message"]


def test_safe_calculator_rejects_letters():
    result = json.loads(safe_calculator.func("10 + x"))
    assert result["status"] == "error"
    assert "Security Error" in result["message"]


def test_safe_calculator_negative_unary():
    result = json.loads(safe_calculator.func("-50 + 100"))
    assert result["status"] == "success"
    assert result["result_value"] == 50.0


def test_safe_calculator_float_result():
    result = json.loads(safe_calculator.func("1/3"))
    assert result["status"] == "success"
    assert abs(result["result_value"] - 0.3333) < 0.001


def test_safe_calculator_returns_expression_evaluated():
    result = json.loads(safe_calculator.func("20*5"))
    assert result["expression_evaluated"] == "20*5"


#-----------------------------------------#
#             sconvert_currency           #
#-----------------------------------------#


def test_convert_currency_same_currency():
    result_str = convert_currency.func(150.0, " eur ", "EUR")
    result = json.loads(result_str)

    assert result["status"] == "success"
    assert result["original_amount"] == 150.0
    assert result["converted_amount"] == 150.0
    assert result["currency"] == "EUR"


@patch("src.agents.infrastructure.tools.operator_tools.requests.get")
def test_convert_currency_success(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "amount": 100.0,
        "base": "GBP",
        "date": "2026-06-22",
        "rates": {"EUR": 118.50}
    }
    mock_get.return_value = mock_response

    result_str = convert_currency.func(100.0, "gbp", "eur")
    result = json.loads(result_str)

    assert result["status"] == "success"
    assert result["original_currency"] == "GBP"
    assert result["target_currency"] == "EUR"
    assert result["converted_amount"] == 118.50
    
    mock_get.assert_called_once()
    called_url = mock_get.call_args[0][0]
    assert "from=GBP" in called_url
    assert "to=EUR" in called_url


@patch("src.agents.infrastructure.tools.operator_tools.requests.get")
def test_convert_currency_unsupported_code_404(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_get.return_value = mock_response

    result_str = convert_currency.func(100.0, "FAKE", "EUR")
    result = json.loads(result_str)

    assert result["status"] == "error"
    assert "Currency code unsupported" in result["message"]
    assert "FAKE to EUR" in result["message"]


@patch("src.agents.infrastructure.tools.operator_tools.requests.get")
def test_convert_currency_server_error_500(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_get.return_value = mock_response

    result_str = convert_currency.func(100.0, "USD", "EUR")
    result = json.loads(result_str)

    assert result["status"] == "error"
    assert "API returned status code 500" in result["message"]


@patch("src.agents.infrastructure.tools.operator_tools.requests.get")
def test_convert_currency_connection_exception(mock_get):
    mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

    result_str = convert_currency.func(100.0, "USD", "EUR")
    result = json.loads(result_str)

    assert result["status"] == "error"
    assert "Connection error to currency API" in result["message"]
    assert "Assume a 1:1 exchange rate" in result["message"]


