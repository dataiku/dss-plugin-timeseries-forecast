import logging
from typing import Any
import re

logger = logging.getLogger(__name__)

DEFAULT_ERROR_MESSAGES = {
    'exists': 'This field is required.',
    'in': 'Should be in {op}. (Currently {value})',
    'not_in': 'Should not be in {op}. (Currently {value})',
    'eq': 'Should be equal to {op} (Currently {value}).',
    'sup': 'Should be greater than {op} (Currently {value}).',
    'sup_eq': 'Should be greater than or equal to {op} (Currently {value}).',
    'inf': 'Should be less than {op} (Currently {value}).',
    'inf_eq': 'Should be less than or equal to {op} (Currently {value}).',
    'between': 'Should be between {op[0]} and {op[1]} inclusive (Currently {value}).',
    'between_strict': 'Should be between {op[0]} and {op[1]} exclusive (Currently {value}).',
    'is_type': 'Should be of type <class \'{op}\'> (Currently {value_type}).',
    'is_castable': 'Should be castable to type {op}> (Currently {value} with type {value_type}.',
    'custom': "There has been an unknown error.",
    'match': "Should match the following pattern: {op}.",
    'is_subset': 'Should be a subset of {op}. (Currently {value})'
}


class CustomCheckError(Exception):
    """Exception raised when condition of CustomCheck are not met."""

    pass


class CustomCheck:
    """Class related to a check. Use run() to verify whether the check fails or pass

    Attributes:
        type (str): Type of the CustomCheck. Must have a related method having "_" before type name
        op (Any, optional): Operator to compare the value to. Unnecessary for som checks
        err_msg (str, optional): Custom message to display if check fails. Default is a generic message
    """

    def __init__(self, type, op: Any = None, err_msg: str = ""):
        """Initialization method for the CustomCheck class

        Args:
            type (str): Type of the CustomCheck. Must have a related method having "_" before type name
            op (Any, optional): Operator to compare the value to. Unnecessary for som checks
            err_msg (str, optional): Custom message to display if check fails. Default is a generic message
        """
        self.type = type
        func_name = "_{}".format(self.type)
        if not hasattr(self, func_name):
            raise CustomCheckError("Check of type {} does not exist.".format(self.type))
        self.op = op
        self.err_msg = err_msg or self.get_default_err_msg()

    def run(self, value: Any = None):
        """Runs the check on a value

        Args:
            value(Any, optional): The value to run the check on. Default is None
        """
        func_name = "_{}".format(self.type)
        result = getattr(self, func_name)(value)
        self.handle_return(result, value)

    def handle_return(self, result: bool, value: Any):
        """Checks whether the check has failed or pass

        Args:
            result(bool): True if check has passed else False
            value(Any): The value on which the test has been ran

        Raises:
            CustomCheckError if check fails
        """
        try:
            assert result
        except AssertionError:
            raise CustomCheckError(self.format_err_msg(value))

    def get_default_err_msg(self) -> str:
        """Returns the default message related to check's type

        Returns:
            str: Unformatted default error message
        """
        return DEFAULT_ERROR_MESSAGES.get(self.type, "custom")

    def format_err_msg(self, value: Any) -> str:
        """Format the error message with the value that has failed the test

        Args:
            value(Any): Failure value

        Returns:
            str: Error messages formatted
        """
        formatted_err_msg = self.err_msg.format(value=value, op=self.op, value_type=type(value))
        return f"{formatted_err_msg}"

    def _exists(self, value: Any) -> bool:
        """Checks whether the value is None

        Args:
            value(Any): Value to test

        Returns:
            bool: Whether the check has succeed
        """
        EMPTY_ENTRIES = [[], "", None]
        return value not in EMPTY_ENTRIES

    def _in(self, value: Any) -> bool:
        """Checks whether the value is in the iterable given in "op" attribute. If the value and the operator are lists,
        it checks if the value is a subset of the operator.

        Args:
            value(Any): Value to test

        Returns:
            bool: Whether the check has succeed
        """
        if isinstance(value, list) and isinstance(self.op, list):
            return all(x in self.op for x in value)
        else:
            return value in self.op

    def _not_in(self, value: Any) -> bool:
        """Checks whether the value is not in the iterable given in "op" attribute

        Args:
            value(Any): Value to test

        Returns:
            bool: Whether the check has succeed
        """
        return value not in self.op

    def _eq(self, value: Any) -> bool:
        """Checks whether the value is equal to "op" attribute

        Args:
            value(Any): Value to test

        Returns:
            bool: Whether the check has succeed
        """
        return value == self.op

    def _sup(self, value: Any) -> bool:
        """Checks whether the value is superior to "op" attribute

        Args:
            value(Any): Value to test

        Returns:
            bool: Whether the check has succeed
        """
        return value > float(self.op)

    def _inf(self, value: Any) -> bool:
        """Checks whether the value is inferior to "op" attribute

        Args:
            value(Any): Value to test

        Returns:
            bool: Whether the check has succeed
        """
        return value < float(self.op)

    def _sup_eq(self, value: Any) -> bool:
        """Checks whether the value is superior or equal to "op" attribute

        Args:
            value(Any): Value to test

        Returns:
            bool: Whether the check has succeed
        """
        return value >= float(self.op)

    def _inf_eq(self, value: Any) -> bool:
        """Checks whether the value is inferior or equal to "op" attribute

        Args:
            value(Any): Value to test

        Returns:
            bool: Whether the check has succeed
        """
        return value <= float(self.op)

    def _between(self, value: Any) -> bool:
        """Checks whether the value is between the first and second member of "op" attribute

        Args:
            value(Any): Value to test

        Returns:
            bool: Whether the check has succeed
        """
        return float(self.op[0]) <= value <= float(self.op[1])

    def _between_strict(self, value: Any) -> bool:
        """Checks whether the value is strictly between the first and second member of "op" attribute

        Args:
            value(Any): Value to test

        Returns:
            bool: Whether the check has succeed
        """
        return float(self.op[0]) < value < float(self.op[1])

    def _is_type(self, value: Any) -> bool:
        """Checks whether the value has the type given in "op" attribute

        Args:
            value(Any): Value to test

        Returns:
            bool: Whether the check has succeed
        """
        return isinstance(value, self.op)

    def _is_castable(self, value: Any) -> bool:
        """Checks whether the value can be cast to the op type

        Args:
            value(Any): Value to test

        Returns:
            bool: Whether the check has succeed
        """
        try:
            _ = self.op(value)
            return True
        except (TypeError, ValueError):
            return False

    def _custom(self, *args) -> bool:
        """Checks whether "op" attribute is true or false

        Returns:
            bool: Whether the check has succeed
        """
        return self.op

    def _match(self, value: Any) -> bool:
        """Checks whether "value" matches the regex provided in "op" attribute

        Returns:
            bool: Whether the check has succeed
        """
        return not not re.match(self.op, value)

    def _is_subset(self, value: Any) -> bool:
        """Checks whether "value" is a subset of "op"

        Returns:
            bool: Whether the check has succeed
        """
        return set(value).issubset(set(self.op))
