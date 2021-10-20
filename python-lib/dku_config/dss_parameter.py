from .custom_check import CustomCheck, CustomCheckError
from typing import Any, List

import logging

logger = logging.getLogger(__name__)


class DSSParameterError(Exception):
    """Exception raised when at least one CustomCheck fails."""

    pass


class DSSParameter:
    """Object related to one parameter. It is mainly used for checks to run in backend for custom forms.

    Attributes:
        name(str): Name of the parameter
        value(Any): Value of the parameter
        checks(list[dict], optional): Checks to run on provided value
        required(bool, optional): Whether the value can be None
        cast_to(type, optional): The type to cast the variable in
        cast_to(Any, optional): The default value of the variable (If value is None)
        label(str, optional): The name displayed to the end user
    """

    def __init__(
        self,
        name: str,
        value: Any,
        checks: List[dict] = None,
        required: bool = False,
        cast_to: type = None,
        default: Any = None,
        label: str = None,
    ):
        """Initialization method for the DSSParameter class

        Args:
            name(str): Name of the parameter
            value(Any): Value of the parameter
            checks(list[dict], optional): Checks to run on provided value
            required(bool, optional): Whether the value can be None
            cast_to(type, optional): The type to cast the variable in
            default(Any, optional): The default value of the variable (If value is None)
            label(str, optional): The name displayed to the end user
        """
        if checks is None:
            checks = []
        self.name = name
        self.value = value if value is not None else default
        self.required = required
        self.cast_to = cast_to
        self.checks = [CustomCheck(**check) for check in checks]
        if label is None:
            self.label = name
        else:
            self.label = label

        value_exists = self.run_checks([CustomCheck(type="exists")], raise_error=self.required)
        if value_exists:
            if self.cast_to:
                self.cast_value()
            self.run_checks(self.checks)

    def cast_value(self):
        """Cast the value if there is as cast_to attribute else return the value as it is"""
        if self.cast_to:
            self.run_checks([CustomCheck(type="is_castable", op=self.cast_to)])
            self.value = self.cast_to(self.value)

    def run_checks(self, checks, raise_error=True):
        """Runs all checks provided for this parameter

        Args:
            checks(list[Check]): Checks to run
            raise_error(bool, optional): Whether to rise an error if a check fails

        Returns:
            bool: Whether all checks have passed

        Raises:
            DSSParameterError: Raises if at least on check fails and raise_error is True
        """
        for check in checks:
            try:
                check.run(self.value)
            except CustomCheckError as err:
                if raise_error:
                    self.handle_failure(err)
                return False
        self.handle_success()
        return True

    def handle_failure(self, error: CustomCheckError):
        """Is called when at least one test fails. It will raise an Exception with understandable text

        Args:
            error(CustomCheckError): Errors met when running checks

        Raises:
            DSSParameterError: Raises if at least on check fails
        """
        raise DSSParameterError(self.format_failure_message(error))

    def format_failure_message(self, error: CustomCheckError) -> str:
        """Format failure text

        Args:
            error (CustomCheckError): Error met when running check

        Returns:
            str: Formatted error message
        """
        return """
        Validation error with parameter \"{name}\":
        {error}
        """.format(
            name=self.label, error=error
        )

    def handle_success(self):
        """Called if all checks are successful. Prints a success message"""
        self.print_success_message()

    def print_success_message(self):
        """Formats the success message"""
        logger.debug("All checks passed successfully for {}.".format(self.name))

    def __repr__(self):
        return "DSSParameter(name={}, value={})".format(self.name, self.value)

    def __str__(self):
        return "DSSParameter(name={}, value={})".format(self.name, self.value)
