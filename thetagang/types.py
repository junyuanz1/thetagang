from enum import Enum


class OptionRight(Enum):
    PUT = "P"
    CALL = "C"

    def p_or_c(self) -> str:
        if self == OptionRight.PUT:
            return "puts"
        elif self == OptionRight.CALL:
            return "calls"
        else:
            raise ValueError(f"Unknown option right: {self}")
