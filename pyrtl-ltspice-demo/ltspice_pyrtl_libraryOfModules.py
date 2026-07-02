import pyrtl


def half_adder(A, B, name="ha"):
    """
    1-bit half adder.

    Inputs:
        A, B: 1-bit PyRTL wires

    Outputs:
        SUM, CARRY: 1-bit PyRTL wires
    """

    SUM = pyrtl.WireVector(1, f"{name}_sum")
    CARRY = pyrtl.WireVector(1, f"{name}_carry")

    SUM <<= A ^ B
    CARRY <<= A & B

    return SUM, CARRY


def full_adder(A, B, Cin, name="fa"):
    """
    1-bit full adder built from basic PyRTL gate operations.

    Inputs:
        A, B, Cin: 1-bit PyRTL wires

    Outputs:
        SUM, Cout: 1-bit PyRTL wires
    """

    partial_sum = pyrtl.WireVector(1, f"{name}_partial_sum")
    carry1 = pyrtl.WireVector(1, f"{name}_carry1")
    carry2 = pyrtl.WireVector(1, f"{name}_carry2")

    SUM = pyrtl.WireVector(1, f"{name}_sum")
    Cout = pyrtl.WireVector(1, f"{name}_cout")

    partial_sum <<= A ^ B
    carry1 <<= A & B
    carry2 <<= partial_sum & Cin

    SUM <<= partial_sum ^ Cin
    Cout <<= carry1 | carry2

    return SUM, Cout
