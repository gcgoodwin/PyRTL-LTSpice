"""PyRTL to transistor-level LTspice netlist converter.

This converter maps PyRTL's internal LogicNet operations to a reusable
transistor-level standard-cell library embedded in the generated .cir file.

Supported PyRTL operations:
    w, ~, &, |, ^, n, x, c, s, +, -, *, =, <, >, r

Not supported:
    m, @   (memory read and memory write)

The public function ``convert_working_block_to_ltspice`` remains compatible
with the earlier converter, while adding direct macro/cell mapping, vector
handling, constants, multiplexers, registers, clock generation, and clearer
errors.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pyrtl


# Operations directly understood by this converter.
SUPPORTED_OPS = frozenset({
    "w", "~", "&", "|", "^", "n", "x", "c", "s",
    "+", "-", "*", "=", "<", ">", "r",
})
MEMORY_OPS = frozenset({"m", "@"})

GATE_MAP = {
    "&": "AND2",
    "|": "OR2",
    "^": "XOR2",
    "n": "NAND2",
}


@dataclass(frozen=True)
class LtspiceConfig:
    """Electrical and transient-simulation settings."""

    vdd: float = 5.0
    input_base_period_us: float = 20.0
    combinational_input_delay_us: float = 1.0
    sequential_input_delay_us: float = 12.0
    rise_fall_ns: float = 50.0
    simulation_time_us: float = 80.0
    max_timestep_ns: float = 100.0

    clock_period_us: float = 10.0
    clock_delay_us: float = 5.0
    clock_high_us: float = 5.0

    # INIT is high for the first rising edge so register reset/initial values
    # are clocked into the generated DFFs. It then remains low.
    initialize_registers: bool = True
    init_release_us: float = 6.0

    output_capacitance: str = "10f"
    output_resistance: str = "100Meg"


class LtspiceEmitter:
    """Small helper that creates unique instance and internal-node names."""

    def __init__(self) -> None:
        self.lines: List[str] = []
        self._instance_count = 0
        self._node_count = 0

    def line(self, text: str = "") -> None:
        self.lines.append(text)

    def node(self, hint: str = "n") -> str:
        self._node_count += 1
        return clean_name(f"__{hint}_{self._node_count}")

    def instance(
        self,
        cell: str,
        nodes: Sequence[str],
        hint: str = "X",
    ) -> str:
        self._instance_count += 1
        name = clean_name(f"X{hint}_{self._instance_count}")
        self.line(f"{name} {' '.join(nodes)} {cell}")
        return name

    def controlled_wire(self, src: str, dst: str, hint: str = "WIRE") -> None:
        """Emit a directional ideal wire using a unity-gain VCVS.

        This preserves useful destination node names without shorting nodes
        together with many parallel zero-volt sources.
        """
        if src == dst:
            return
        self._instance_count += 1
        name = clean_name(f"E{hint}_{self._instance_count}")
        self.line(f"{name} {dst} 0 {src} 0 1")


# ---------------------------------------------------------------------------
# Name and bit helpers
# ---------------------------------------------------------------------------


def clean_name(name: object) -> str:
    """Make a PyRTL name safe and unambiguous for SPICE."""
    result = str(name)
    result = result.replace("[", "_").replace("]", "")
    result = result.replace("/", "_")
    result = re.sub(r"[^A-Za-z0-9_]", "_", result)
    if not result:
        result = "unnamed_node"
    # Avoid names that SPICE may parse as a numeric node.
    if result[0].isdigit():
        result = "n_" + result
    return result


def wire_bit_name(wire: pyrtl.WireVector, bit: int) -> str:
    """Return the SPICE node for one bit of a non-constant PyRTL wire."""
    if bit < 0 or bit >= len(wire):
        raise IndexError(f"Bit {bit} is outside wire {wire.name!r}/{len(wire)}")
    if len(wire) == 1:
        return clean_name(wire.name)
    return clean_name(f"{wire.name}[{bit}]")


def const_bit_value(wire: pyrtl.Const, bit: int) -> int:
    if bit < 0:
        raise IndexError("Negative constant bit index")
    if bit >= len(wire):
        return 0
    return (int(wire.val) >> bit) & 1


def signal_bit_node(wire: pyrtl.WireVector, bit: int) -> str:
    """Return a node for ``wire[bit]``, zero-extending when necessary."""
    if bit >= len(wire):
        return "0"
    if isinstance(wire, pyrtl.Const):
        return "VDD" if const_bit_value(wire, bit) else "0"
    return wire_bit_name(wire, bit)


def all_wire_bit_nodes(wire: pyrtl.WireVector) -> List[str]:
    return [signal_bit_node(wire, bit) for bit in range(len(wire))]


def iter_io_bits(wires: Iterable[pyrtl.WireVector]) -> Iterable[Tuple[pyrtl.WireVector, int, str]]:
    for wire in wires:
        for bit in range(len(wire)):
            yield wire, bit, wire_bit_name(wire, bit)


# ---------------------------------------------------------------------------
# PyRTL preprocessing and register metadata
# ---------------------------------------------------------------------------


def preprocess_block(
    block: pyrtl.Block,
    remove_wire_nets: bool = False,
    synthesize: bool = False,
) -> pyrtl.Block:
    """Optionally lower the PyRTL design before LTspice conversion."""
    if synthesize:
        block = pyrtl.synthesize(
            update_working_block=False,
            merge_io_vectors=False,
            block=block,
        )

    if remove_wire_nets:
        # _remove_wire_nets is an internal PyRTL pass. Keep it optional because
        # internal APIs may change between PyRTL releases.
        remove_fn = getattr(pyrtl.passes, "_remove_wire_nets", None)
        if remove_fn is None:
            raise RuntimeError(
                "This PyRTL version does not expose pyrtl.passes._remove_wire_nets"
            )
        remove_fn(block, skip_sanity_check=True)

    return block


def build_register_init_map(block: pyrtl.Block) -> Dict[pyrtl.WireVector, int]:
    """Map each register bit wire to its intended initial value.

    PyRTL's synthesized one-bit registers lose the original ``reset_value`` on
    the bit object, but PostSynthBlock.reg_map preserves the correspondence.
    A reset_value of None is treated as zero for deterministic LTspice startup,
    matching normal PyRTL simulation startup behavior.
    """
    result: Dict[pyrtl.WireVector, int] = {}

    reg_map = getattr(block, "reg_map", None)
    if reg_map:
        for original_reg, synthesized_bits in reg_map.items():
            reset_value = original_reg.reset_value
            if reset_value is None:
                reset_value = 0
            for bit_index, bit_reg in enumerate(synthesized_bits):
                result[bit_reg] = (int(reset_value) >> bit_index) & 1

    for wire in block.wirevector_set:
        if isinstance(wire, pyrtl.Register) and wire not in result:
            reset_value = wire.reset_value
            if reset_value is None:
                reset_value = 0
            # A direct multi-bit register is represented by one WireVector. The
            # per-bit value is recovered later from this packed integer.
            result[wire] = int(reset_value)

    return result


# ---------------------------------------------------------------------------
# Cell-level emitters
# ---------------------------------------------------------------------------


def emit_gate(em: LtspiceEmitter, cell: str, a: str, b: str, y: str, hint: str) -> None:
    em.instance(cell, [a, b, y, "VDD", "0"], hint)


def emit_inv(em: LtspiceEmitter, a: str, y: str, hint: str = "INV") -> None:
    em.instance("INV", [a, y, "VDD", "0"], hint)


def emit_mux(em: LtspiceEmitter, a: str, b: str, sel: str, y: str, hint: str = "MUX") -> None:
    # Pin order: A(false), B(true), S, Y, VDD, VSS
    em.instance("MUX2", [a, b, sel, y, "VDD", "0"], hint)


def emit_full_adder(
    em: LtspiceEmitter,
    a: str,
    b: str,
    cin: str,
    sum_node: str,
    cout: str,
    hint: str = "FA",
) -> None:
    em.instance("FULLADDER", [a, b, cin, sum_node, cout, "VDD", "0"], hint)


def emit_ripple_add(
    em: LtspiceEmitter,
    a_nodes: Sequence[str],
    b_nodes: Sequence[str],
    out_nodes: Sequence[str],
    carry_in: str = "0",
    invert_b: bool = False,
    hint: str = "ADD",
) -> str:
    """Emit an arbitrary-width ripple adder and return final carry node."""
    carry = carry_in
    for bit, out_node in enumerate(out_nodes):
        a = a_nodes[bit] if bit < len(a_nodes) else "0"
        b = b_nodes[bit] if bit < len(b_nodes) else "0"
        if invert_b:
            b_inv = em.node(f"{hint}_binv_{bit}")
            emit_inv(em, b, b_inv, f"{hint}_BINV")
            b = b_inv
        next_carry = em.node(f"{hint}_carry_{bit}")
        emit_full_adder(em, a, b, carry, out_node, next_carry, f"{hint}_{bit}")
        carry = next_carry
    return carry


def emit_equality(
    em: LtspiceEmitter,
    a_nodes: Sequence[str],
    b_nodes: Sequence[str],
    dest: str,
    hint: str = "EQ",
) -> None:
    width = max(len(a_nodes), len(b_nodes))
    if width == 0:
        em.controlled_wire("VDD", dest, hint)
        return

    eq_acc = "VDD"
    for bit in range(width):
        a = a_nodes[bit] if bit < len(a_nodes) else "0"
        b = b_nodes[bit] if bit < len(b_nodes) else "0"
        xnor = em.node(f"{hint}_xnor_{bit}")
        em.instance("XNOR2", [a, b, xnor, "VDD", "0"], f"{hint}_XNOR")
        if eq_acc == "VDD":
            eq_acc = xnor
        else:
            eq_next = em.node(f"{hint}_acc_{bit}")
            emit_gate(em, "AND2", eq_acc, xnor, eq_next, f"{hint}_AND")
            eq_acc = eq_next
    em.controlled_wire(eq_acc, dest, hint)


def emit_unsigned_less_than(
    em: LtspiceEmitter,
    a_nodes: Sequence[str],
    b_nodes: Sequence[str],
    dest: str,
    hint: str = "LT",
) -> None:
    """Emit an MSB-first unsigned comparator."""
    width = max(len(a_nodes), len(b_nodes))
    eq_prefix = "VDD"
    lt_acc = "0"

    for bit in reversed(range(width)):
        a = a_nodes[bit] if bit < len(a_nodes) else "0"
        b = b_nodes[bit] if bit < len(b_nodes) else "0"

        not_a = em.node(f"{hint}_nota_{bit}")
        emit_inv(em, a, not_a, f"{hint}_NOTA")

        a_lt_b = em.node(f"{hint}_bitlt_{bit}")
        emit_gate(em, "AND2", not_a, b, a_lt_b, f"{hint}_BITLT")

        qualified_lt = em.node(f"{hint}_qualified_{bit}")
        emit_gate(em, "AND2", eq_prefix, a_lt_b, qualified_lt, f"{hint}_QUAL")

        if lt_acc == "0":
            lt_next = qualified_lt
        else:
            lt_next = em.node(f"{hint}_ltacc_{bit}")
            emit_gate(em, "OR2", lt_acc, qualified_lt, lt_next, f"{hint}_OR")
        lt_acc = lt_next

        bit_equal = em.node(f"{hint}_eqbit_{bit}")
        em.instance("XNOR2", [a, b, bit_equal, "VDD", "0"], f"{hint}_XNOR")
        eq_next = em.node(f"{hint}_eqprefix_{bit}")
        emit_gate(em, "AND2", eq_prefix, bit_equal, eq_next, f"{hint}_EQAND")
        eq_prefix = eq_next

    em.controlled_wire(lt_acc, dest, hint)


def emit_unsigned_multiply(
    em: LtspiceEmitter,
    a_nodes: Sequence[str],
    b_nodes: Sequence[str],
    out_nodes: Sequence[str],
    hint: str = "MUL",
) -> None:
    """Emit a shift-and-add unsigned multiplier."""
    width = len(out_nodes)
    accumulator: List[str] = ["0"] * width

    for b_index, b in enumerate(b_nodes):
        partial: List[str] = []
        for out_bit in range(width):
            a_index = out_bit - b_index
            if 0 <= a_index < len(a_nodes):
                p = em.node(f"{hint}_pp_{b_index}_{out_bit}")
                emit_gate(em, "AND2", a_nodes[a_index], b, p, f"{hint}_PP")
                partial.append(p)
            else:
                partial.append("0")

        next_acc: List[str] = [em.node(f"{hint}_acc_{b_index}_{i}") for i in range(width)]
        emit_ripple_add(
            em,
            accumulator,
            partial,
            next_acc,
            carry_in="0",
            invert_b=False,
            hint=f"{hint}_ROW{b_index}",
        )
        accumulator = next_acc

    for src, dst in zip(accumulator, out_nodes):
        em.controlled_wire(src, dst, hint)


# ---------------------------------------------------------------------------
# LogicNet mapping
# ---------------------------------------------------------------------------


def emit_logic_net(
    em: LtspiceEmitter,
    net: pyrtl.LogicNet,
    register_init_map: Mapping[pyrtl.WireVector, int],
    config: LtspiceConfig,
) -> None:
    op = net.op

    if op in MEMORY_OPS:
        raise NotImplementedError(
            f"PyRTL memory operation {op!r} is not supported yet: {net}. "
            "Implementing m/@ requires a memory-cell or register-file backend, "
            "address decoding, and write-enable/clock semantics."
        )

    if op not in SUPPORTED_OPS:
        raise NotImplementedError(
            f"Unsupported PyRTL operation {op!r} in net: {net}. "
            f"Supported operations are: {', '.join(sorted(SUPPORTED_OPS))}."
        )

    # Directional wiring.
    if op == "w":
        src, dest = net.args[0], net.dests[0]
        if len(src) != len(dest):
            raise ValueError(f"Wire width mismatch in net: {net}")
        for bit in range(len(dest)):
            em.controlled_wire(signal_bit_node(src, bit), wire_bit_name(dest, bit), "WIRE")
        return

    # Concatenation. First arg is MSB-side; last arg is LSB-side.
    if op == "c":
        dest = net.dests[0]
        sources_lsb_first: List[Tuple[pyrtl.WireVector, int]] = []
        for arg in reversed(net.args):
            sources_lsb_first.extend((arg, bit) for bit in range(len(arg)))
        if len(sources_lsb_first) != len(dest):
            raise ValueError(f"Concat width mismatch in net: {net}")
        for dest_bit, (arg, arg_bit) in enumerate(sources_lsb_first):
            em.controlled_wire(
                signal_bit_node(arg, arg_bit),
                wire_bit_name(dest, dest_bit),
                "CONCAT",
            )
        return

    # Selection/reordering. op_param gives source bit for each destination bit.
    if op == "s":
        src, dest = net.args[0], net.dests[0]
        selected_bits = tuple(net.op_param)
        if len(selected_bits) != len(dest):
            raise ValueError(f"Select width mismatch in net: {net}")
        for dest_bit, src_bit in enumerate(selected_bits):
            em.controlled_wire(
                signal_bit_node(src, int(src_bit)),
                wire_bit_name(dest, dest_bit),
                "SELECT",
            )
        return

    # Register: positive-edge-triggered DFF, one cell per bit.
    if op == "r":
        next_value, reg = net.args[0], net.dests[0]
        if len(next_value) != len(reg):
            raise ValueError(f"Register width mismatch in net: {net}")

        packed_init = int(register_init_map.get(reg, 0))
        # Synthesized register bits may have map values of 0/1 directly.
        if len(reg) == 1 and reg in register_init_map:
            packed_init = int(register_init_map[reg])

        for bit in range(len(reg)):
            d = signal_bit_node(next_value, bit)
            q = wire_bit_name(reg, bit)
            init_value = (packed_init >> bit) & 1
            init_node = "VDD" if init_value else "0"

            if config.initialize_registers:
                d_after_init = em.node(f"{reg.name}_dinit_{bit}")
                emit_mux(em, d, init_node, "INIT", d_after_init, "REG_INIT")
                d = d_after_init

            em.instance("DFF", [d, q, "CLK", "VDD", "0"], f"DFF_{reg.name}_{bit}")
            # Helpful operating-point hint. The INIT/clock sequence is the
            # actual deterministic initialization mechanism.
            em.line(f".ic V({q})={config.vdd if init_value else 0}")
        return

    # Unary bitwise NOT.
    if op == "~":
        src, dest = net.args[0], net.dests[0]
        if len(src) != len(dest):
            raise ValueError(f"NOT width mismatch in net: {net}")
        for bit in range(len(dest)):
            emit_inv(em, signal_bit_node(src, bit), wire_bit_name(dest, bit), "NOT")
        return

    # Binary bitwise gates.
    if op in GATE_MAP:
        a, b = net.args
        dest = net.dests[0]
        for bit in range(len(dest)):
            emit_gate(
                em,
                GATE_MAP[op],
                signal_bit_node(a, bit),
                signal_bit_node(b, bit),
                wire_bit_name(dest, bit),
                GATE_MAP[op],
            )
        return

    # Vector mux.
    if op == "x":
        sel, falsecase, truecase = net.args
        dest = net.dests[0]
        if len(sel) != 1:
            raise ValueError(f"Mux select must be one bit: {net}")
        if len(falsecase) != len(truecase) or len(dest) != len(falsecase):
            raise ValueError(f"Mux data width mismatch: {net}")
        select_node = signal_bit_node(sel, 0)
        for bit in range(len(dest)):
            emit_mux(
                em,
                signal_bit_node(falsecase, bit),
                signal_bit_node(truecase, bit),
                select_node,
                wire_bit_name(dest, bit),
                "MUX",
            )
        return

    # Arithmetic and comparisons.
    if op == "+":
        a, b = net.args
        dest = net.dests[0]
        emit_ripple_add(
            em,
            all_wire_bit_nodes(a),
            all_wire_bit_nodes(b),
            [wire_bit_name(dest, bit) for bit in range(len(dest))],
            carry_in="0",
            invert_b=False,
            hint="ADD",
        )
        return

    if op == "-":
        a, b = net.args
        dest = net.dests[0]
        emit_ripple_add(
            em,
            all_wire_bit_nodes(a),
            all_wire_bit_nodes(b),
            [wire_bit_name(dest, bit) for bit in range(len(dest))],
            carry_in="VDD",
            invert_b=True,
            hint="SUB",
        )
        return

    if op == "*":
        a, b = net.args
        dest = net.dests[0]
        emit_unsigned_multiply(
            em,
            all_wire_bit_nodes(a),
            all_wire_bit_nodes(b),
            [wire_bit_name(dest, bit) for bit in range(len(dest))],
            hint="MUL",
        )
        return

    if op == "=":
        a, b = net.args
        dest = net.dests[0]
        emit_equality(
            em,
            all_wire_bit_nodes(a),
            all_wire_bit_nodes(b),
            wire_bit_name(dest, 0),
            "EQ",
        )
        return

    if op == "<":
        a, b = net.args
        dest = net.dests[0]
        emit_unsigned_less_than(
            em,
            all_wire_bit_nodes(a),
            all_wire_bit_nodes(b),
            wire_bit_name(dest, 0),
            "LT",
        )
        return

    if op == ">":
        a, b = net.args
        dest = net.dests[0]
        emit_unsigned_less_than(
            em,
            all_wire_bit_nodes(b),
            all_wire_bit_nodes(a),
            wire_bit_name(dest, 0),
            "GT",
        )
        return

    raise AssertionError(f"Unhandled supported operation {op!r}")


# ---------------------------------------------------------------------------
# Embedded transistor-level cell library
# ---------------------------------------------------------------------------


def write_gate_library(f) -> None:
    """Write MOS models and reusable transistor-level cells.

    Sequential cells use transmission-gate latches instead of cross-coupled
    NAND gate macros. Tiny storage capacitors and weak shunts improve LTspice
    transient convergence without changing the intended digital behavior.
    """
    f.write(r"""

* 5) TRANSISTOR MODELS

* Conservative long-channel dimensions are used with these Level-1 models.
.model NMOS NMOS (LEVEL=1 VTO=0.7 KP=120u LAMBDA=0.02)
.model PMOS PMOS (LEVEL=1 VTO=-0.7 KP=60u LAMBDA=0.02)


* 6) TRANSISTOR-LEVEL STANDARD-CELL LIBRARY

* Inverter
* Pin order: IN OUT VDD VSS
.subckt INV IN OUT VDD VSS
MP1 OUT IN VDD VDD PMOS W=80u L=4u
MN1 OUT IN VSS VSS NMOS W=40u L=4u
.ends INV


* Two-input NAND. Series NMOS devices are widened.
* Pin order: A B Y VDD VSS
.subckt NAND2 A B Y VDD VSS
MP1 Y A VDD VDD PMOS W=80u L=4u
MP2 Y B VDD VDD PMOS W=80u L=4u
MN1 Y A NINT VSS NMOS W=80u L=4u
MN2 NINT B VSS VSS NMOS W=80u L=4u
.ends NAND2


* AND = NAND + inverter
* Pin order: A B Y VDD VSS
.subckt AND2 A B Y VDD VSS
XNAND A B N_NAND VDD VSS NAND2
XINV N_NAND Y VDD VSS INV
.ends AND2


* Two-input NOR. Series PMOS devices are widened.
* Pin order: A B Y VDD VSS
.subckt NOR2 A B Y VDD VSS
MP1 NPU A VDD VDD PMOS W=160u L=4u
MP2 Y B NPU VDD PMOS W=160u L=4u
MN1 Y A VSS VSS NMOS W=40u L=4u
MN2 Y B VSS VSS NMOS W=40u L=4u
.ends NOR2


* OR = NOR + inverter
* Pin order: A B Y VDD VSS
.subckt OR2 A B Y VDD VSS
XNOR A B N_NOR VDD VSS NOR2
XINV N_NOR Y VDD VSS INV
.ends OR2


* XOR from four NAND gates
* Pin order: A B Y VDD VSS
.subckt XOR2 A B Y VDD VSS
XN1 A B N1 VDD VSS NAND2
XN2 A N1 N2 VDD VSS NAND2
XN3 B N1 N3 VDD VSS NAND2
XN4 N2 N3 Y VDD VSS NAND2
.ends XOR2


* XNOR = XOR + inverter
* Pin order: A B Y VDD VSS
.subckt XNOR2 A B Y VDD VSS
XXOR A B N_XOR VDD VSS XOR2
XINV N_XOR Y VDD VSS INV
.ends XNOR2


* 2:1 multiplexer
* Y=A when S=0; Y=B when S=1
* Pin order: A B S Y VDD VSS
.subckt MUX2 A B S Y VDD VSS
XINV_S S S_B VDD VSS INV
XAND_A A S_B N_A VDD VSS AND2
XAND_B B S N_B VDD VSS AND2
XOR_OUT N_A N_B Y VDD VSS OR2
.ends MUX2


* Full adder
* Pin order: A B CIN SUM COUT VDD VSS
.subckt FULLADDER A B CIN SUM COUT VDD VSS
XXOR1 A B P VDD VSS XOR2
XXOR2 P CIN SUM VDD VSS XOR2
XAND1 A B C1 VDD VSS AND2
XAND2 P CIN C2 VDD VSS AND2
XOR1 C1 C2 COUT VDD VSS OR2
.ends FULLADDER


* Fast positive-edge-triggered DFF using LTspice's native digital device.
* External pin order remains: D Q CLK VDD VSS
*
* VDD remains in the subcircuit interface so the existing converter
* does not need any changes, although the native DFLOP does not use it.
.subckt DFF D Q CLK VDD VSS

* Native DFLOP pin order:
* D+ D- CLK PRE CLR QBAR Q COMMON
*
* Nonzero delay prevents zero-delay feedback races in counters.
A_DFF D VSS CLK VSS VSS VSS Q VSS DFLOP Vhigh=5 Vlow=0 Ref=2.5 Td=5n Trise=1n Tfall=1n

.ends DFF
""")


# ---------------------------------------------------------------------------
# Main conversion API
# ---------------------------------------------------------------------------


def convert_working_block_to_ltspice(
    output_filename: str,
    block: Optional[pyrtl.Block] = None,
    remove_wire_nets: bool = False,
    synthesize: bool = False,
    config: Optional[LtspiceConfig] = None,
) -> str:
    """Convert a PyRTL working block to a transistor-level LTspice .cir file.

    Args:
        output_filename: Destination .cir path.
        block: PyRTL Block. Defaults to pyrtl.working_block().
        remove_wire_nets: Optionally invoke PyRTL's internal wire-removal pass.
        synthesize: Optionally lower the block with pyrtl.synthesize first.
            Direct mapping is now capable of handling most high-level PyRTL
            operations, so synthesis is no longer required for +, -, *, muxes,
            comparisons, concatenation, selection, or registers.
        config: Electrical/timing settings.
    """
    if block is None:
        block = pyrtl.working_block()
    if config is None:
        config = LtspiceConfig()

    block = preprocess_block(
        block,
        remove_wire_nets=remove_wire_nets,
        synthesize=synthesize,
    )
    block.sanity_check()

    unsupported = sorted({net.op for net in block.logic if net.op not in SUPPORTED_OPS})
    if unsupported:
        if any(op in MEMORY_OPS for op in unsupported):
            raise NotImplementedError(
                "This design contains PyRTL memory operations "
                f"{unsupported}. The current backend supports registers but not "
                "MemBlock/RomBlock reads and writes yet."
            )
        raise NotImplementedError(
            f"This design contains unsupported PyRTL operations: {unsupported}"
        )

    output_dir = os.path.dirname(output_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    inputs = sorted(
        [w for w in block.wirevector_set if isinstance(w, pyrtl.Input)],
        key=lambda w: w.name,
    )
    outputs = sorted(
        [w for w in block.wirevector_set if isinstance(w, pyrtl.Output)],
        key=lambda w: w.name,
    )
    has_registers = any(net.op == "r" for net in block.logic)
    register_init_map = build_register_init_map(block)

    input_bits = list(iter_io_bits(inputs))
    output_bits = list(iter_io_bits(outputs))

    em = LtspiceEmitter()
    # Sorting makes generated files reproducible. SPICE element order does not
    # determine logical evaluation order.
    for net in sorted(block.logic, key=lambda n: (str(n.dests), n.op, str(n.args))):
        emit_logic_net(em, net, register_init_map, config)

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"* {os.path.basename(output_filename)}\n")
        f.write("* Generated from a PyRTL working block.\n")
        f.write("* Transistor-level cells are embedded at the end of this file.\n")
        f.write("* Supported: combinational logic, muxes, arithmetic, comparisons, and registers.\n")
        if has_registers:
            f.write("* Registers are positive-edge-triggered and share CLK.\n")

        plot_nodes = [node for _, _, node in input_bits + output_bits]
        if has_registers:
            plot_nodes.append("CLK")
        if plot_nodes:
            f.write("* Suggested plot nodes: ")
            f.write(", ".join(f"V({node})" for node in plot_nodes))
            f.write("\n")

        f.write("\n\n* 1) POWER SUPPLY\n\n")
        f.write(f"VDD_SOURCE VDD 0 {config.vdd:g}\n")

        if has_registers:
            f.write("\n* Shared positive-edge clock\n")
            f.write(
                "VCLK CLK 0 "
                f"PULSE(0 {config.vdd:g} {config.clock_delay_us:g}u "
                f"{config.rise_fall_ns:g}n {config.rise_fall_ns:g}n "
                f"{config.clock_high_us:g}u {config.clock_period_us:g}u)\n"
            )
            if config.initialize_registers:
                # High through the first rising edge, then low for the remainder
                # of the default transient simulation.
                f.write(
                    "VINIT INIT 0 "
                    f"PWL(0 {config.vdd:g} "
                    f"{config.init_release_us:g}u {config.vdd:g} "
                    f"{config.init_release_us + config.rise_fall_ns / 1000.0:g}u 0)\n"
                )

        f.write("\n\n* 2) INPUT VOLTAGE PULSES\n\n")
        input_delay = (
            config.sequential_input_delay_us
            if has_registers
            else config.combinational_input_delay_us
        )
        for index, (_, _, node) in enumerate(input_bits):
            period = config.input_base_period_us * (2 ** index)
            high_time = period / 2.0
            source_name = clean_name(f"VINPUT_{node}")
            f.write(
                f"{source_name} {node} 0 "
                f"PULSE(0 {config.vdd:g} {input_delay:g}u "
                f"{config.rise_fall_ns:g}n {config.rise_fall_ns:g}n "
                f"{high_time:g}u {period:g}u)\n"
            )

        f.write("\n\n* 3) GENERATED CIRCUIT\n\n")
        for line in em.lines:
            f.write(line + "\n")

        f.write("\n\n* Output loads\n\n")
        for _, _, node in output_bits:
            f.write(f"CLOAD_{node} {node} 0 {config.output_capacitance}\n")
            f.write(f"RLOAD_{node} {node} 0 {config.output_resistance}\n")

        f.write("\n\n* 4) SIMULATION COMMAND\n\n")
        f.write(".options reltol=0.01 abstol=1n vntol=1m method=gear gshunt=1e-12 cshunt=1f\n")
        f.write(
            f".tran 0 {config.simulation_time_us:g}u 0 "
            f"{config.max_timestep_ns:g}n uic\n"
        )

        save_nodes = [node for _, _, node in input_bits + output_bits]
        if has_registers:
            save_nodes.append("CLK")
            if config.initialize_registers:
                save_nodes.append("INIT")
        if save_nodes:
            f.write(".save ")
            f.write(" ".join(f"V({node})" for node in save_nodes))
            f.write("\n")

        write_gate_library(f)
        f.write("\n.end\n")

    print(f"Wrote LTspice file: {output_filename}")
    return output_filename


__all__ = [
    "LtspiceConfig",
    "SUPPORTED_OPS",
    "convert_working_block_to_ltspice",
    "preprocess_block",
    "write_gate_library",
]
