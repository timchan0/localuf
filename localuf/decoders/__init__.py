"""Module for decoders.

Available decoders:
* Actis
* BUF
* Macar
* MWPM
* NodeBUF
* NodeUF
* Snowflake
* UF

In drawings of UF-type decoders:
* ungrown edges are invisible
* half-grown edges are dotted
* fully grown edges are solid
"""

from localuf.decoders.uf import UF
from localuf.decoders.buf import BUF
from localuf.decoders.node_uf import NodeUF
from localuf.decoders.node_buf import NodeBUF
from localuf.decoders.luf import Macar, Actis
from localuf.decoders.mwpm import MWPM
from localuf.decoders.snowflake import Snowflake