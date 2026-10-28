# JMDA only needs the official PDA trainer. Avoid importing unrelated trainers
# and their optional dependencies when this package is used as a vendored lib.
from .da.pda import PDA


