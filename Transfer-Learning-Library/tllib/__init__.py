from . import alignment
from . import self_training
from . import utils
from . import vision
from . import modules
__version__ = '0.4'
__all__ = ['alignment', 'self_training', 'utils', 'vision', 'modules']
# NOTE(JMDA adaptation): trimmed unused subpackages (translation/regularization/ranking)
# which were not downloaded for this minimal reproduction.
