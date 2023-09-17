from numbers import Number
from typing import Dict, Union, Tuple

SearchSpace = Dict[str, Dict[str, Union[Tuple, Number]]]
HyperparametersConfig = Dict[str, Union[Tuple, Number]]
