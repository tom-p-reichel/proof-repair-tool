"""
functions not specific to Coq or proofs
such as for general async
"""
from typing import Callable, DefaultDict, List, Optional, Protocol,\
    Sequence, SupportsIndex, TypeVar, cast
from collections import defaultdict
from random import choices as random_choices
import re

import asyncio as aio

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import softmax as torch_softmax
from torch.types import Number

T = TypeVar("T")
async def run_multiple(how_many_times : SupportsIndex,
                       f_to_do : Callable[...,Optional[T]],
                       *args,**kwargs) -> Optional[T]:
    """
    schedule f_to_do how_many_times with the same arguments and keyword arguments
    in one path they all return None when they finish and this entire thing returns None
        after all that
    if any of them finish with something else, then cancel everything that is still going
        and return that instead
    """
    tasks : List[aio.Task[Optional[T]]] = []
    for _ in range(how_many_times):
        tasks.append(aio.create_task(f_to_do(*args,**kwargs)))

    while len(tasks)>0:
        done,tasks = await aio.wait(tasks,return_when=aio.FIRST_COMPLETED)
        for cur_done in done:
            tmp = await cur_done
            if tmp is not None:
                for cur_ongoing in tasks:
                    cur_ongoing.cancel()
                return tmp

    return None


type ProbsKey = Number # an integral index into logits
type ProbsValue = Number # a probability in [0,1]
# the type information of x.item() is not enough
# to say that one should be integral
# and the later unit interval valued
def process_logits(logits : Tensor,
                   temperature=0.6,
                   topk=100) -> DefaultDict[ProbsKey,ProbsValue]:
    """
    get the top k values in logits and their indices
    consider that as energies of that system
    and give the Boltzmann weights on those indices
    """
    tmp = torch.topk(logits,topk)
    probs : DefaultDict[ProbsKey,ProbsValue] = defaultdict(lambda:cast(ProbsValue,0.0))
    probs.update(
        zip(
            map(lambda x: x.item(),tmp.indices),
            map(lambda x: x.item(),
                torch_softmax(tmp.values/temperature))))
    return probs

EXCESSIVE_WHITESPACE_PATTERN = re.compile(r"(\s)\s+")
def simplify_whitespace(s : str) -> str:
    """
    get rid of the boomer 2 spaces after a period (legacy of typewriter education) and similar
    """
    return EXCESSIVE_WHITESPACE_PATTERN.sub(" ",s.strip())

_T_co = TypeVar("_T_co", covariant=True)
class SupportsLenAndGetItem(Protocol[_T_co]):
    """
    bound needed by random.choices
    """
    def __len__(self) -> int: ...
    def __getitem__(self, k: int, /) -> _T_co: ...

T = TypeVar("T")
def one_random_choice(population: SupportsLenAndGetItem[T],
                      weights : Optional[Sequence[float]] = None,
                      cum_weights: Sequence[float] | None = None) -> T:
    """
    first from random.choices
    entries of weights and cum_weights sequences should should also be allowed
        to be Fraction in addition to float, but that type definition is missing
        though it appears as the annotation in some places
    """
    return random_choices(population,weights=weights,cum_weights=cum_weights,k=1)[0]

def unseen_test(x_array: np._ArrayLike[float],s : float) -> float:
    """
    TODO
    """
    if np.sum(x_array)+s > 1.0:
        return 0.0
    #pylint:disable=invalid-name
    C = np.cumsum(np.concatenate([[0],x_array[:-1]]))
    return np.prod((1-C-s)/(1-C))
