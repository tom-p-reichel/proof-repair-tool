"""
isolate stack manager
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast
import asyncio as aio
from coqtop import CoqProcess

type NonTrivialStdOut = Any
type StackManagerEvaluatesItem = Any # useable as proofscript argument in coq.run
type StacksKeys = Tuple[StackManagerEvaluatesItem,...]
type StacksValues = Optional[NonTrivialStdOut]


@dataclass(repr=False,eq=False,frozen=False,match_args=False,slots=True)
class SingleContext:
    """
    A single entry in StackManager's self.ctxs
    """
    my_single_stack : StacksKeys
    my_coq_process : CoqProcess
    my_lock : aio.Lock

    def unpack(self) -> Tuple[StacksKeys,CoqProcess,aio.Lock]:
        """
        give the 3 parts
        """
        return self.my_single_stack, self.my_coq_process, self.my_lock

    def set_stack(self, new_stack : StacksKeys):
        """
        replace my_single_stack
        """
        self.my_single_stack = new_stack

#pylint:disable=too-many-instance-attributes
class StackManager():
    """
    TODO
    """
    def __init__(self,prefix : List[str],flags : str,n : int=1):
        self.stacks : Dict[StacksKeys,StacksValues] = {}
        self.n = n
        self.initialized = False
        self.prefix = prefix

        self.offset = len(prefix)+1

        self.flags = flags

        self.sema = aio.Semaphore(n)
        self.biglock = aio.Lock()

    async def __postinit__(self):
        self.initialized = True
        #pylint:disable=attribute-defined-outside-init
        self.ctxs : List[SingleContext] = []
        for _i in range(self.n):
            self.ctxs.append(SingleContext(
                my_single_stack=(),
                my_coq_process=CoqProcess(*self.flags.split(),verbose=False),
                my_lock=aio.Lock()))
            _stdout,stderr = await self.ctxs[-1].my_coq_process.run(
                "\n".join(self.prefix), return_stderr=True)

            if "Error:" in stderr:
                raise ValueError(stderr)

    async def evaluate(self,
                       stack : StacksKeys | List[StackManagerEvaluatesItem]) -> \
                        Optional[StacksValues]:
        """
        if already have stack in self.stacks then return the associated value
        if not grab an unlocked context from self.ctxs
            with the CoqProcess therein and it's context stack
            continue or restart back to only the prefix
        None when the process running the commands in stack gave an error in stderr
        """
        if isinstance(stack,list):
            stack = tuple(stack)
        if stack in self.stacks:
            return self.stacks[stack]

        if not self.initialized:
            async with self.biglock:
                if not self.initialized:
                    await self.__postinit__()

        async with self.sema:
            try:
                ctx = next(
                    (x for x in self.ctxs if not x.my_lock.locked())
                )
            except StopIteration as e:
                # the semaphore should have ensured that
                # at least one of self.ctxs was not locked
                raise e
            ctxstack,coq,ctxlock = ctx.unpack()
            async with ctxlock:
                #pylint:disable=pointless-string-statement
                """
                for a in stack:
                    # POISON. if we lose track of sentence counts the entire context breaks.
                    if len(ROUGH_SENTENCE_PATTERN.findall(a)) > 1:
                        return None
                """
                if stack[:len(ctxstack)] == ctxstack:
                    # we're just adding commands
                    for j,new_command in enumerate(stack[len(ctxstack):]):
                        _stdout,stderr = await coq.run(new_command,return_stderr=True)
                        if "Error" in stderr:
                            ctx.set_stack(ctxstack + stack[len(ctxstack):len(ctxstack)+j])
                            self.stacks[stack] = None
                            break
                    else:
                        # we successfully added all the commands
                        ctx.set_stack(stack)
                        self.stacks[stack] = cast(Optional[NonTrivialStdOut],
                                                  await coq.run("Show.",return_stderr=False))
                else:
                    _stdout,stderr = await coq.run(
                        f"BackTo {self.offset}.\n"+\
                            "\n".join(f"timeout 1 {x}" if x[0].islower() else x  for x in stack),
                            return_stderr=True)
                    output = cast(Optional[NonTrivialStdOut],
                                  await coq.run("Show.",return_stderr=False))

                    #pylint:disable=no-else-return
                    if "Error" in stderr:
                        self.stacks[stack]=None
                        ctx.set_stack(())
                        await coq.run(f"BackTo {self.offset}.")
                        return None
                    else:
                        ctx.set_stack(stack)
                        self.stacks[stack] = output

        return self.stacks[stack]
