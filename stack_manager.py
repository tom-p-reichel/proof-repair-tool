"""
isolate stack manager
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import asyncio as aio
from coqtop import CoqProcess

type NonTrivialStdOut = Any
type NonTrivialStdErr = Any
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

    def append_stack(self, more_stack : StacksKeys):
        """
        tuple addition on my_single_stack
        """
        self.my_single_stack = self.my_single_stack + more_stack

    def can_just_add_more(self, more_stack: StacksKeys) -> Optional[int]:
        """
        self.my_single_stack is just a prefix of more_stack
        continue on from the given index in more_stack
        """
        len_my_single_stack = len(self.my_single_stack)
        if len(more_stack)<len_my_single_stack:
            return None
        is_prefix = all(
            more_stack[idx] == self.my_single_stack[idx]
            for idx in range(len_my_single_stack))
        if is_prefix:
            return len_my_single_stack
        return None

    async def run_fallible_command(self, proofscript: str | StackManagerEvaluatesItem) -> \
        Tuple[Optional[NonTrivialStdOut],Optional[NonTrivialStdErr]]:
        """
        the type signature of coq.run is complicated by dependence on return_stderr flag
        so hide that from StackManager, so it gets both
        stdout and stderr each able to be annotated as needed
        """
        stdout, stderr = await self.my_coq_process.run(proofscript, return_stderr=True)
        return stdout, stderr

    async def run_infallible_command(self, proofscript: str | StackManagerEvaluatesItem) -> \
        Optional[NonTrivialStdOut]:
        """
        the type signature of coq.run is complicated by dependence on return_stderr flag
        so hide that from StackManager, so it gets just the stdout
        annotated as needed, ignoring stderror
        """
        stdout = await self.my_coq_process.run(proofscript, return_stderr=False)
        return stdout

#pylint:disable=too-many-instance-attributes
class StackManager():
    """
    can do multiple evaluates
    all have the same prefix
    and the memoization is coordinated among them
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

    async def __evaluate_hard_work(self, stack : StacksKeys) -> \
        Optional[StacksValues]:
        """
        grab an unlocked context from self.ctxs
            with the CoqProcess therein and it's context stack
            continue or restart back to only the prefix
            in the end of the run, that gives some output that is put into self.stacks
            assuming no error
        None when the process running the commands in stack gave an error in stderr
        """
        async with self.sema:
            try:
                ctx = next(
                    (x for x in self.ctxs if not x.my_lock.locked())
                )
            except StopIteration as e:
                # the semaphore should have ensured that
                # at least one of self.ctxs was not locked
                raise e
            async with ctx.lock:
                #pylint:disable=pointless-string-statement
                """
                for a in stack:
                    # POISON. if we lose track of sentence counts the entire context breaks.
                    if len(ROUGH_SENTENCE_PATTERN.findall(a)) > 1:
                        return None
                """
                if (where_to_continue := ctx.can_just_add_more(stack)) is not None:
                    for j,new_command in enumerate(stack[where_to_continue:]):
                        _stdout,stderr = await ctx.run_fallible_command(new_command)
                        if "Error" in stderr:
                            ctx.append_stack(stack[where_to_continue:where_to_continue+j])
                            self.stacks[stack] = None
                            break
                    else:
                        # we successfully added all the commands
                        ctx.set_stack(stack)
                        self.stacks[stack] = await ctx.run_infallible_command("Show.")
                else:
                    _stdout,stderr = await ctx.run_fallible_command(
                        f"BackTo {self.offset}.\n"+\
                            "\n".join(f"timeout 1 {x}" if x[0].islower() else x  for x in stack))
                    output = await ctx.run_infallible_command("Show.")

                    #pylint:disable=no-else-return
                    if "Error" in stderr:
                        self.stacks[stack]=None
                        ctx.set_stack(())
                        _ = await ctx.run_infallible_command(f"BackTo {self.offset}.")
                        return None
                    else:
                        ctx.set_stack(stack)
                        self.stacks[stack] = output
            # end async with ctx.lock
        # end async with the semaphore

    async def evaluate(self,
                       stack : StacksKeys | List[StackManagerEvaluatesItem]) -> \
                        Optional[StacksValues]:
        """
        if already have stack in self.stacks then return the associated value
        if not grab an unlocked context from self.ctxs
            with the CoqProcess therein and it's context stack
            continue or restart back to only the prefix
            in the end of the run, that gives some output that is put into self.stacks
            assuming no error
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

        await self.__evaluate_hard_work(stack)

        return self.stacks[stack]
