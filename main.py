"""
python {this script} make
Makefile of where above done has the relevant .v Coq file
"""
# Standard/General Purpose Imports
from typing import Any, List, Optional, Tuple, cast, Sequence
import argparse
import asyncio as aio
import logging
import os
import re
import subprocess
from pathlib import Path

# Coq Specific Imports
import prism.util.alignment as align_module
from prism.util.alignment import Alignment, RightMatch
from prism.language.heuristic.parser import HeuristicParser, CoqSentence, CoqComment
from prism.util.build_tools.strace import CoqContext, strace_build
from prism.util.opam import OpamSwitch
from coqtop import CoqProcess

DO_LOGGING = False
NEED_COQ_PARSER = False
PROOF_ENDINGS = ["Qed","Abort","Defined","Save","Admitted"]

def setup_scriptargs() -> argparse.Namespace:
    """
    TODO
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd",action="store")
    parser.add_argument("--clean",default=None)
    return parser.parse_args()

script_args = setup_scriptargs()

if NEED_COQ_PARSER:
    coq_parser = HeuristicParser()

#pylint:disable=invalid-name
logger = None
if DO_LOGGING:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

def clean() -> None:
    """
    TODO
    """
    if script_args.clean is None:
        return
    subprocess.run(script_args.clean.split(), check=False)

def try_build(command : str) -> Tuple[bool,List[CoqContext]]:
    """
    assuming all .v files are below this directory...
    TODO
    """
    args_pattern = re.compile(".*(" + \
                              "|".join(re.escape(x.stem) for x in Path().rglob("*.v")) + \
                                ").*")
    out_contexts,out_return_code,_out_stdout,_out_stderr  = \
        strace_build(OpamSwitch(), command,regex=args_pattern, check=False)

    builds = [x for x in out_contexts if x.executable.endswith("coqc")]

    return out_return_code==0,builds

type NonTrivialStdError = Any
type BadStepThruReturns = Tuple[CoqSentence,Optional[NonTrivialStdError]]
type StepThruReturns = Optional[BadStepThruReturns]

async def step_thru(my_flags,my_sentences : List[CoqSentence]) -> StepThruReturns:
    """
    TODO
    """
    coq = CoqProcess(*my_flags.split())
    done = True
    is_defined = False
    for idx_sentence,cur_sentence in enumerate(my_sentences):
        if not (done or is_defined):
            if any(cur_sentence.text.strip().startswith(x) for x in PROOF_ENDINGS):
                done = True
            elif cur_sentence.text.strip()[0].isupper():
                print("potentially missed proof ending:",cur_sentence)
            continue

        _stdout,stderr = await coq.run(cur_sentence.text,return_stderr=True)
        stderr = cast(Optional[NonTrivialStdError],stderr)

        done = await coq.done()

        if not (done or is_defined):
            # rising edge
            for jdx_sentence in range(idx_sentence+1,len(my_sentences)):
                if any(
                    my_sentences[jdx_sentence].text.strip().startswith(x) for
                    x in PROOF_ENDINGS):
                    is_defined = my_sentences[jdx_sentence].text.strip().startswith("Defined")
                    break
            if not is_defined:
                _stdout = await coq.run("Admitted.")

        is_defined = is_defined and not done

        if "Error:" in stderr:
            coq.close()
            return cur_sentence, stderr

    if not done:
        coq.close()
        raise ValueError("not done at end of file!")

    coq.close()
    return None

async def get_broken_proof(flags : str,
                           sentences : List[CoqSentence]) -> \
                            Optional[Tuple[int,int]]:
    """
    TODO
    """
    coq = CoqProcess(*flags.split())
    done = True
    last_proof = None
    for idx_sentence,cur_sentence in enumerate(sentences):
        _stdout, stderr = await coq.run(cur_sentence.text,return_stderr=True)

        if cur_sentence.text in ["+","-","*","{","}"]:
            continue
        if (await coq.done()) != done:
            if done:
                # don't include proof definition
                last_proof = idx_sentence+1
                # don't include Proof. or Proof using ....
                if sentences[idx_sentence+1].text.startswith("Proof"):
                    last_proof += 1
                done = not done
            else:
                if any(cur_sentence.text.startswith(x) for x in PROOF_ENDINGS):
                    last_proof = None
                    done = not done

        if "Error:" in stderr:
            break

    if last_proof is None:
        return None

    for jdx_sentence in range(last_proof+1, len(sentences)):
        if any(sentences[jdx_sentence].text.startswith(x) for x in  PROOF_ENDINGS):
            return (last_proof,jdx_sentence)

    return None

type SentencesComments = List[CoqSentence] | Tuple[List[CoqSentence],List[CoqComment]]

def structural_error_from_err_file(last_out : CoqContext) -> \
    Tuple[str,str,List[CoqSentence],StepThruReturns]:
    """
    TODO
    """
    err_file = last_out.target
    if not err_file.endswith(".v"):
        err_file = err_file + ".v"
    print("looking at",err_file)

    sentences = \
        cast(List[CoqSentence],
             HeuristicParser.parse_sentences_from_file(
                 err_file, return_locations=True, glom_proofs=False))

    flags = str(last_out.serapi_options)

    structural_error = aio.run(step_thru(flags, sentences))
    return err_file, flags, sentences, structural_error

def fixup_structural_error(structural_error : BadStepThruReturns,
                           err_file: str) -> Tuple[bool,List[CoqContext]]:
    """
    let the user edit the file at the place of the structural error
    then redo the try_build
    """
    err_sentence, err_text = structural_error
    print("error found, but not in proof. please fix:")
    print(err_text)
    _ = input()
    os.system(f"vim +{err_sentence.location.lineno+1} {err_file}")
    success, out = try_build(script_args.cmd)
    return success, out

def make_new_proof(out_ints: Tuple[int,int],
                   sentences : List[CoqSentence],
                   diff : str,
                   flags : str) -> Optional[Sequence[CoqSentence]]:
    """
    do repair_proof with run_multiple so as soon as any one of them
    produce a sequence of CoqSentence's as a nontrivial
    RepairProofReturn, then we can stop all the other tasks still trying
    to find answer to that same question
    """
    #pylint:disable=import-outside-toplevel
    from repair import repair_proof, RepairProofReturn
    from general import run_multiple
    i, j = out_ints
    try:
        new_proof : Optional[RepairProofReturn] = aio.run(
            aio.wait_for(
                fut=run_multiple(1,repair_proof,sentences,i,j,diff,flags,aio.Lock()),
                timeout=300
                )
            )
    except aio.TimeoutError:
        new_proof = None
    return new_proof

#pylint:disable=too-many-locals
def main_running_loop() -> None:
    """
    TODO
    """
    full_alignment  = align_module.align_factory(
        calign=lambda x,y,align=align_module : \
            align.fast_edit_distance(x.text,y),
        cskip=lambda x: \
            len(x if isinstance(x,str) else x.text),
        numba=False)
    success, out = try_build(script_args.cmd)
    while not success:
        err_file, flags, sentences, structural_error = structural_error_from_err_file(out[-1])

        if structural_error is not None:
            success, out = fixup_structural_error(structural_error, err_file)
            continue

        print("ok, running repair")
        diff = subprocess.run(["git","diff"],stdout=subprocess.PIPE,check=False).stdout.decode()
        out_ints : Optional[Tuple[int,int]] = aio.run(get_broken_proof(flags, sentences))
        if out_ints is None:
            # this happens sometimes. something something parallel rebuilds.
            print("didn't find the error in the last compiled file. trying to rebuild again.")
            print("kill this process if we get stuck in a loop, please.")
            success, out = try_build(script_args.cmd)
            continue
        i,j = out_ints

        new_proof = make_new_proof(out_ints, sentences, diff, flags)

        print(new_proof)
        if new_proof is not None:
            align = cast(Alignment[CoqSentence],
                         full_alignment(sentences[i:j], new_proof, return_cost = False))
            new_part = cast(RightMatch[CoqSentence],
                            (None,"(* This proof was automatically repaired. *)"))
            align = cast(Alignment[CoqSentence],[new_part] + align)
        else:
            align = cast(Alignment,[(None,"(*")] + \
                [(x,x.text) for x in sentences[i:j]] + \
                    [(sentences[j],"*)\nAdmitted.")])

        offset = 0

        #pylint:disable=unspecified-encoding
        with open(err_file,"r") as f:
            file_contents = f.read()

        write_pos = sentences[i-1].location.end_charno+1
        for (x,y) in align:
            if y is None:
                y = ""
            if x is None:
                file_contents = file_contents[:write_pos] + f" {y}" + file_contents[write_pos:]
                write_pos += len(y)+1
                offset += len(y)+1
                continue
            if x is not None and y is not None and x.text==y:
                _xp = x.text.strip().split()
                _yp = y.strip().split()
                if x.text.strip().split() == y.strip().split():
                    write_pos = x.location.end_charno+1+offset
                    continue

            file_contents = file_contents[:x.location.beg_charno+offset] + y + \
                file_contents[x.location.end_charno+1+offset:]
            offset += len(y) - (x.location.end_charno + 1 - x.location.beg_charno)
            write_pos = x.location.end_charno+1+offset

        #pylint:disable=unspecified-encoding
        with open(err_file,"w") as f:
            f.write(file_contents)

        success, out = try_build(script_args.cmd)

    print("build succeeded.")

main_running_loop()
