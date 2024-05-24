import argparse

import subprocess

from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("cmd",action="store")

parser.add_argument("--clean",default=None)

args = parser.parse_args()


from prism.util.opam import OpamSwitch
from prism.util.build_tools.strace import strace_build, _EXECUTABLE, _REGEX
from prism.language.heuristic.parser import HeuristicParser
from coqtop import CoqProcess

coq_parser = HeuristicParser()


import logging
import asyncio as aio
import os
import re


#logging.getLogger().setLevel(logging.DEBUG)

def clean():
    if args.clean is None:
        return

    subprocess.run(args.clean.split())



def try_build(cmd):
    # assuming all .v files are below this directory...
    args = re.compile(".*(" + "|".join(re.escape(x.stem) for x in Path().rglob("*.v")) + ").*")
    out = strace_build(OpamSwitch(), cmd,regex=args, check=False)
    
    builds = [x for x in out[0] if x.executable.endswith("coqc")]

    return out[2]==0,builds


proof_endings = ["Qed","Abort","Defined","Save","Admitted"]

async def step_thru(flags,sentences):
    coq = CoqProcess(*flags.split())
    done = True
    for i,s in enumerate(sentences):
        if not done:
            if any(s.text.strip().startswith(x) for x in proof_endings):
                done = True
            elif s.text.strip()[0].isupper():
                print("potentially missed proof ending:",s)
            continue

        stdout,stderr = await coq.run(s.text,return_stderr=True)

        done = await coq.done()

        if not done:
            await coq.run("Admitted.")

        if "Error:" in stderr:
            coq.close()
            return s, stderr

    if not done:
        coq.close()
        raise ValueError("not done at end of file!")
    
    coq.close()
    return None


async def get_broken_proof(flags,sentences):
    coq = CoqProcess(*flags.split())
    done = True
    last_proof = None
    for i,s in enumerate(sentences):
        stdout, stderr = await coq.run(s.text,return_stderr=True)

        if s.text in ["+","-","*","{","}"]:
            continue
        if (await coq.done()) != done:
            if done:
                # don't include proof definition
                last_proof = i+1
                # don't include Proof. or Proof using ....
                if sentences[i+1].text.startswith("Proof"):
                    last_proof += 1
                done = not done
            else:
                if any(s.text.startswith(x) for x in proof_endings):
                    last_proof = None
                    done = not done
        


        if "Error:" in stderr:
            break

    if last_proof is None:
        return None

    for j in range(last_proof+1, len(sentences)):
        if any(sentences[j].text.startswith(x) for x in  proof_endings):
            return (last_proof,j)

    return None


success, out = try_build(args.cmd)

from repair import repair_proof

import random

while not success:
    err_file = out[-1].target
    if not err_file.endswith(".v"):
        err_file = err_file + ".v"

    print("looking at",err_file)

    sentences = HeuristicParser.parse_sentences_from_file(err_file, return_locations=True, glom_proofs=False)
    
    flags = str(out[-1].serapi_options)

    structural_error = aio.run(step_thru(flags, sentences))

    if structural_error is not None:
        err_sentence, err_text = structural_error
        print("error found, but not in proof. please fix:")
        print(err_text)
        input()
        os.system(f"vim +{err_sentence.location.lineno+1} {err_file}")
        success, out = try_build(args.cmd)
        continue
    else:
        print("ok, running repair")
        diff = subprocess.run(["git","diff"],stdout=subprocess.PIPE).stdout.decode()
        out = aio.run(get_broken_proof(flags, sentences)) 
        if out is None:
            # this happens sometimes. something something parallel rebuilds.
            print("didn't find the error in the last compiled file. trying to rebuild again.")
            print("kill this process if we get stuck in a loop, please.")
            continue
        aio.run(repair_proof(sentences,i,j,diff,flags))
        break


    


print("build succeeded.")

