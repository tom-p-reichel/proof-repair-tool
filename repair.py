from transformers import AutoModelForCausalLM,AutoTokenizer,StoppingCriteria,StoppingCriteriaList,BitsAndBytesConfig
from typing import NamedTuple
from coqtop import CoqProcess
import more_itertools
import asyncio as aio
from pathlib import Path
import numpy as np
import math
from tqdm import tqdm
import pickle
import torch
from dataclasses import dataclass
import prism.util.alignment as  align
import random
import re
from functools import lru_cache
from collections import defaultdict
import pickle

import goodinference

prefix_alignment  = align.align_factory(lru_cache(maxsize=30000)(lambda x,y : align.fast_edit_distance(x,y)),lambda x: len(x),select_best=lambda D: (D[:,-1].argmin(),D.shape[1]-1),numba=False)


ROUGH_SENTENCE_PATTERN = re.compile(r".(?:\s|$)")


class StackManager():
    def __init__(self,prefix,flags,n=1):
        self.stacks = {}
        self.n = n
        self.initialized = False
        self.prefix = prefix

        self.offset = len(prefix)+1

        self.flags = flags

        self.sema = aio.Semaphore(n)
        self.biglock = aio.Lock()

    async def __postinit__(self):
        self.initialized = True
        self.ctxs = []
        for i in range(self.n):
            self.ctxs.append([(),CoqProcess(*self.flags.split(),verbose=False),aio.Lock()])
            stdout,stderr = await self.ctxs[-1][1].run("\n".join(self.prefix), return_stderr=True)

            if "Error:" in stderr:
                raise ValueError(stderr)


    async def evaluate(self,stack):
        if type(stack)==list:
            stack = tuple(stack)
        if stack in self.stacks:
            return self.stacks[stack]

        if not self.initialized:
            async with self.biglock:
                if not self.initialized:
                    await self.__postinit__()

        async with self.sema:
            ctx = next((x for x in self.ctxs if not x[2].locked()))
            ctxstack,coq,ctxlock = ctx
            async with ctxlock:
                """
                for a in stack:
                    # POISON. if we lose track of sentence counts the entire context breaks.
                    if len(ROUGH_SENTENCE_PATTERN.findall(a)) > 1:
                        return None
                """
                if stack[:len(ctxstack)] == ctxstack:
                    # we're just adding commands
                    for j,new_command in enumerate(stack[len(ctxstack):]):
                        stdout,stderr = await coq.run(new_command,return_stderr=True)
                        if "Error" in stderr:
                            ctx[0] = ctxstack + stack[len(ctxstack):len(ctxstack)+j]
                            self.stacks[stack] = None
                            break
                    else:
                        ctx[0] = stack
                        self.stacks[stack] = await coq.run("Show.")
                else:
                    stdout,stderr = await coq.run(f"BackTo {self.offset}.\n"+"\n".join(f"timeout 1 {x}" if x[0].islower() else x  for x in stack),return_stderr=True)
                    output = await coq.run("Show.")

                    if "Error" in stderr:
                        self.stacks[stack]=None
                        ctx[0] = ()
                        await coq.run(f"BackTo {self.offset}.")
                        return None
                    else:
                        ctx[0] = stack
                        self.stacks[stack] = output

        return self.stacks[stack]




def trim_kvs(kvs,l):
    if type(kvs)==tuple:
        return tuple(trim_kvs(x,l) for x in kvs)
    if type(kvs) == torch.Tensor:
        return kvs[:,:,:l,:]
    return kvs



def process_logits(logits,temperature=0.6,topk=100):
    tmp = torch.topk(logits,topk)
    probs = defaultdict(lambda:0)
    probs.update(zip(map(lambda x: x.item(),tmp.indices),map(lambda x: x.item(),torch.nn.functional.softmax(tmp.values/temperature))))
    return probs



from contextlib import contextmanager

@contextmanager
def get_search_model(model):
    model.set_adapter("search")
    try:
        yield model.model
    finally:
        model.set_adapter("tactic")

EXCESSIVE_WHITESPACE_PATTERN = re.compile(r"(\s)\s+")

def simplify_whitespace(s):
    return EXCESSIVE_WHITESPACE_PATTERN.sub(" ",s.strip())


if Path("/tmp/vector_cache.torch").exists():
    with open("/tmp/vector_cache_index.pk","rb") as f:
        embed_cache = (pickle.load(f), torch.load("/tmp/vector_cache.torch"))
else:
    embed_cache = ([],None)

def fetch_embeds(model,tok,thms):
    # 1. check memory cache
    global embed_cache
        
    index,embeds = embed_cache
    
    unembedded = set(thms) - set(index)

    if len(unembedded)==0:
        return embed_cache

    # 2. embed them ourselves

    unembedded = list(unembedded)
    
    vecs = goodinference.embed(model,tok,[f"Theorem {x[0].split('.')[-1]} : {simplify_whitespace(x[1])}." for x in unembedded],progress=True)

    vecs = vecs.cuda().half()

    if embed_cache[1] is None:
        embed_cache = (unembedded, vecs)
    else:
        embed_cache = (index+unembedded, torch.vstack([embeds,vecs]))

    torch.save(embed_cache[1],"/tmp/vector_cache.torch")
    with open("/tmp/vector_cache_index.pk","wb") as f:
        pickle.dump(embed_cache[0],f)
    
    current_thms = set(thms)

    keep = [(x in current_thms) for x in embed_cache[0]]
    
    keep_order = [x for x in embed_cache[0] if x in current_thms]
    
    return keep_order, embed_cache[1][keep]

def unseen_test(X,s):
    if np.sum(X)+s > 1.0:
        return 0.0
    C = np.cumsum(np.concatenate([[0],X[:-1]]))
    return np.prod((1-C-s)/(1-C))

@torch.no_grad()
def sample(model,tokenizer,prompt,env,temperature=0.60,maxlength=1024, p = 0.01):

    # we're gonna add fake entries
    env = env.copy()

    sample_thresh = p

    index = list(env.items())

    with get_search_model(model) as search_model:
        index, vecs = fetch_embeds(search_model,tokenizer,index) # goodinference.embed(model.model,tokenizer,[f"Theorem {x[0].split('.')[-1]} : {simplify_whitespace(x[1])}." for x in index],progress=True)

    vecs = torch.nn.functional.normalize(vecs)

    logits = {}

    
    LOOKUP_TOKEN = tokenizer.convert_tokens_to_ids("<LOOKUP>")
    UNLOOKUP_TOKEN = tokenizer.convert_tokens_to_ids("</LOOKUP>")
    COLON_TOKEN = tokenizer.convert_tokens_to_ids('▁:')

    prompt_tokens = tokenizer([prompt], return_tensors="pt")
    prompt_length = len(prompt_tokens.input_ids[0])
    tmp = model(**prompt_tokens, use_cache=True)
    cache = ((), tmp.past_key_values)
    logits[()] = process_logits(tmp.logits[0,-1,:],temperature=temperature)
    
    stack = []

    removed_probs = []

    continue_p = 1.0

    # when the model looks up a definition that doesn't exist and we figure out what it meant
    # we put it in here.
    fake_env = {}

    while len(removed_probs)==0 or (continue_p := unseen_test(removed_probs,0.05)) > sample_thresh:
        if tuple(stack) not in logits:
            prefix_length = 0
            for x,y in zip(cache[0],stack):
                if x == y:
                    prefix_length += 1
                else:
                    break

            if prefix_length == len(stack):
                print("really weird at", tokenizer.decode(stack), stack)
                prefix_length -= 1


            tmp = model(torch.tensor([stack[prefix_length:]]),use_cache=True,past_key_values=trim_kvs(cache[1], prefix_length+prompt_length))
            
            logits[tuple(stack)] = process_logits(tmp.logits[0,-1,:],temperature=temperature)
            
            cache = (tuple(stack), tmp.past_key_values)

        p = logits[tuple(stack)]

        stack.append(random.choices(list(p.keys()),weights=p.values())[0])

        # model just finished typing the name+type of a theorem in the environment that we couldn't find.
        # probably not a real theorem. we'll help it out.
        if len(stack)>0 and stack[-1] == UNLOOKUP_TOKEN:
            print("attempting search")
            theorem = stack[-list(reversed(stack)).index(LOOKUP_TOKEN):-1]
            try:
                theorem_name = tokenizer.decode(theorem[:theorem.index(COLON_TOKEN)]).strip()
                theorem_type = tokenizer.decode(theorem[theorem.index(COLON_TOKEN)+1:]).strip()
            except ValueError:
                print("malformed search...")
                continue


            search = f"Theorem {theorem_name} : {theorem_type}."
            
            # OK, now we just.... actually search
            model.set_adapter("search")

            with get_search_model(model) as search_model:
                vec = goodinference.embed(search_model,tokenizer,[search])[0].cuda().half()

            model.set_adapter("tactic")

            result = torch.argmax((vecs@vec).flatten())
            
            print(search,index[result])

            fake_env[theorem_name] = index[result]

            env[theorem_name] = index[result][1]

            # ok, now we'll just forget the type we made up and pretend it was a real thing that
            # always existed in the environment. we fall directly into type resolution below.

            stack = stack[:-list(reversed(stack)).index(LOOKUP_TOKEN)+theorem.index(COLON_TOKEN)+1]





        # if the model is typing the name of a thing that really exists in the environment,
        # go ahead and supply the type for it.
        if len(stack)>0 and stack[-1] == COLON_TOKEN:
            if stack.count(LOOKUP_TOKEN)>stack.count(UNLOOKUP_TOKEN):

                theorem_name = tokenizer.decode(stack[-list(reversed(stack)).index(LOOKUP_TOKEN):-1])

                if theorem_name in env:
                    print("resolved name", theorem_name)
                    ty = re.sub(r"(\s)\s+"," ",env[theorem_name].strip())
                    new_toks = tokenizer(ty,add_special_tokens=False).input_ids + [UNLOOKUP_TOKEN]

                    for tok in new_toks:
                        probs = defaultdict(lambda:0)
                        probs[tok] = 1.0
                        logits[tuple(stack)] = probs
                        stack.append(tok)



        if len(stack)>maxlength or (tokenizer.decode(stack).endswith("\n```") and len(stack)>0):
            
            if len(stack) <= maxlength:
                tactic_string = tokenizer.decode(stack)[:-4]
                # process lookups
                replacements = {}
                for m in re.finditer(r"<LOOKUP>\s*([^\s]+) : .+?<\/LOOKUP>", tactic_string):
                    theorem_name = m.group(1) 
                    if theorem_name in fake_env:
                        theorem_name = fake_env[theorem_name][0]
                    replacements[m.group(0)] = theorem_name
                for r in replacements:
                    tactic_string = tactic_string.replace(r,replacements[r])
                yield tactic_string
            # trim off the ".\n```"
            stack = stack[:-3]
            prob = 1.0
            for i in range(len(stack)):
                prob *= logits[tuple(stack[:-i-1])][stack[-i-1]]
                logits[tuple(stack[:-i-1])][stack[-i-1]] -= prob
            removed_probs.append(prob)
            print(f"continue_p at {continue_p}")
            stack = []



def tokenize_glb_chunks(tok,chunks,m):
    """
    return `i` such that tokenizing `chunks[:i]` will not exceed `m` tokens
    """
    acc = 0
    for i,c in enumerate(chunks):
        acc += len(tok(c).input_ids)
        if acc >= m:
            return chunks[:i]
    return chunks


def mkprompt(tok,diff,proof_history,proof_pane, budget=2048):
    # we really want to include this
    
    if len(proof_pane)>5000:
        return None
    
    budget -= len(tok(proof_pane).input_ids)
    if budget <= 400 :
        return None
    
    if diff is None: # synth example
        history_budget = budget
    else:
        history_budget = budget/3
    
    proof_history = "\n".join(tokenize_glb_chunks(tok,proof_history[::-1],history_budget)[::-1])
    
    budget -= len(tok(proof_history).input_ids)
    
    if diff is not None:
        # spend the rest on the diff
        diff = "\n".join(tokenize_glb_chunks(tok,diff.split("\n"),budget))
    
    blocks = [
            f"Ongoing Proof History:\n```\n{proof_history}\n```",
            f"Proof State:\n```\n{proof_pane}\n```"
    ]
    
    if diff is not None:
        blocks = [f"Commit Diff:\n```\n{diff}\n```"] + blocks
    
    #random.shuffle(blocks)
    
    return "\n\n".join(blocks) + "\nNext Tactic:\n```\n"

bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_compute_dtype=torch.float16
)

base_dir = "/home/tpr/s2loop/"

m = AutoModelForCausalLM.from_pretrained(f"{base_dir}/base_model/",device_map="auto", use_cache=False, quantization_config=bnb_config,
        attn_implementation="flash_attention_2")

m.load_adapter("tomreichel/proofdb-HN-CLM","search")
m.disable_adapters()
m.load_adapter(f"{base_dir}/model/","tactic")
m.disable_adapters()
m.enable_adapters()
m.set_adapter("tactic")

tokenizer = AutoTokenizer.from_pretrained(f"{base_dir}/base_model/")



async def repair_proof(sentences,proof_start,proof_end,diff,flags,gpu_lock):
    coq = CoqProcess(*flags.split())
    await coq.run("\n".join(x.text for x in sentences[:proof_start]))
    env = await coq.environment()
    coq.close()

    stack_manager = StackManager([x.text for x in sentences[:proof_start]], flags)

    # assuming we won't more-than-double proof length
    stack = []

    state = await stack_manager.evaluate([])

    for _ in range((proof_end-proof_start)*2):

        print(stack)

        test = await stack_manager.evaluate(stack + ["Qed."]) 
        if test is not None:
            print("we did it!")
            print(test)
            break
        else:
            print("not done yet.")


        alignment = prefix_alignment([x.text for x in sentences[proof_start:proof_end]], stack)

        old_cnt = 0
        proof_history = []
        for j,(x,y) in enumerate(alignment):
            if x is None:
                proof_history.append(f"+  {y}")
                continue
            old_cnt += 1 # used a stmt from old proof
            if y is None:
                proof_history.append(f"-  {x}")
                continue
            if (x==y):
                proof_history.append(f"   {x}")
            else:
                proof_history.append(f"-  {x}")
                proof_history.append(f"+  {y}")
        
        recommendation = sentences[old_cnt+proof_start].text
        for s in sentences[old_cnt+proof_start:old_cnt+proof_start+3]:
            proof_history.append(f"?  {s.text}")

        if (await stack_manager.evaluate(stack+[recommendation])) is not None: 
            # ok, it looks like the original proof still works here.
            # we'll just take that line
            stack.append(sentences[old_cnt+proof_start].text)
            continue

        
        proof_pane = await stack_manager.evaluate(stack)

        print(proof_pane)

        if proof_pane is None:
            raise ValueError("previous state invalid.")
 
        # unfortunately, the fast tokenizer implementation
        # is not thread safe.
        async with gpu_lock:
            prompt = mkprompt(tokenizer,diff,proof_history,proof_pane)
        print(prompt)

        s = sample(m,tokenizer,prompt,env)

        attempts = []

        """
        "StopIteration interacts badly with generators and cannot be raised into a Future"
        """
        def workaround_next(gen):
            try:
                x = next(gen)
                return x
            except StopIteration:
                return None

        while True:
            async with gpu_lock:
                tactic = await aio.to_thread(workaround_next,s)
                if tactic is None:
                    break

            print(tactic)
            # TODO: real search logic...

            if tactic[0].isupper():
                continue

            if tactic[0].islower():
                attempt = await stack_manager.evaluate(stack + [f"progress {tactic}"])
            else:
                attempt = await stack_manager.evaluate(stack + [tactic])

            if attempt is not None:
                attempts.append(tactic)


        print(attempts)
        
        # evaluate attempts
        best = None
        best_score = None
        for tactic in attempts:
            future_score = 0
            for j in range(old_cnt+proof_start+1,proof_end):
                res = await stack_manager.evaluate(stack+[x.text for x in sentences[old_cnt+proof_start+1:j+1]])
                if res is not None:
                    future_score += 1
                else:
                    break
            else:
                res = await stack_manager.evaluate(stack + [x.text for x in sentences[old_cnt+proof_start+1:proof_end]] + ["Qed."])
                if res is not None:
                    future_score += 99999
            
            past_score, _ = prefix_alignment([x.text for x in sentences[proof_start:proof_end]],stack + [tactic], return_cost=True)

            score = (future_score,-past_score/sum(len(x) for x in stack + [tactic]))

            if best_score is None or score > best_score:
                best = tactic
                best_score = score

        print("running",best,"with score",best_score)

                
        stack.append(best)



    else:
        return None

    return stack
